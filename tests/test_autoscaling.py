import asyncio
import copy
import time
import typing
from collections import Counter

import pytest
import ray
from ray.cluster_utils import AutoscalingCluster
from ray.util.state import list_actors, list_nodes
from ray.util.state.common import ActorState, NodeState

from rarg_ray_patterns.autoscaling import (
  ActorAutoscaler,
  ActorSpec,
  _class_key,
  _NodeDeployment,
)

# Maximum number of cpu_node workers the cluster will provision. The autoscaler
# pins one MonitorActor per node, so this also caps how many actors can be
# scheduled regardless of how many workers are requested. Shared with the tests.
NODE_CAP = 20

# How long to wait for the cluster to provision nodes and converge. Bringing the
# fake multi-node cluster up to full capacity (NODE_CAP node processes) is slow,
# so give the convergence-to-capacity case generous headroom.
CONVERGE_TIMEOUT = 240.0
POLL_INTERVAL = 3.0
# A second sample window used to confirm the converged count is stable and
# doesn't overshoot the requested worker count.
STABILITY_WINDOW = 6.0

AUTOSCALING_CLUSTER_CONFIG = {
  "head_resources": {"CPU": 0},
  "worker_node_types": {
    "cpu_node": {
      "resources": {
        "CPU": 1,
        "object_store_memory": 100 * 1024 * 1024,
      },
      "node_config": {},
      "min_workers": 0,
      "max_workers": NODE_CAP,
    },
  },
  "min_workers": 0,
  "max_workers": NODE_CAP * 5,
  "autoscaler_v2": True,
}


@ray.remote
class EchoActor:
  """Test actor capturing its constructor arguments."""

  def __init__(self, value: str, repeat: int = 1) -> None:
    self._value = value * repeat

  async def value(self) -> str:
    return self._value


class PlainCounter:
  """Plain (undecorated) test class; ActorSpec wraps it with ray.remote."""

  def __init__(self, start: int = 0) -> None:
    self._start = start

  def start(self) -> int:
    return self._start


@ray.remote
class Scheduler:
  def __init__(
    self,
    nworkers: int = 10,
    batch_size: int = 2,
    install_on_head: bool = False,
    label_selector: dict[str, str] | None = None,
    actor_specs: tuple[ActorSpec, ...] = (),
  ):
    self._autoscaler = ActorAutoscaler(
      nworkers, batch_size, install_on_head, label_selector, actor_specs
    )
    self._target = nworkers
    self._event = asyncio.Event()

  async def deployments(
    self, actor_cls: type | tuple[type, ...] | None = None
  ) -> dict[str, typing.Any]:
    return self._autoscaler.deployments(actor_cls)

  async def run(self) -> None:
    try:
      while not self._event.is_set():
        await asyncio.sleep(1.0)
        await self._autoscaler.autoscale(target=self._target)
    finally:
      await self._autoscaler.close()

  async def resize(self, nworkers: int) -> None:
    self._target = nworkers

  async def stop(self) -> None:
    self._event.set()


def _alive_actors(class_name: str = "MonitorActor") -> list[ActorState]:
  """Return the live ``class_name`` actors currently scheduled on the cluster."""
  # list_actors is typed as returning Any; pin the element type here.
  actors: list[ActorState] = list_actors(
    filters=[
      ("class_name", "=", class_name),
      ("state", "=", "ALIVE"),
    ],
    detail=True,
  )
  return actors


def _await_actor_count(
  expected: int, class_name: str = "MonitorActor"
) -> list[ActorState]:
  """Poll until exactly ``expected`` actors are alive, or the deadline passes."""
  deadline = time.monotonic() + CONVERGE_TIMEOUT
  actors = _alive_actors(class_name)
  while len(actors) != expected and time.monotonic() < deadline:
    time.sleep(POLL_INTERVAL)
    actors = _alive_actors(class_name)
  assert len(actors) == expected, (
    f"expected {expected} {class_name}s scheduled, "
    f"saw {len(actors)} after {CONVERGE_TIMEOUT}s"
  )
  return actors


def _head_node_id() -> str:
  """Return the cluster's head node id (the node with node:__internal_head__)."""
  return typing.cast(
    str,
    next(
      n["NodeID"]
      for n in ray.nodes()  # type: ignore[no-untyped-call]
      if n.get("Alive") and "node:__internal_head__" in n.get("Resources", {})
    ),
  )


# The cheap nworkers=8 cases stay below the node ceiling and exercise the
# install_on_head flag in both states (head excluded vs. monitored). The single
# nworkers=100 case exercises convergence to the cluster's node capacity; it is
# slow, so we run it once with the default (head excluded) rather than for both
# flag values, since the flag behaviour is already covered by the nworkers=8 pair.
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.parametrize(
  "nworkers, batch_size, install_on_head", [(8, 2, False), (8, 2, True), (10, 2, False)]
)
def test_actor_autoscaling(
  nworkers: int,
  batch_size: int,
  install_on_head: bool,
) -> None:
  schedulable = NODE_CAP + 1 if install_on_head else NODE_CAP
  expected = min(nworkers, schedulable)

  cfg = typing.cast(dict[str, typing.Any], copy.deepcopy(AUTOSCALING_CLUSTER_CONFIG))
  cfg["worker_node_types"]["cpu_node"]["max_workers"] = schedulable
  cfg["max_workers"] = schedulable * 5
  with pytest.warns(ResourceWarning, match="unclosed file"):
    cluster = AutoscalingCluster(**cfg)

  try:
    cluster.start()
    ray.init("auto", runtime_env={"excludes": ".*"})
    scheduler = Scheduler.options(num_cpus=0).remote(  # type: ignore[attr-defined]
      nworkers, batch_size, install_on_head
    )
    run_future = scheduler.run.remote()

    # Poll until the cluster has scheduled the expected number of monitor
    # actors, rather than waiting a fixed (and flaky) duration.
    deadline = time.monotonic() + CONVERGE_TIMEOUT
    monitors = _alive_actors()
    while len(monitors) < expected and time.monotonic() < deadline:
      time.sleep(POLL_INTERVAL)
      monitors = _alive_actors()

    assert len(monitors) == expected, (
      f"expected {expected} MonitorActors scheduled, "
      f"saw {len(monitors)} after {CONVERGE_TIMEOUT}s"
    )

    # Each actor must be pinned to a distinct node (one monitor per node).
    node_ids = {m.node_id for m in monitors}
    assert len(node_ids) == expected, (
      f"expected actors on {expected} distinct nodes, saw {len(node_ids)}"
    )

    # The head node is monitored only when install_on_head is set.
    head_node_id = _head_node_id()
    if install_on_head:
      assert head_node_id in node_ids, "expected a monitor on the head node"
    else:
      assert head_node_id not in node_ids, "head node should not be monitored"

    # The converged count must be stable and must never overshoot the
    # requested worker count.
    time.sleep(STABILITY_WINDOW)
    monitors = _alive_actors()
    assert len(monitors) == expected, (
      f"actor count drifted to {len(monitors)} after convergence"
    )
    assert len(monitors) <= nworkers

    ray.get(scheduler.stop.remote())
    # Surface any error from the run loop.
    ray.get(run_future)
  finally:
    ray.shutdown()
    cluster.shutdown()  # type: ignore[no-untyped-call]


def test_label_selector_reserved_key() -> None:
  """The node-id label is reserved for the autoscaler's exclusion/pinning."""
  with pytest.raises(ValueError, match="ray.io/node-id"):
    ActorAutoscaler(label_selector={"ray.io/node-id": "some-node-id"})


def test_actor_spec_normalises_actor_class() -> None:
  """A plain class is wrapped with ray.remote; non-classes are rejected."""
  spec = ActorSpec(PlainCounter)
  assert isinstance(spec.actor_class, ray.actor.ActorClass)
  # An already-decorated class passes through untouched.
  assert ActorSpec(EchoActor).actor_class is EchoActor
  with pytest.raises(TypeError, match="actor_class"):
    ActorSpec(42)


def test_actor_spec_reserved_key() -> None:
  """The node-id label is reserved for the autoscaler's node pinning."""
  with pytest.raises(ValueError, match="ray.io/node-id"):
    ActorSpec(PlainCounter, options={"label_selector": {"ray.io/node-id": "a-node"}})


def test_deployments_actor_cls_selection() -> None:
  """deployments() selects each node's handles by class, in the requested shape."""
  specs = (ActorSpec(EchoActor, args=("a",)), ActorSpec(PlainCounter))
  autoscaler = ActorAutoscaler(actor_specs=specs)
  # Fake handles: deployments() only indexes them, so sentinels suffice.
  monitor, echo, counter = (typing.cast(typing.Any, object()) for _ in range(3))
  autoscaler._deployments["node-1"] = _NodeDeployment(monitor, (echo, counter))

  # None returns every instance, in actor_specs order.
  assert autoscaler.deployments() == {"node-1": (echo, counter)}
  # A singleton class returns a bare handle; decorated (EchoActor) and plain
  # (PlainCounter) classes both resolve to their spec.
  assert autoscaler.deployments(typing.cast(type, EchoActor)) == {"node-1": echo}
  assert autoscaler.deployments(PlainCounter) == {"node-1": counter}
  # A tuple of classes returns handles in the requested order, not spec order.
  assert autoscaler.deployments((PlainCounter, typing.cast(type, EchoActor))) == {
    "node-1": (counter, echo)
  }


def test_deployments_actor_cls_errors() -> None:
  """Unknown and ambiguous classes are rejected."""
  specs = (ActorSpec(PlainCounter), ActorSpec(PlainCounter))
  autoscaler = ActorAutoscaler(actor_specs=specs)

  class Unknown:
    pass

  with pytest.raises(ValueError, match="matches 0 actor specs"):
    autoscaler.deployments(Unknown)
  with pytest.raises(ValueError, match="matches 2 actor specs"):
    autoscaler.deployments(PlainCounter)
  # actor_cls=None remains available when duplicate specs make a class ambiguous.
  assert autoscaler.deployments() == {}


# The label key constraining which node class an autoscaler manages, and how
# many nodes of each class the labelled cluster may provision.
NODE_CLASS_LABEL = "rarg.io/node-class"
CLASS_WORKERS = 3

LABELLED_CLUSTER_CONFIG = {
  "head_resources": {"CPU": 0},
  "worker_node_types": {
    "compute_node": {
      "resources": {
        "CPU": 1,
        "object_store_memory": 100 * 1024 * 1024,
      },
      "labels": {NODE_CLASS_LABEL: "compute"},
      "node_config": {},
      "min_workers": 0,
      "max_workers": CLASS_WORKERS,
    },
    "io_node": {
      "resources": {
        "CPU": 1,
        "object_store_memory": 100 * 1024 * 1024,
      },
      "labels": {NODE_CLASS_LABEL: "io"},
      "node_config": {},
      "min_workers": 0,
      "max_workers": CLASS_WORKERS,
    },
  },
  "min_workers": 0,
  "max_workers": CLASS_WORKERS * 2 * 5,
  "autoscaler_v2": True,
}


def _monitor_class_counts(monitors: list[ActorState]) -> Counter[str | None]:
  """Count monitors by the node class label of the node each is pinned to."""
  # list_nodes is typed as returning Any; pin the element type here.
  nodes: list[NodeState] = list_nodes(detail=True)
  node_labels = {n.node_id: n.labels or {} for n in nodes}
  # A monitor with no (or an unknown) node id counts under class None, which
  # any exact per-class assertion then reports loudly.
  return Counter(
    node_labels.get(m.node_id or "", {}).get(NODE_CLASS_LABEL) for m in monitors
  )


# One cluster spin-up with two labelled node classes and one autoscaler per
# class: each must converge on its own class and never claim the other's
# nodes, which also proves selector-partitioned autoscalers can coexist.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_actor_autoscaling_label_selectors() -> None:
  batch_size = 2
  node_classes = ("compute", "io")
  expected = CLASS_WORKERS * len(node_classes)

  cfg = typing.cast(dict[str, typing.Any], copy.deepcopy(LABELLED_CLUSTER_CONFIG))
  with pytest.warns(ResourceWarning, match="unclosed file"):
    cluster = AutoscalingCluster(**cfg)

  try:
    cluster.start()
    ray.init("auto", runtime_env={"excludes": ".*"})
    schedulers = [
      Scheduler.options(num_cpus=0).remote(  # type: ignore[attr-defined]
        CLASS_WORKERS, batch_size, False, {NODE_CLASS_LABEL: node_class}
      )
      for node_class in node_classes
    ]
    run_futures = [scheduler.run.remote() for scheduler in schedulers]

    monitors = _await_actor_count(expected)

    # Each actor must be pinned to a distinct node (one monitor per node).
    node_ids = {m.node_id for m in monitors}
    assert len(node_ids) == expected, (
      f"expected actors on {expected} distinct nodes, saw {len(node_ids)}"
    )

    # Every monitor must sit on a node of its autoscaler's class; with both
    # classes at their target this means an exact per-class split.
    counts = _monitor_class_counts(monitors)
    assert counts == {node_class: CLASS_WORKERS for node_class in node_classes}, (
      f"expected {CLASS_WORKERS} monitors per node class, saw {dict(counts)}"
    )

    # The converged counts must be stable: no drift and no cross-class claims.
    time.sleep(STABILITY_WINDOW)
    monitors = _alive_actors()
    counts = _monitor_class_counts(monitors)
    assert counts == {node_class: CLASS_WORKERS for node_class in node_classes}, (
      f"per-class monitor counts drifted to {dict(counts)} after convergence"
    )

    ray.get([scheduler.stop.remote() for scheduler in schedulers])
    # Surface any error from the run loops.
    ray.get(run_futures)
  finally:
    ray.shutdown()
    cluster.shutdown()  # type: ignore[no-untyped-call]


# One cluster spin-up (they are slow) exercising both directions: converge up,
# resize down, then resize back up to prove reaped nodes can be re-used. We do
# not assert that Ray reaps the idled nodes themselves - the fake cluster's
# idle timeout makes that slow and flaky; the converged actor count is the
# contract this package owns.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_actor_downscaling() -> None:
  nworkers, batch_size = 8, 2

  cfg = typing.cast(dict[str, typing.Any], copy.deepcopy(AUTOSCALING_CLUSTER_CONFIG))
  cfg["worker_node_types"]["cpu_node"]["max_workers"] = nworkers
  cfg["max_workers"] = nworkers * 5
  with pytest.warns(ResourceWarning, match="unclosed file"):
    cluster = AutoscalingCluster(**cfg)

  try:
    cluster.start()
    ray.init("auto", runtime_env={"excludes": ".*"})
    scheduler = Scheduler.options(num_cpus=0).remote(  # type: ignore[attr-defined]
      nworkers, batch_size
    )
    run_future = scheduler.run.remote()

    for target in (nworkers, 3, 6):
      ray.get(scheduler.resize.remote(target))
      monitors = _await_actor_count(target)
      node_ids = {m.node_id for m in monitors}
      assert len(node_ids) == target, (
        f"expected actors on {target} distinct nodes, saw {len(node_ids)}"
      )

      # The converged count must be stable: no drift, and in particular no
      # reprovisioning of intentionally reaped workers.
      time.sleep(STABILITY_WINDOW)
      monitors = _alive_actors()
      assert len(monitors) == target, (
        f"actor count drifted to {len(monitors)} after converging on {target}"
      )

    ray.get(scheduler.stop.remote())
    # Surface any error from the run loop.
    ray.get(run_future)
  finally:
    ray.shutdown()
    cluster.shutdown()  # type: ignore[no-untyped-call]


# One cluster spin-up proving spec actors are installed alongside the monitor:
# each managed node carries one instance of each spec with its constructor
# arguments intact, and downscaling reaps the whole per-node set.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_actor_spec_installation() -> None:
  # The spec classes are defined inside the test so cloudpickle serializes
  # them by value: the test module itself is not importable on cluster
  # workers, so a by-reference pickle of a module-level class would fail to
  # deserialize inside the Scheduler actor.
  @ray.remote
  class SpecEcho:
    def __init__(self, value: str, repeat: int = 1) -> None:
      self._value = value * repeat

    async def value(self) -> str:
      return self._value

  class SpecCounter:
    """Plain (undecorated) class; ActorSpec wraps it with ray.remote."""

    def __init__(self, start: int = 0) -> None:
      self._start = start

    def start(self) -> int:
      return self._start

  nworkers, batch_size = 3, 2
  # The state API reports an actor's class by qualname; for these nested
  # classes that is "test_actor_spec_installation.<locals>.Spec...".
  spec_classes = (_class_key(SpecEcho)[1], _class_key(SpecCounter)[1])
  specs = (
    # Also exercises options merging: num_cpus=0 rides along with the pin.
    ActorSpec(SpecEcho, args=("ping",), kwargs={"repeat": 2}, options={"num_cpus": 0}),
    # A plain class with Ray's default resource request (1 CPU).
    ActorSpec(SpecCounter, args=(7,)),
  )

  cfg = typing.cast(dict[str, typing.Any], copy.deepcopy(AUTOSCALING_CLUSTER_CONFIG))
  cfg["worker_node_types"]["cpu_node"]["max_workers"] = nworkers
  cfg["max_workers"] = nworkers * 5
  with pytest.warns(ResourceWarning, match="unclosed file"):
    cluster = AutoscalingCluster(**cfg)

  try:
    cluster.start()
    ray.init("auto", runtime_env={"excludes": ".*"})
    scheduler = Scheduler.options(num_cpus=0).remote(  # type: ignore[attr-defined]
      nworkers, batch_size, False, None, specs
    )
    run_future = scheduler.run.remote()

    monitors = _await_actor_count(nworkers)
    monitor_nodes = {m.node_id for m in monitors}

    # One instance of each spec class per monitored node.
    for class_name in spec_classes:
      actors = _await_actor_count(nworkers, class_name)
      assert {a.node_id for a in actors} == monitor_nodes, (
        f"{class_name}s not colocated with the monitors"
      )

    # Constructor arguments reached the installed actors, whose handles are
    # exposed per node in spec order.
    deployments = ray.get(scheduler.deployments.remote())
    assert set(deployments) == monitor_nodes
    for echo, counter in deployments.values():
      assert ray.get(echo.value.remote()) == "pingping"
      assert ray.get(counter.start.remote()) == 7

    # A singleton actor_cls selects one bare handle per node. The class
    # crosses the process boundary by value (a second pickle of SpecEcho), so
    # this also proves matching survives serialization.
    echoes = ray.get(scheduler.deployments.remote(SpecEcho))
    assert set(echoes) == monitor_nodes
    for echo in echoes.values():
      assert ray.get(echo.value.remote()) == "pingping"

    # A tuple of classes returns handles in the requested order, not spec order.
    reordered = ray.get(scheduler.deployments.remote((SpecCounter, SpecEcho)))
    assert set(reordered) == monitor_nodes
    for counter, echo in reordered.values():
      assert ray.get(counter.start.remote()) == 7
      assert ray.get(echo.value.remote()) == "pingping"

    # Downscaling reaps each node's whole actor set.
    target = 1
    ray.get(scheduler.resize.remote(target))
    for class_name in ("MonitorActor", *spec_classes):
      _await_actor_count(target, class_name)

    ray.get(scheduler.stop.remote())
    # Surface any error from the run loop.
    ray.get(run_future)
  finally:
    ray.shutdown()
    cluster.shutdown()  # type: ignore[no-untyped-call]
