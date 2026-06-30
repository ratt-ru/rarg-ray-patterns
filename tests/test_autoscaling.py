import asyncio
import copy
import time
import typing

import pytest
import ray
from ray.cluster_utils import AutoscalingCluster
from ray.util.state import list_actors
from ray.util.state.common import ActorState

from rarg_ray_patterns.autoscaling import ActorAutoscaler

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
class Scheduler:
  def __init__(
    self, nworkers: int = 10, batch_size: int = 2, install_on_head: bool = False
  ):
    self._autoscaler = ActorAutoscaler(nworkers, batch_size, install_on_head)
    self._event = asyncio.Event()

  async def run(self) -> None:
    try:
      while not self._event.is_set():
        await asyncio.sleep(1.0)
        await self._autoscaler.autoscale()
    finally:
      await self._autoscaler.close()

  async def stop(self) -> None:
    self._event.set()


def _alive_monitors() -> list[ActorState]:
  """Return the live MonitorActor entries currently scheduled on the cluster."""
  # list_actors is typed as returning Any; pin the element type here.
  actors: list[ActorState] = list_actors(
    filters=[
      ("class_name", "=", "MonitorActor"),
      ("state", "=", "ALIVE"),
    ],
    detail=True,
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
  "nworkers, batch_size, install_on_head", [(8, 5, False), (8, 5, True), (20, 5, False)]
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
    monitors = _alive_monitors()
    while len(monitors) < expected and time.monotonic() < deadline:
      time.sleep(POLL_INTERVAL)
      monitors = _alive_monitors()

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
    monitors = _alive_monitors()
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
