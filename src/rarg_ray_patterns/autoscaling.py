import asyncio
import inspect
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, cast

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rarg_ray_patterns.utils import wrap_future

KEEPALIVE_DURATION = 60.0

# The label key Ray uses to address a specific node; reserved by the
# autoscaler for its own node exclusion and pinning.
NODE_ID_LABEL = "ray.io/node-id"

# Sentinel distinguishing "head node id not yet looked up" from "no head found".
_UNSET = object()


@ray.remote
class MonitorActor:
  def __init__(self) -> None:
    self._heartbeat = asyncio.Event()

  async def heartbeat(self) -> None:
    """Block forever; the awaiter sees RayActorError when the actor dies."""
    await self._heartbeat.wait()

  async def stop(self) -> None:
    self._heartbeat.set()


@ray.remote
def keep_alive(timeout: float) -> None:
  """Sleep for timeout in order to keep the worker (and node) alive"""
  time.sleep(timeout)


@ray.remote(num_cpus=0, num_returns=2)
def discover_node() -> tuple[Any, Any]:
  """Return this node's id and a keep-alive ref pinned to the same node.

  The keep-alive task takes 1 CPU on *this* node specifically (via
  NodeAffinity), so the autoscaler will not reap the node before the caller
  has a chance to schedule a long-lived actor on it.
  """
  node_id = ray.get_runtime_context().get_node_id()
  keepalive_ref = keep_alive.options(
    num_cpus=1,
    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False),
  ).remote(KEEPALIVE_DURATION)
  return node_id, keepalive_ref


@dataclass(frozen=True)
class ActorSpec:
  """An actor installed on each node an :class:`ActorAutoscaler` manages.

  Args:
    actor_class: The actor's class: either ``@ray.remote``-decorated or a
      plain class, which is wrapped with ``ray.remote`` on construction.
    args: Positional constructor arguments.
    kwargs: Keyword constructor arguments.
    options: Extra ``.options(...)`` applied at install time, e.g.
      ``num_cpus`` or ``max_restarts``. A ``label_selector`` may constrain
      the actor further, but the ``ray.io/node-id`` key is reserved for the
      autoscaler's node pinning.

  The whole spec, ``args`` and ``kwargs`` included, must be Ray-serializable:
  it is typically built in one process and installed from another (e.g. an
  autoscaler running inside an actor).

  The autoscaler does not watch installed actors for crashes; only the node's
  ``MonitorActor`` drives reprovisioning. For crash resilience pass
  ``max_restarts`` in ``options`` — restarted actors keep their node pin.
  """

  actor_class: Any
  args: tuple[Any, ...] = ()
  kwargs: dict[str, Any] = field(default_factory=dict)
  options: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if not isinstance(self.actor_class, ray.actor.ActorClass):
      if not inspect.isclass(self.actor_class):
        raise TypeError(
          f"actor_class must be an actor or plain class, got {self.actor_class!r}"
        )
      # Frozen dataclass; bypass the immutability to normalise the class.
      object.__setattr__(self, "actor_class", ray.remote(self.actor_class))
    if NODE_ID_LABEL in (self.options.get("label_selector") or {}):
      raise ValueError(
        f"'{NODE_ID_LABEL}' is reserved for the ActorAutoscaler's node "
        "pinning; constrain placement with other labels"
      )


class _NodeDeployment(NamedTuple):
  """One managed node's actors: the monitor plus one instance per spec."""

  monitor: ray.actor.ActorHandle[Any]
  actors: tuple[ray.actor.ActorHandle[Any], ...]


def _class_key(actor_cls: Any) -> tuple[str, str]:
  """A pickle-stable identity for a user class: its module and qualname.

  Matching classes by identity would break across process boundaries —
  cloudpickle's by-value copies are distinct class objects — so requested
  classes are matched to specs by provenance instead.
  """
  if isinstance(actor_cls, ray.actor.ActorClass):
    # The stubs don't describe the metadata carrying the wrapped class.
    modified = actor_cls.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    actor_cls = getattr(modified, "__ray_actor_class__", modified)
  return (actor_cls.__module__, actor_cls.__qualname__)


class ActorAutoscaler:
  """Maintain a per-node set of actors on ``nworkers`` cluster nodes.

  Each managed node carries a ``MonitorActor`` — the keepalive and death
  sentinel that drives (re)provisioning — plus one instance of each spec in
  ``actor_specs``, all pinned to the node and reaped together.

  Args:
    nworkers: Initial target number of monitored nodes; ``autoscale(target=...)``
      retargets it.
    batch_size: Maximum number of node discoveries requested per ``autoscale``
      call.
    install_on_head: Whether the head node may be monitored. When a
      ``label_selector`` is given, the head is additionally eligible only if it
      matches the selector.
    label_selector: Ray label selector constraining which nodes this autoscaler
      may manage, e.g. ``{"rarg.io/node-class": "compute"}``. Values use Ray's
      selector syntax verbatim (``in(a,b)``, ``!x``, ...). The
      ``ray.io/node-id`` key is reserved for the autoscaler's own node
      exclusion and pinning. Multiple autoscalers can share a cluster, each
      managing its own node class; their selectors must be disjoint — each
      autoscaler excludes only nodes *it* monitors, so overlapping selectors
      can claim the same node twice.
    actor_specs: Actors installed on each managed node: one instance of each
      spec per node, alongside the node's ``MonitorActor``. Their combined
      resource requests must fit the node class, or the surplus actors queue
      forever while the monitor holds the node.
  """

  def __init__(
    self,
    nworkers: int = 10,
    batch_size: int = 2,
    install_on_head: bool = False,
    label_selector: dict[str, str] | None = None,
    actor_specs: Sequence[ActorSpec] = (),
  ):
    if label_selector and NODE_ID_LABEL in label_selector:
      raise ValueError(
        f"'{NODE_ID_LABEL}' is reserved for the ActorAutoscaler's node "
        "exclusion and pinning; constrain node classes with other labels"
      )
    self._nworkers: int = nworkers
    self._batch_size: int = batch_size
    self._install_on_head: bool = install_on_head
    self._label_selector: dict[str, str] = dict(label_selector or {})
    self._actor_specs: tuple[ActorSpec, ...] = tuple(actor_specs)
    # Resolved lazily on first autoscale(); cached because the head node id is
    # stable for the cluster's lifetime. _UNSET until looked up.
    self._head_node_id: Any = _UNSET
    self._deployments: dict[str, _NodeDeployment] = {}
    # node-id future -> (node-id ObjectRef, keepalive ObjectRef)
    self._pending: dict[
      asyncio.Future[Any], tuple[ray.ObjectRef[Any], ray.ObjectRef[Any]]
    ] = {}
    # heartbeat futures held to keep them from being GC'd
    self._heartbeats: set[asyncio.Future[Any]] = set()
    self._closed: bool = False

  def _spec_index(self, actor_cls: type) -> int:
    key = _class_key(actor_cls)
    matches = [
      i
      for i, spec in enumerate(self._actor_specs)
      if _class_key(spec.actor_class) == key
    ]
    if len(matches) != 1:
      raise ValueError(
        f"actor_cls {key[1]!r} matches {len(matches)} actor specs; select a "
        "class matching exactly one spec, or actor_cls=None for all instances"
      )
    return matches[0]

  def deployments(
    self, actor_cls: type | tuple[type, ...] | None = None
  ) -> dict[str, Any]:
    """Installed spec actors per managed node.

    Args:
      actor_cls: Selects which of each node's spec actors to return. A single
        class returns ``{node_id: actor}``; a tuple of classes returns
        ``{node_id: (actor, ...)}`` in the requested order; ``None`` (the
        default) returns every instance in ``actor_specs`` order. Classes are
        matched by module and qualname, so copies that crossed a process
        boundary still resolve; each must match exactly one spec.
    """
    if actor_cls is None:
      return {nid: d.actors for nid, d in self._deployments.items()}
    if isinstance(actor_cls, tuple):
      indices = tuple(self._spec_index(cls) for cls in actor_cls)
      return {
        nid: tuple(d.actors[i] for i in indices) for nid, d in self._deployments.items()
      }
    index = self._spec_index(actor_cls)
    return {nid: d.actors[index] for nid, d in self._deployments.items()}

  async def close(self) -> None:
    self._closed = True

    # Cancel pending discovery tasks; collect their futures so we can drain
    # them below. Without that drain, asyncio warns "Future exception was
    # never retrieved" when the TaskCancelledError-bearing futures are GC'd.
    pending_futures = list(self._pending.keys())
    for node_ref, keepalive_ref in self._pending.values():
      ray.cancel(node_ref, force=True)
      ray.cancel(keepalive_ref, force=True)
    self._pending.clear()

    # Kill each node's actor set. The monitors' heartbeat futures will then
    # resolve with RayActorError; same drain rule applies.
    heartbeats = list(self._heartbeats)
    self._heartbeats.clear()
    for deployment in self._deployments.values():
      self._kill_deployment(deployment)
    self._deployments.clear()

    if pending_futures or heartbeats:
      await asyncio.gather(*pending_futures, *heartbeats, return_exceptions=True)

  def _resolve_head_node_id(self) -> str | None:
    """Return the head node's id (cached), or None if no head node is found.

    The head node is the live node carrying Ray's auto-resource
    ``node:__internal_head__``.
    """
    if self._head_node_id is _UNSET:
      self._head_node_id = next(
        (
          n["NodeID"]
          for n in ray.nodes()  # type: ignore[no-untyped-call]
          if n.get("Alive") and "node:__internal_head__" in n.get("Resources", {})
        ),
        None,
      )
    return self._head_node_id  # type: ignore[no-any-return]

  async def autoscale(self, timeout: float = 1.0, target: int | None = None) -> None:
    # A target retargets the autoscaler; None keeps the current one. The
    # constructor's nworkers is thus only the initial target.
    if target is not None:
      self._nworkers = target

    # Drain completed discoveries first so capacity decisions reflect them.
    if self._pending:
      ready, _ = await asyncio.wait(
        self._pending.keys(),
        return_when="ALL_COMPLETED",
        timeout=timeout,
      )
      self._handle_ready(ready)

    # If we hit capacity (e.g. nworkers shrunk, or a burst overshot),
    # cancel surplus pending work instead of letting it leak extra nodes,
    # and reap surplus live monitors so their nodes can be released.
    if len(self._deployments) >= self._nworkers:
      for node_ref, keepalive_ref in self._pending.values():
        ray.cancel(node_ref, force=True)
        ray.cancel(keepalive_ref, force=True)
      self._pending.clear()
      if (surplus := len(self._deployments) - self._nworkers) > 0:
        self._reap(surplus)
      print(f"Autoscaler at capacity: {list(self._deployments)}")
      return

    shortfall = self._nworkers - len(self._deployments) - len(self._pending)
    requested = min(shortfall, self._batch_size)
    if requested <= 0:
      return

    # Keep discovery off already-monitored nodes and, unless explicitly enabled,
    # off the head node too.
    excluded = set(self._deployments)
    if not self._install_on_head:
      if (head := self._resolve_head_node_id()) is not None:
        excluded.add(head)

    # Selector keys AND together: discovery lands only on nodes of the
    # requested class that aren't already monitored.
    selector = dict(self._label_selector)
    if excluded:
      selector[NODE_ID_LABEL] = f"!in({','.join(excluded)})"

    options: dict[str, Any] = {"num_cpus": 0, "scheduling_strategy": "SPREAD"}
    if selector:
      options["label_selector"] = selector

    print(f"Requesting {requested} nodes")
    for _ in range(requested):
      # num_returns=2 makes .remote() yield a 2-tuple of ObjectRefs, but the
      # type stubs only describe the single-return case.
      node_ref, keepalive_ref = discover_node.options(**options).remote()  # type: ignore[misc]
      future = wrap_future(node_ref)
      self._pending[future] = (node_ref, keepalive_ref)

  @staticmethod
  def _kill_deployment(deployment: _NodeDeployment) -> None:
    ray.kill(deployment.monitor)
    for spec_actor in deployment.actors:
      ray.kill(spec_actor)

  def _install(self, spec: ActorSpec, node_id: str) -> ray.actor.ActorHandle[Any]:
    """Instantiate one spec actor pinned to ``node_id``."""
    options = dict(spec.options)
    options["label_selector"] = {
      **(options.get("label_selector") or {}),
      NODE_ID_LABEL: node_id,
    }
    return cast(
      "ray.actor.ActorHandle[Any]",
      spec.actor_class.options(**options).remote(*spec.args, **spec.kwargs),
    )

  def _reap(self, count: int) -> None:
    head = self._resolve_head_node_id()
    # Kill worker-node deployments first: their nodes can be released by the
    # Ray autoscaler, whereas the head node persists regardless, so its
    # deployment (when install_on_head) is the cheapest one to keep.
    victims = sorted(self._deployments, key=lambda nid: nid == head)[:count]
    for node_id in victims:
      # Popping before the kill marks the death as intentional, so
      # _on_actor_dead won't treat it as a crash. The node id also leaves the
      # discovery exclusion set, letting the node be re-used if the target
      # grows again.
      self._kill_deployment(self._deployments.pop(node_id))
    print(f"Reaped {len(victims)} workers: {victims}")

  def _handle_ready(self, ready: set[asyncio.Future[Any]]) -> None:
    new_nodes = []
    for future in ready:
      _, keepalive_ref = self._pending.pop(future)

      if future.cancelled():
        ray.cancel(keepalive_ref, force=True)
        continue

      if exc := future.exception():
        print(f"node discovery failed: {exc!r}")
        ray.cancel(keepalive_ref, force=True)
        continue

      node_id = future.result()
      if node_id in self._deployments:
        # Already monitored; release the keepalive so the node can return
        # to its existing actor's exclusive use.
        ray.cancel(keepalive_ref, force=True)
        continue

      if not self._install_on_head and node_id == self._resolve_head_node_id():
        # Discovery raced ahead of head-node resolution and landed on the head
        # node; drop it and release the keepalive.
        ray.cancel(keepalive_ref, force=True)
        continue

      # Pin the node's long-lived actor set — the monitor plus one instance of
      # each spec — to this node. The autoscaler treats the pinned actors as
      # demand on this node and keeps the node alive.
      monitor = MonitorActor.options(  # type: ignore[attr-defined]
        num_cpus=0,
        label_selector={NODE_ID_LABEL: node_id},
      ).remote()
      spec_actors = tuple(self._install(spec, node_id) for spec in self._actor_specs)
      self._deployments[node_id] = _NodeDeployment(monitor, spec_actors)
      new_nodes.append(node_id)

      # Free the CPU so the queued actors can claim it. The autoscaler treats
      # the queued actors as resource demand and will not reap the node.
      ray.cancel(keepalive_ref, force=True)

      # Watch for actor death so we'll reprovision next iteration.
      heartbeat = wrap_future(monitor.heartbeat.remote())
      self._heartbeats.add(heartbeat)
      heartbeat.add_done_callback(self._heartbeats.discard)
      heartbeat.add_done_callback(
        lambda f, nid=node_id, actor=monitor: self._on_actor_dead(nid, actor, f)  # type: ignore[misc]
      )

    if new_nodes:
      print(f"Procured {len(new_nodes)} nodes: {new_nodes}")

  def _on_actor_dead(
    self, node_id: str, actor: ray.actor.ActorHandle[Any], future: asyncio.Future[Any]
  ) -> None:
    # Always retrieve the exception so asyncio doesn't warn about it.
    exc = None if future.cancelled() else future.exception()
    if self._closed:
      return
    deployment = self._deployments.get(node_id)
    if deployment is None or deployment.monitor is not actor:
      # Reaped by downscaling (or already replaced); not a crash.
      return
    print(f"Worker on {node_id} died ({exc!r}); will reprovision")
    self._deployments.pop(node_id, None)
    # If only the monitor crashed, the node — and its spec actors — may have
    # survived; kill them so reprovisioning doesn't install duplicates.
    for spec_actor in deployment.actors:
      ray.kill(spec_actor)
