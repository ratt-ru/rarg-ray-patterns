import asyncio
import time
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rarg_ray_patterns.utils import wrap_future

KEEPALIVE_DURATION = 60.0

# Sentinel distinguishing "head node id not yet looked up" from "no head found".
_UNSET = object()


@ray.remote
class MonitorActor:
  async def heartbeat(self) -> None:
    """Block forever; the awaiter sees RayActorError when the actor dies."""
    await asyncio.Event().wait()


@ray.remote  # type: ignore[misc]
def keep_alive(timeout: float) -> None:
  """Sleep for timeout in order to keep the worker (and node) alive"""
  time.sleep(timeout)


@ray.remote(num_cpus=0, num_returns=2)  # type: ignore[misc]
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


class ActorAutoscaler:
  def __init__(
    self, nworkers: int = 10, batch_size: int = 2, install_on_head: bool = False
  ):
    self._nworkers: int = nworkers
    self._batch_size: int = batch_size
    self._install_on_head: bool = install_on_head
    # Resolved lazily on first autoscale(); cached because the head node id is
    # stable for the cluster's lifetime. _UNSET until looked up.
    self._head_node_id: Any = _UNSET
    self._workers: dict[str, ray.actor.ActorHandle] = {}
    # node-id future -> (node-id ObjectRef, keepalive ObjectRef)
    self._pending: dict[asyncio.Future[Any], tuple[ray.ObjectRef, ray.ObjectRef]] = {}
    # heartbeat futures held to keep them from being GC'd
    self._heartbeats: set[asyncio.Future[Any]] = set()
    self._closed: bool = False

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

    # Kill monitor actors. Their heartbeat futures will then resolve with
    # RayActorError; same drain rule applies.
    heartbeats = list(self._heartbeats)
    self._heartbeats.clear()
    for worker in self._workers.values():
      ray.kill(worker)
    self._workers.clear()

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
          for n in ray.nodes()
          if n.get("Alive") and "node:__internal_head__" in n.get("Resources", {})
        ),
        None,
      )
    return self._head_node_id  # type: ignore[no-any-return]

  async def autoscale(self, timeout: float = 1.0) -> None:
    # Drain completed discoveries first so capacity decisions reflect them.
    if self._pending:
      ready, _ = await asyncio.wait(
        self._pending.keys(),
        return_when="ALL_COMPLETED",
        timeout=timeout,
      )
      self._handle_ready(ready)

    # If we hit capacity (e.g. nworkers shrunk, or a burst overshot),
    # cancel surplus pending work instead of letting it leak extra nodes.
    if len(self._workers) >= self._nworkers:
      for node_ref, keepalive_ref in self._pending.values():
        ray.cancel(node_ref, force=True)
        ray.cancel(keepalive_ref, force=True)
      self._pending.clear()
      print(f"Autoscaler at capacity: {list(self._workers)}")
      return

    shortfall = self._nworkers - len(self._workers) - len(self._pending)
    requested = min(shortfall, self._batch_size)
    if requested <= 0:
      return

    # Keep discovery off already-monitored nodes and, unless explicitly enabled,
    # off the head node too.
    excluded = set(self._workers)
    if not self._install_on_head:
      if (head := self._resolve_head_node_id()) is not None:
        excluded.add(head)

    options: dict[str, Any] = {"num_cpus": 0, "scheduling_strategy": "SPREAD"}
    if excluded:
      options["label_selector"] = {"ray.io/node-id": f"!in({','.join(excluded)})"}

    print(f"Requesting {requested} nodes")
    for _ in range(requested):
      node_ref, keepalive_ref = discover_node.options(**options).remote()
      future = wrap_future(node_ref)
      self._pending[future] = (node_ref, keepalive_ref)

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
      if node_id in self._workers:
        # Already monitored; release the keepalive so the node can return
        # to its existing actor's exclusive use.
        ray.cancel(keepalive_ref, force=True)
        continue

      if not self._install_on_head and node_id == self._resolve_head_node_id():
        # Discovery raced ahead of head-node resolution and landed on the head
        # node; drop it and release the keepalive.
        ray.cancel(keepalive_ref, force=True)
        continue

      # Pin a long-lived monitor actor to this node. num_cpus=1 makes it the
      # node's permanent CPU consumer so the autoscaler keeps the node alive.
      monitor = MonitorActor.options(  # type: ignore[attr-defined]
        num_cpus=0,
        label_selector={"ray.io/node-id": node_id},
      ).remote()
      self._workers[node_id] = monitor
      new_nodes.append(node_id)

      # Free the CPU so the queued actor can claim it. The autoscaler treats
      # the queued actor as resource demand and will not reap the node.
      ray.cancel(keepalive_ref, force=True)

      # Watch for actor death so we'll reprovision next iteration.
      heartbeat = wrap_future(monitor.heartbeat.remote())
      self._heartbeats.add(heartbeat)
      heartbeat.add_done_callback(self._heartbeats.discard)
      heartbeat.add_done_callback(lambda f, nid=node_id: self._on_actor_dead(nid, f))  # type: ignore[misc]

    if new_nodes:
      print(f"Procured {len(new_nodes)} nodes: {new_nodes}")

  def _on_actor_dead(self, node_id: str, future: asyncio.Future[Any]) -> None:
    # Always retrieve the exception so asyncio doesn't warn about it.
    exc = None if future.cancelled() else future.exception()
    if self._closed:
      return
    print(f"Worker on {node_id} died ({exc!r}); will reprovision")
    self._workers.pop(node_id, None)
