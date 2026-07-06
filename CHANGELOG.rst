=========
Changelog
=========

0.0.3 (unreleased)
------------------

* Breaking: ``ActorAutoscaler.deployments`` is a method rather than a
  property. It accepts an ``actor_cls`` argument selecting which of each
  node's spec actors to return: a single class returns ``{node_id: actor}``,
  a tuple of classes returns ``{node_id: (actor, ...)}`` in the requested
  order, and ``None`` (the default) returns every instance in ``actor_specs``
  order (:pr:`4`).
* ``ActorAutoscaler`` accepts ``actor_specs``: a sequence of ``ActorSpec``\s
  describing an actor class, its constructor arguments and its
  ``.options(...)`` overrides. One instance of each spec is installed on
  every managed node alongside its ``MonitorActor``, pinned to the node and
  reaped with it (:pr:`4`).

0.0.2 (03-07-2026)
------------------

* Support a ``label_selector`` argument in ``ActorAutoscaler`` constraining
  which nodes it may manage, so multiple autoscalers with disjoint selectors
  can each handle a different node class in the same cluster (:pr:`3`).
* Support downscaling in ``ActorAutoscaler``: ``autoscale`` accepts a
  ``target`` argument that retargets the worker count at runtime, and surplus
  ``MonitorActor``\s are reaped so their nodes can be released back to the Ray
  autoscaler. (:pr:`2`)

0.0.1 (03-07-2026)
------------------

* Initial release.
