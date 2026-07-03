=========
Changelog
=========

0.0.2 (unreleased)
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
