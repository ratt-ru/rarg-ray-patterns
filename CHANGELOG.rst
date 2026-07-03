=========
Changelog
=========

0.0.2 (unreleased)
------------------

* Support downscaling in ``ActorAutoscaler``: ``autoscale`` accepts a
  ``target`` argument that retargets the worker count at runtime, and surplus
  ``MonitorActor``\s are reaped so their nodes can be released back to the Ray
  autoscaler.

0.0.1 (03-07-2026)
------------------

* Initial release.
