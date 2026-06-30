from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import ray


def wrap_future(obj_ref: ray.ObjectRef[Any]) -> asyncio.Future[Any]:
  """Wrap a ray ObjectReference as an asyncio Future"""
  return asyncio.wrap_future(obj_ref.future())
