"""Hard, guaranteed timeout for blocking calls.

Model SDKs (notably LiteLLM streaming) do not always enforce their own request
timeout — a stuck connection can block a worker thread forever, which in turn
stalls the whole evaluation (the evaluator waits on that future indefinitely).

``run_with_timeout`` runs a callable on a throwaway worker thread and enforces a
hard wall-clock deadline on the *caller*. If the deadline passes, it raises
:class:`HardTimeout` and abandons the worker thread (``shutdown(wait=False)``);
the leaked thread is reclaimed when the process exits. This guarantees the
caller is never blocked past ``timeout_s`` regardless of the underlying SDK.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class HardTimeout(Exception):
    """Raised when a wrapped call exceeds its hard client-side deadline."""


def run_with_timeout(fn: Callable[..., T], timeout_s: float, *args: Any, **kwargs: Any) -> T:
    """Run ``fn(*args, **kwargs)`` but never block longer than ``timeout_s``.

    A non-positive ``timeout_s`` disables the guard and calls ``fn`` directly.
    Exceptions raised by ``fn`` propagate unchanged; only a deadline breach is
    converted to :class:`HardTimeout`.
    """
    if not timeout_s or timeout_s <= 0:
        return fn(*args, **kwargs)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        raise HardTimeout(
            f"call exceeded hard client timeout of {timeout_s:.0f}s "
            "(endpoint unresponsive / stream stalled)"
        ) from exc
    finally:
        # Never wait on a potentially hung call; the leaked thread dies at process exit.
        executor.shutdown(wait=False)
