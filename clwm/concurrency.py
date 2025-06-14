"""Concurrency helpers for background CPU/GPU work.

This module centralises threading as well as CUDA *stream* handling logic
that is reused across the project.  A small abstraction layer keeps the
implementation lightweight while providing a clean public API.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Iterable, Optional

import torch


# -------------------------------------------------------------------------
# Threading helpers -------------------------------------------------------
# -------------------------------------------------------------------------


class AsyncExecutor:
    """Run *fn* in a dedicated background thread.

    The executor is **fire-and-forget** - once started the underlying thread
    stays alive until the provided callable returns *or* the instance is
    explicitly :pyfunc:`stop`ped.  The abstraction is intentionally minimal
    so that switching to a standard ``ThreadPoolExecutor`` in the future
    remains straightforward.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *args: Any,
        daemon: bool = True,
        **kwargs: Any,
    ) -> None:
        self._stop_event = threading.Event()

        def _target():
            fn(*args, **kwargs, stop_event=self._stop_event)  # type: ignore[misc]

        self._thread = threading.Thread(target=_target, daemon=daemon)

    # ------------------------------------------------------------------
    # Lifecycle ---------------------------------------------------------
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background thread (idempotent)."""

        if not self._thread.is_alive():
            self._thread.start()

    def stop(
        self, join: bool = False, timeout: Optional[float] = None
    ) -> None:
        """Signal the worker to terminate and optionally *join* it."""

        self._stop_event.set()
        if join:
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------
    # Introspection -----------------------------------------------------
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:  # noqa: D401 – property helper
        return self._thread.is_alive()


# -------------------------------------------------------------------------
# CUDA stream helpers -----------------------------------------------------
# -------------------------------------------------------------------------


class StreamManager:
    """Utility that owns a *private* CUDA stream for H→D pre-fetching."""

    __slots__ = ("stream",)

    def __init__(self, device: int | str | torch.device | None = None):
        device = torch.cuda.current_device() if device is None else device
        self.stream = torch.cuda.Stream(device=device)

    # ------------------------------------------------------------------
    # Convenience wrappers ---------------------------------------------
    # ------------------------------------------------------------------

    def run(self, fn, *args, **kwargs):
        """Execute *fn* on the owned CUDA stream and return its result.

        The helper blocks the *caller* stream afterwards so the returned
        tensors are always safe to use without additional synchronisation
        calls.
        """

        if self.stream is None:
            return fn(*args, **kwargs)

        with torch.cuda.stream(self.stream):
            out = fn(*args, **kwargs)

        torch.cuda.current_stream().wait_stream(self.stream)
        return out

    def to_device(self, tensor: torch.Tensor, device, **to_kwargs):
        """Asynchronous H→D copy guarded by the private stream."""

        if self.stream is None:
            return tensor.to(device, **to_kwargs)

        with torch.cuda.stream(self.stream):
            gpu_tensor = tensor.to(device, non_blocking=True, **to_kwargs)

        torch.cuda.current_stream().wait_stream(self.stream)
        return gpu_tensor

    # ------------------------------------------------------------------
    # Context management ------------------------------------------------
    # ------------------------------------------------------------------

    def __enter__(self):
        self._prev_stream = torch.cuda.current_stream()
        torch.cuda.set_stream(self.stream)

        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401
        torch.cuda.set_stream(self._prev_stream)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def wait_for(self, *streams: "StreamManager | torch.cuda.Stream") -> None:  # type: ignore[name-defined]
        """Block *this* stream until *streams* have completed."""

        if self.stream is None:
            return  # CPU-only run – nothing to synchronise

        for other in streams:
            s = other.stream if isinstance(other, StreamManager) else other
            if s is not None:
                self.stream.wait_stream(s)
