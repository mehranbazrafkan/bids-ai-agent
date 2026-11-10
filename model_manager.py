"""Manages model lifecycle: lazy load, use, offload to free VRAM."""

from __future__ import annotations

import gc
from typing import Any, Callable, Optional

import torch


class ModelManager:
    """Loads models on demand and offloads them to free VRAM.

    Only one model is kept in VRAM at a time. When a different model is
    requested, the current one is offloaded first.
    """

    def __init__(self):
        self._loaded: dict[str, Any] = {}
        self._current: Optional[str] = None

    def get(
        self,
        key: str,
        loader: Callable[[], Any],
    ) -> Any:
        """Return the model identified by *key*, loading it if necessary.

        *key* is opaque: callers that can load the same model with different
        settings should include those settings in it (see
        ``llm_client.LLM._load_signature``), so a setting change is a cache
        miss and reloads instead of reusing the wrong variant.
        """
        if key in self._loaded and self._current == key:
            return self._loaded[key]

        self.offload_current()

        model = loader()
        self._loaded[key] = model
        self._current = key
        return model

    def offload_current(self) -> None:
        """Offload the currently loaded model to free VRAM."""
        if self._current is None:
            return
        model = self._loaded.pop(self._current, None)
        del model
        self._current = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def offload_all(self) -> None:
        """Offload every loaded model."""
        self._loaded.clear()
        self._current = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Module-level singleton.
manager = ModelManager()
