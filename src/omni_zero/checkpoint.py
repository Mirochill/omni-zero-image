from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ModelConfig


def load_state(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(target)
    if target.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install safetensors with: pip install -e '.[train]'") from exc
        return dict(load_file(str(target)))
    import torch

    payload = torch.load(target, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a state dict")
    return payload


def save_training_checkpoint(
    path: str | Path,
    state_dict: dict[str, Any],
    config: ModelConfig,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "config": config.to_dict(),
        "step": step,
        "extra": extra or {},
    }
    import torch

    torch.save(payload, target)

