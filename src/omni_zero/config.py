from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by training, inference, and export."""

    image_size: int = 512
    latent_channels: int = 4
    latent_downsample: int = 8
    text_vocab_size: int = 32768
    text_max_tokens: int = 64
    text_dim: int = 384
    model_dim: int = 384
    layers: int = 8
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0
    patch_size: int = 2

    @property
    def latent_size(self) -> int:
        if self.image_size % self.latent_downsample != 0:
            raise ValueError("image_size must be divisible by latent_downsample")
        return self.image_size // self.latent_downsample

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> ModelConfig:
    """Load a JSON config file or return the default config."""

    if path is None:
        return ModelConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a JSON object")
    return ModelConfig(**payload)


def save_config(config: ModelConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

