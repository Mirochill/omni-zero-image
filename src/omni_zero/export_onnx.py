from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_state
from .config import load_config
from .model import OmniZeroBundle, torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Omni Zero flow denoiser to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    bundle = OmniZeroBundle(config).eval()
    bundle.load_state_dict(load_state(args.checkpoint), strict=True)
    latent = torch.randn(1, config.latent_channels, config.latent_size, config.latent_size)
    timestep = torch.ones(1)
    token_ids = torch.zeros(1, config.text_max_tokens, dtype=torch.long)
    token_mask = torch.ones(1, config.text_max_tokens, dtype=torch.bool)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        bundle.flow,
        (latent, timestep, token_ids, token_mask),
        str(target),
        input_names=["latents", "timesteps", "token_ids", "token_mask"],
        output_names=["velocity"],
        dynamic_axes={
            "latents": {0: "batch"},
            "timesteps": {0: "batch"},
            "token_ids": {0: "batch"},
            "token_mask": {0: "batch"},
            "velocity": {0: "batch"},
        },
        opset_version=args.opset,
    )
    print(target)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

