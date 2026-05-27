from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .config import load_config
from .draft import DraftGenerator
from .image_io import save_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="omnizero", description="Omni Zero Image CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate an image from text")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--out", required=True)
    generate.add_argument("--mode", choices=["auto", "draft", "model"], default="auto")
    generate.add_argument("--checkpoint")
    generate.add_argument("--config")
    generate.add_argument("--size", type=int, default=512)
    generate.add_argument("--steps", type=int, default=8)
    generate.add_argument("--guidance", type=float, default=3.0)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--allow-random", action="store_true")

    edit = sub.add_parser("edit", help="Edit an image using text and optional mask")
    edit.add_argument("--image", required=True)
    edit.add_argument("--prompt", required=True)
    edit.add_argument("--out", required=True)
    edit.add_argument("--mask")
    edit.add_argument("--mode", choices=["auto", "draft", "model"], default="auto")
    edit.add_argument("--checkpoint")
    edit.add_argument("--config")
    edit.add_argument("--size", type=int, default=512)
    edit.add_argument("--steps", type=int, default=8)
    edit.add_argument("--guidance", type=float, default=3.0)
    edit.add_argument("--strength", type=float, default=0.65)
    edit.add_argument("--seed", type=int, default=0)
    edit.add_argument("--allow-random", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            image = _generate(args)
        elif args.command == "edit":
            image = _edit(args)
        else:  # pragma: no cover
            parser.error("unknown command")
            return 2
        target = save_image(image, args.out)
        print(target)
        return 0
    except Exception as exc:
        print(f"omnizero: error: {exc}", file=sys.stderr)
        return 1


def _resolve_mode(mode: str, checkpoint: str | None) -> str:
    if mode == "auto":
        return "model" if checkpoint else "draft"
    return mode


def _generate(args: argparse.Namespace):
    mode = _resolve_mode(args.mode, args.checkpoint)
    if mode == "draft":
        return DraftGenerator(size=args.size).generate(args.prompt, seed=args.seed).image
    if not args.checkpoint and not args.allow_random:
        raise ValueError("model mode requires --checkpoint or --allow-random for software smoke tests")
    from .sampler import OmniZeroPipeline, SampleOptions

    config = load_config(args.config)
    pipe = OmniZeroPipeline(config=config, checkpoint=args.checkpoint)
    return pipe.generate(
        SampleOptions(
            prompt=args.prompt,
            size=args.size,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
        )
    )


def _edit(args: argparse.Namespace):
    mode = _resolve_mode(args.mode, args.checkpoint)
    if mode == "draft":
        return DraftGenerator(size=args.size).edit(
            image_path=Path(args.image),
            prompt=args.prompt,
            mask_path=Path(args.mask) if args.mask else None,
            strength=args.strength,
            seed=args.seed,
        ).image
    if not args.checkpoint and not args.allow_random:
        raise ValueError("model mode requires --checkpoint or --allow-random for software smoke tests")
    from .sampler import OmniZeroPipeline, SampleOptions

    config = load_config(args.config)
    pipe = OmniZeroPipeline(config=config, checkpoint=args.checkpoint)
    return pipe.edit(
        image_path=Path(args.image),
        mask_path=Path(args.mask) if args.mask else None,
        options=SampleOptions(
            prompt=args.prompt,
            size=args.size,
            steps=args.steps,
            guidance=args.guidance,
            seed=args.seed,
            strength=args.strength,
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
