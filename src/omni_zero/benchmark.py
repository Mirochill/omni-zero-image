from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .config import load_config
from .draft import DraftGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Omni Zero inference paths")
    parser.add_argument("--prompt", default="a futuristic city at sunrise")
    parser.add_argument("--mode", choices=["draft", "model"], default="draft")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    timings: list[float] = []
    if args.mode == "draft":
        engine = DraftGenerator(size=args.size)
        for idx in range(args.runs):
            start = time.perf_counter()
            image = engine.generate(args.prompt, seed=idx).image
            timings.append(time.perf_counter() - start)
    else:
        if not args.checkpoint:
            raise ValueError("model benchmark requires --checkpoint")
        from .sampler import OmniZeroPipeline, SampleOptions

        config = load_config(args.config)
        pipe = OmniZeroPipeline(config=config, checkpoint=args.checkpoint)
        for idx in range(args.runs):
            start = time.perf_counter()
            image = pipe.generate(
                SampleOptions(prompt=args.prompt, size=args.size, steps=args.steps, seed=idx)
            )
            timings.append(time.perf_counter() - start)
    if args.out:
        image.save(Path(args.out))
    payload = {
        "mode": args.mode,
        "size": args.size,
        "steps": args.steps if args.mode == "model" else 0,
        "runs": args.runs,
        "mean_seconds": sum(timings) / len(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
