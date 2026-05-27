from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import save_training_checkpoint
from .config import load_config
from .data import JsonlImageDataset, collate_batch
from .model import OmniZeroBundle, torch
from .tokenizer import HashTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Omni Zero Image with latent flow matching")
    parser.add_argument("--dataset", required=True, help="JSONL file with image and caption fields")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = HashTokenizer(config.text_vocab_size, config.text_max_tokens)
    dataset = JsonlImageDataset(args.dataset, config.image_size, tokenizer)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    model = OmniZeroBundle(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    step = 0
    model.train()
    while step < args.max_steps:
        for batch in loader:
            image = batch["image"].to(device)
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            with torch.no_grad():
                target_latents = model.encode(image)
            noise = torch.randn_like(target_latents)
            timesteps = torch.rand(target_latents.shape[0], device=device)
            view_shape = (target_latents.shape[0],) + (1,) * (target_latents.ndim - 1)
            t_view = timesteps.view(view_shape)
            noisy_latents = target_latents * (1.0 - t_view) + noise * t_view
            target_velocity = noise - target_latents
            predicted = model(noisy_latents, timesteps, token_ids, token_mask)
            loss = torch.nn.functional.mse_loss(predicted, target_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 25 == 0:
                print(f"step={step} loss={loss.item():.6f}")
            if step % args.save_every == 0 or step == args.max_steps:
                save_training_checkpoint(
                    output / f"checkpoint-{step:08d}.pt",
                    model.state_dict(),
                    config,
                    step,
                    extra={"loss": float(loss.detach().cpu())},
                )
            if step >= args.max_steps:
                break
    save_training_checkpoint(output / "last.pt", model.state_dict(), config, step)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

