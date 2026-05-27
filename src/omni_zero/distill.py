from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import load_state, save_training_checkpoint
from .config import load_config
from .data import JsonlImageDataset, collate_batch
from .model import OmniZeroBundle, torch
from .tokenizer import HashTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Distill a teacher flow model into a faster student")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True, help="Student config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=500)
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
        collate_fn=collate_batch,
    )

    teacher = OmniZeroBundle(config).to(device).eval()
    teacher.load_state_dict(load_state(args.teacher), strict=True)
    for param in teacher.parameters():
        param.requires_grad_(False)

    student = OmniZeroBundle(config).to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.max_steps:
        for batch in loader:
            image = batch["image"].to(device)
            token_ids = batch["token_ids"].to(device)
            token_mask = batch["token_mask"].to(device)
            with torch.no_grad():
                latents = teacher.encode(image)
                noise = torch.randn_like(latents)
                timesteps = torch.rand(latents.shape[0], device=device)
                view_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
                noised = latents * (1.0 - timesteps.view(view_shape)) + noise * timesteps.view(view_shape)
                teacher_velocity = teacher(noised, timesteps, token_ids, token_mask)
            student_velocity = student(noised, timesteps, token_ids, token_mask)
            loss = torch.nn.functional.mse_loss(student_velocity, teacher_velocity)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 25 == 0:
                print(f"step={step} distill_loss={loss.item():.6f}")
            if step % args.save_every == 0 or step == args.max_steps:
                save_training_checkpoint(
                    output / f"student-{step:08d}.pt",
                    student.state_dict(),
                    config,
                    step,
                    extra={"distill_loss": float(loss.detach().cpu())},
                )
            if step >= args.max_steps:
                break
    save_training_checkpoint(output / "student-last.pt", student.state_dict(), config, step)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

