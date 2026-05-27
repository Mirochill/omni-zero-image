# Omni Zero Image

Omni Zero Image is an open research prototype for fast text-to-image and
image-to-image generation. It is designed around a small rectified-flow latent
model, a reference/edit conditioning path, and a zero-checkpoint draft renderer
that keeps the CLI functional before expensive training.

This repository does not claim parity with Nano Banana Pro, GPT Image, FLUX,
Imagen, Midjourney, or other large proprietary systems. Matching those systems
requires large-scale curated data, heavy distillation, human evaluation, safety
work, and remote GPU validation. The code here provides the architecture,
training path, export path, and verification harness needed to build toward
that target without pretending that an untrained checkpoint is already a
frontier model.

## What Is Implemented

- Text-to-image CLI with two execution modes:
  - `draft`: instant CPU renderer with no checkpoint, useful for smoke tests and
    examples of the end-to-end interface.
  - `model`: PyTorch latent rectified-flow sampler using a checkpoint.
- Image-to-image/edit CLI with optional mask support.
- Prompt hashing tokenizer with no hosted dependency.
- Compact autoencoder, DiT-style flow core, reference encoder, and mask-aware
  edit path.
- Training script for captioned image datasets.
- Distillation script skeleton for turning a multi-step teacher into a 1-4 step
  student.
- ONNX export entry point for deployment.
- Tests that validate core shapes, prompt determinism, draft generation, and
  image editing contracts.
- GitHub Actions workflow templates for remote CPU validation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,torch,train]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,torch,train]"
```

## Quick Start

Generate an instant draft image without model weights:

```bash
omnizero generate \
  --prompt "a coherent futuristic city at sunrise, glass towers, warm light" \
  --out outputs/city.png \
  --mode draft \
  --size 768
```

Edit an existing image with the draft engine:

```bash
omnizero edit \
  --image examples/reference.png \
  --prompt "make the lighting warmer and add a cinematic blue sky" \
  --out outputs/edit.png \
  --mode draft
```

Run with a trained checkpoint:

```bash
omnizero generate \
  --prompt "a detailed product photo of a transparent wireless speaker" \
  --out outputs/product.png \
  --mode model \
  --checkpoint checkpoints/omnizero-student.safetensors \
  --steps 4 \
  --guidance 3.5
```

## Train

Prepare a JSONL dataset:

```jsonl
{"image":"data/images/000001.jpg","caption":"a sharp studio photo of a red chair"}
{"image":"data/images/000002.jpg","caption":"an isometric city block with trees"}
```

Start a small remote smoke train:

```bash
python -m omni_zero.train \
  --dataset data/train.jsonl \
  --config configs/omnizero-tiny.json \
  --output runs/tiny-smoke \
  --max-steps 1000
```

The model architecture is intentionally scalable. The tiny config is for CI and
shape checks only. A frontier-quality model would need a much larger config,
curated licensed data, preference tuning, and distillation.

## Architecture Summary

Omni Zero Image uses:

- a compact convolutional autoencoder for latent images;
- a prompt hashing tokenizer for reproducible offline conditioning;
- a DiT-style rectified-flow core that predicts velocity in latent space;
- a reference encoder for image-to-image conditioning;
- mask-aware latent blending for local edits;
- consistency distillation hooks for 1-4 step inference;
- optional quantization/export path for ONNX and TensorRT style runtimes.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

## Validation Status

Current status: prototype code and contracts are present. The untrained model
path should be treated as a software verification target, not as an art-quality
checkpoint. Quality claims require remote training and evaluation.

Remote checks can run through GitHub Actions after copying the templates from
`workflow-templates/` into `.github/workflows/` with a GitHub token that has the
`workflow` scope:

```bash
pytest
ruff check .
```

See [docs/VALIDATION.md](docs/VALIDATION.md) for the completion checklist and
what evidence is still required before claiming model quality.

## Benchmark Targets

The intended deployment target is:

- draft mode: sub-second CPU execution;
- distilled student: 1-4 denoising steps;
- local GPU: interactive latency for 1024 px images after quantization;
- CPU/WebGPU: useful previews, not guaranteed frontier quality.

Benchmarks must be generated from real runs and should not be copied from
marketing claims. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Repository Layout

```text
src/omni_zero/       Python package
configs/            model configs
tests/              contract tests
docs/               architecture and validation docs
examples/           prompt manifests and reference placeholders
workflow-templates/ GitHub Actions templates
```

## License

MIT. Training data and third-party checkpoints must carry their own compatible
licenses.
