# Benchmarks

Benchmarks must be generated from real commands and committed with hardware,
commit, checkpoint hash, and seed. Placeholder scores are not acceptable.

## Latency Commands

Draft path:

```bash
python -m omni_zero.benchmark \
  --mode draft \
  --prompt "a futuristic city at sunrise" \
  --size 512 \
  --runs 10 \
  --out outputs/benchmark-draft.png
```

Model path:

```bash
python -m omni_zero.benchmark \
  --mode model \
  --checkpoint checkpoints/omnizero-student.safetensors \
  --config configs/omnizero-base.json \
  --prompt "a futuristic city at sunrise" \
  --size 1024 \
  --steps 4 \
  --runs 10 \
  --out outputs/benchmark-model.png
```

## Metrics To Report

- Mean latency.
- P50/P95 latency.
- Peak VRAM/RAM.
- Image size.
- Number of denoising steps.
- Checkpoint hash.
- Runtime backend: PyTorch eager, ONNX Runtime, TensorRT, WebGPU, etc.

## Quality Metrics

Automated metrics are useful but insufficient:

- CLIPScore or similar prompt-image alignment.
- Aesthetic predictor score.
- OCR accuracy for typography prompts.
- LPIPS or masked-region difference for edits.
- Identity/reference consistency metrics for reference tasks.

Human preference testing is required before comparing to commercial systems.

