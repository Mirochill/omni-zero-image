# Model Card

## Model Name

Omni Zero Image

## Version

0.1.0 prototype

## Intended Use

- Research into low-cost image generation.
- Text-to-image and image-to-image editing experiments.
- Distillation and deployment experiments.
- Integration tests for image generation user interfaces.

## Out-of-Scope Use

- Production image generation from the untrained checkpoint.
- Claims of parity with commercial frontier models without benchmark evidence.
- Generating illegal, harmful, deceptive, or rights-violating content.

## Architecture

- Latent autoencoder.
- Hash tokenizer.
- DiT-style rectified-flow denoiser.
- Reference/image-edit conditioning.
- Mask-aware latent preservation.

## Training Data

No dataset is included. Users must provide licensed data and comply with all
dataset terms.

## Limitations

- No trained checkpoint is included.
- Draft mode is deterministic procedural rendering, not the neural model.
- The tokenizer is minimal.
- Safety systems are not included.

## Evaluation

Evaluation must include both automated metrics and human review. See
`docs/VALIDATION.md`.

