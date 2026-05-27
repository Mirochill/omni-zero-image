# Architecture

Omni Zero Image is built as a low-cost latent rectified-flow image model with a
draft renderer for instant software validation.

## Goals

- One interface for text-to-image and image-to-image editing.
- 1-4 step inference after distillation.
- Small enough to quantize and deploy on consumer hardware.
- Offline tokenizer and local inference path.
- Clear separation between software functionality and trained model quality.

## Pipeline

```text
prompt -> hash tokenizer -> text conditioner
image  -> autoencoder    -> latent reference path
mask   -> latent mask    -> edit conditioning

noise/init latent + t + conditions -> rectified-flow denoiser -> latent velocity
latent solver -> decoded RGB image
```

## Components

### Hash Tokenizer

`HashTokenizer` turns prompt tokens into stable integer IDs with BLAKE2 hashing.
It has no remote vocabulary dependency, so the repository can be tested offline.
For a production frontier model, this should be replaced or augmented with a
learned multilingual text encoder.

### Autoencoder

`TinyAutoencoder` compresses RGB images by 8x into a 4-channel latent grid. This
keeps denoising cheap. The current implementation is intentionally compact for
tests; a production checkpoint should train a stronger perceptual autoencoder
with reconstruction, adversarial, and perceptual losses.

### Rectified-Flow Core

`OmniZeroModel` predicts latent velocity rather than denoised pixels. This makes
the model compatible with flow matching and consistency distillation:

```text
x_t = (1 - t) * clean_latent + t * noise
target_velocity = noise - clean_latent
```

During inference, an Euler solver moves from noise toward the image manifold.
The final student target is 1-4 steps.

### Reference/Edit Conditioning

Image-to-image mode encodes the input image into latents and passes those
latents through a reference encoder. Optional edit masks are pooled into patch
tokens and also used to preserve unmasked latent regions during sampling.

### Draft Renderer

`DraftGenerator` is not the neural model. It is a deterministic CPU renderer
that keeps:

- CLI smoke tests usable without a checkpoint;
- sample generation possible in remote CI without GPUs;
- UI integrations testable before training.

## Scaling Plan

The base config is a target shape, not a validated frontier checkpoint:

- Tiny: CI and local shape tests.
- Base: first serious training target.
- Distilled: 1-4 step student checkpoint.
- Quantized: ONNX/TensorRT/WebGPU deployment target.

## Training Stages

1. Train or import a licensed latent autoencoder.
2. Train the flow core on captioned image data.
3. Add reference and masked editing data.
4. Distill from many-step teacher to 1-4 step student.
5. Quantize and export.
6. Run human and automated evaluations.

## Known Gaps

- No trained checkpoint is committed.
- The hash tokenizer is a bootstrap tool, not a state-of-the-art semantic text
  encoder.
- Safety filtering, watermarking, and dataset governance are not solved here.
- Claims about quality require real remote training and benchmark evidence.

