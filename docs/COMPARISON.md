# Comparison Notes

This file is a comparison framework, not a claim that Omni Zero Image already
matches frontier image models.

## Current Baselines To Track

| Model family | Public positioning | Why it matters for Omni Zero |
| --- | --- | --- |
| GPT Image 2 | OpenAI documents it as a fast, high-quality image generation and editing model. | Sets a commercial quality target for prompt adherence, editing, and typography. |
| Nano Banana Pro / Gemini image generation | Google positions Nano Banana Pro as an advanced image generation model with stronger quality, editing, and control. | Sets a target for multimodal context and controllable editing. |
| FLUX.1 Kontext | Black Forest Labs describes it as a unified text-to-image and in-context editing family. | Closest public architectural comparison for unified generation/editing. |
| Stable Diffusion 3.5 | Stability AI released open model variants designed for customization and consumer hardware. | Important open-weight baseline for cost and deployability. |

## Sources

- OpenAI GPT Image 2 docs: https://developers.openai.com/api/docs/models/gpt-image-2
- Google Nano Banana Pro product post: https://blog.google/products/gemini/where-to-use-nano-banana-pro/
- Google Gemini image help: https://support.google.com/gemini/answer/14286560
- Black Forest Labs FLUX.1 Kontext: https://bfl.ai/models/flux-kontext
- FLUX.1 Kontext paper: https://arxiv.org/abs/2506.15742
- Stability AI Stable Diffusion 3.5 announcement: https://stability.ai/news/introducing-stable-diffusion-3-5

## Required Fair Comparison Protocol

1. Use the same prompt set for every model.
2. Use image-to-image tests with the same references and masks.
3. Record model version, date, resolution, seed if available, and settings.
4. Separate speed, cost, quality, prompt adherence, and edit locality.
5. Avoid comparing draft-mode outputs to production model outputs.
6. Report failures and cherry-picking policy.

## Omni Zero Current Position

- Software architecture: implemented.
- Draft inference: implemented for pipeline validation.
- Trained neural checkpoint: not published.
- Production quality: not established.
- Cost target: plausible only after distillation and quantization are measured.

