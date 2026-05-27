from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .checkpoint import load_state
from .config import ModelConfig
from .image_io import mask_to_tensor, open_luma, open_rgb, pil_to_tensor, tensor_to_pil
from .model import OmniZeroBundle, torch
from .tokenizer import HashTokenizer


@dataclass(frozen=True)
class SampleOptions:
    prompt: str
    size: int = 512
    steps: int = 8
    guidance: float = 3.0
    seed: int = 0
    strength: float = 0.75


class OmniZeroPipeline:
    def __init__(
        self,
        config: ModelConfig | None = None,
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = HashTokenizer(self.config.text_vocab_size, self.config.text_max_tokens)
        self.bundle = OmniZeroBundle(self.config).to(self.device).eval()
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        state = load_state(checkpoint)
        missing, unexpected = self.bundle.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected[:8]}")
        if missing:
            raise ValueError(f"Missing checkpoint keys: {missing[:8]}")

    @torch.inference_mode()
    def generate(self, options: SampleOptions) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(options.seed)
        latent_size = options.size // self.config.latent_downsample
        shape = (1, self.config.latent_channels, latent_size, latent_size)
        latents = torch.randn(shape, generator=generator, device=self.device)
        token_ids, token_mask = self._tokens(options.prompt)
        latents = self._sample_loop(latents, token_ids, token_mask, options)
        decoded = self.bundle.decode(latents)
        return tensor_to_pil(decoded)

    @torch.inference_mode()
    def edit(
        self,
        image_path: str | Path,
        options: SampleOptions,
        mask_path: str | Path | None = None,
    ) -> Image.Image:
        init = pil_to_tensor(open_rgb(image_path), size=options.size).to(self.device)
        init_latents = self.bundle.encode(init)
        generator = torch.Generator(device=self.device).manual_seed(options.seed)
        noise = torch.randn(init_latents.shape, generator=generator, device=self.device)
        strength = min(1.0, max(0.0, options.strength))
        latents = init_latents * (1.0 - strength) + noise * strength
        token_ids, token_mask = self._tokens(options.prompt)
        edit_mask = None
        if mask_path:
            edit_mask = mask_to_tensor(open_luma(mask_path), init_latents.shape[-1]).to(self.device)
        latents = self._sample_loop(
            latents,
            token_ids,
            token_mask,
            options,
            reference_latents=init_latents,
            edit_mask=edit_mask,
            preserve_latents=init_latents,
        )
        decoded = self.bundle.decode(latents)
        return tensor_to_pil(decoded)

    def _tokens(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids, mask = self.tokenizer.encode(prompt)
        token_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        token_mask = torch.tensor([mask], dtype=torch.bool, device=self.device)
        return token_ids, token_mask

    def _sample_loop(
        self,
        latents: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        options: SampleOptions,
        reference_latents: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
        preserve_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        steps = max(1, int(options.steps))
        blank_ids, blank_mask = self._tokens("")
        for idx in range(steps):
            t_value = 1.0 - (idx / steps)
            next_t = 1.0 - ((idx + 1) / steps)
            dt = t_value - next_t
            timesteps = torch.full((latents.shape[0],), t_value, device=self.device)
            cond = self.bundle(latents, timesteps, token_ids, token_mask, reference_latents, edit_mask)
            if options.guidance != 1.0:
                uncond = self.bundle(latents, timesteps, blank_ids, blank_mask, reference_latents, edit_mask)
                velocity = uncond + options.guidance * (cond - uncond)
            else:
                velocity = cond
            latents = latents - dt * velocity
            if edit_mask is not None and preserve_latents is not None:
                latents = latents * edit_mask + preserve_latents * (1.0 - edit_mask)
        return latents

