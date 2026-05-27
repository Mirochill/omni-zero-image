from __future__ import annotations

import math

from .config import ModelConfig


def _require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install torch support with: pip install -e '.[torch]'") from exc
    return torch, nn, F


torch, nn, F = _require_torch()


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TinyAutoencoder(nn.Module):
    """Compact latent image codec used by the prototype model path."""

    def __init__(self, latent_channels: int = 4, base_channels: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels * 2, latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * 2, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.encoder(image)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)


class TextConditioner(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.token = nn.Embedding(config.text_vocab_size, config.text_dim)
        self.position = nn.Parameter(torch.zeros(1, config.text_max_tokens, config.text_dim))
        self.norm = nn.LayerNorm(config.text_dim)
        self.proj = nn.Linear(config.text_dim, config.model_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token(token_ids) + self.position[:, : token_ids.shape[1]]
        x = self.norm(x)
        if token_mask is not None:
            x = x * token_mask.unsqueeze(-1).to(x.dtype)
        return self.proj(x)


class ReferenceEncoder(nn.Module):
    def __init__(self, latent_channels: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, model_dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(model_dim // 2, model_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, reference_latents: torch.Tensor) -> torch.Tensor:
        x = self.net(reference_latents).mean(dim=(2, 3))
        return self.norm(x).unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, model_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, model_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, latents: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(latents)
        height, width = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, height, width


class PatchUnembed(nn.Module):
    def __init__(self, out_channels: int, model_dim: int, patch_size: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.proj = nn.Linear(model_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch = tokens.shape[0]
        patch = self.patch_size
        x = self.proj(tokens)
        x = x.view(batch, height, width, self.out_channels, patch, patch)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(batch, self.out_channels, height * patch, width * patch)


class FlowBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim)
        self.cross_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * mlp_ratio, model_dim),
        )
        self.time_scale = nn.Linear(model_dim, model_dim)
        self.time_shift = nn.Linear(model_dim, model_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_embed: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.time_scale(time_embed).unsqueeze(1)
        shift = self.time_shift(time_embed).unsqueeze(1)
        y = self.norm1(x) * (1 + scale) + shift
        y, _ = self.self_attn(y, y, y, need_weights=False)
        x = x + y
        key_padding_mask = None
        if context_mask is not None:
            key_padding_mask = ~context_mask.bool()
        y = self.norm2(x)
        y, _ = self.cross_attn(y, context, context, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm3(x))
        return x


class OmniZeroModel(nn.Module):
    """Latent rectified-flow denoiser with text, reference, and mask conditioning."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.latent_size % config.patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size")
        self.config = config
        self.text = TextConditioner(config)
        self.reference = ReferenceEncoder(config.latent_channels, config.model_dim)
        self.patch = PatchEmbed(config.latent_channels, config.model_dim, config.patch_size)
        self.unpatch = PatchUnembed(config.latent_channels, config.model_dim, config.patch_size)
        token_count = (config.latent_size // config.patch_size) ** 2
        self.position = nn.Parameter(torch.zeros(1, token_count, config.model_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.SiLU(),
            nn.Linear(config.model_dim * 4, config.model_dim),
        )
        self.mask_proj = nn.Linear(1, config.model_dim)
        self.blocks = nn.ModuleList(
            [
                FlowBlock(config.model_dim, config.heads, config.mlp_ratio, config.dropout)
                for _ in range(config.layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        reference_latents: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, height, width = self.patch(latents)
        x = x + self._position_for(height, width, x.device, x.dtype)
        time_embed = self.time_mlp(timestep_embedding(timesteps, self.config.model_dim))
        context = self.text(token_ids, token_mask)
        context_mask = token_mask
        if reference_latents is not None:
            ref = self.reference(reference_latents)
            context = torch.cat([context, ref], dim=1)
            if context_mask is not None:
                ref_mask = torch.ones((context_mask.shape[0], 1), dtype=context_mask.dtype, device=context_mask.device)
                context_mask = torch.cat([context_mask, ref_mask], dim=1)
        if edit_mask is not None:
            pooled_mask = F.avg_pool2d(edit_mask, kernel_size=self.config.patch_size, stride=self.config.patch_size)
            mask_tokens = pooled_mask.flatten(2).transpose(1, 2)
            x = x + self.mask_proj(mask_tokens)
        for block in self.blocks:
            x = block(x, context, time_embed, context_mask)
        x = self.final_norm(x)
        return self.unpatch(x, height, width)

    def _position_for(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        needed = height * width
        if needed <= self.position.shape[1]:
            return self.position[:, :needed].to(device=device, dtype=dtype)
        base = int(math.sqrt(self.position.shape[1]))
        pos = self.position.reshape(1, base, base, self.config.model_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(height, width), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, needed, self.config.model_dim).to(device=device, dtype=dtype)


class OmniZeroBundle(nn.Module):
    """Autoencoder plus flow model for simple checkpoint packaging."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.autoencoder = TinyAutoencoder(config.latent_channels)
        self.flow = OmniZeroModel(config)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(image)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        reference_latents: torch.Tensor | None = None,
        edit_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.flow(latents, timesteps, token_ids, token_mask, reference_latents, edit_mask)
