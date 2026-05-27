import pytest

torch = pytest.importorskip("torch")

from omni_zero.config import ModelConfig
from omni_zero.model import OmniZeroBundle


def test_model_forward_shape() -> None:
    config = ModelConfig(
        image_size=64,
        latent_channels=4,
        latent_downsample=8,
        text_vocab_size=512,
        text_max_tokens=12,
        text_dim=64,
        model_dim=64,
        layers=2,
        heads=4,
        patch_size=2,
    )
    model = OmniZeroBundle(config).eval()
    latents = torch.randn(2, config.latent_channels, config.latent_size, config.latent_size)
    timesteps = torch.rand(2)
    token_ids = torch.randint(0, config.text_vocab_size, (2, config.text_max_tokens))
    token_mask = torch.ones(2, config.text_max_tokens, dtype=torch.bool)
    reference = torch.randn_like(latents)
    edit_mask = torch.ones(2, 1, config.latent_size, config.latent_size)
    with torch.inference_mode():
        velocity = model(latents, timesteps, token_ids, token_mask, reference, edit_mask)
    assert velocity.shape == latents.shape


def test_autoencoder_roundtrip_shape() -> None:
    config = ModelConfig(
        image_size=64,
        latent_channels=4,
        latent_downsample=8,
        text_vocab_size=512,
        text_max_tokens=12,
        text_dim=64,
        model_dim=64,
        layers=1,
        heads=4,
        patch_size=2,
    )
    model = OmniZeroBundle(config).eval()
    image = torch.randn(1, 3, config.image_size, config.image_size)
    with torch.inference_mode():
        latents = model.encode(image)
        decoded = model.decode(latents)
    assert latents.shape == (1, config.latent_channels, config.latent_size, config.latent_size)
    assert decoded.shape == image.shape

