import pytest

torch = pytest.importorskip("torch")

from omni_zero.config import ModelConfig
from omni_zero.sampler import OmniZeroPipeline, SampleOptions


def test_sampler_random_weight_smoke() -> None:
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
    pipe = OmniZeroPipeline(config=config, device="cpu")
    image = pipe.generate(SampleOptions(prompt="a tiny smoke test", size=64, steps=1, seed=1))
    assert image.size == (64, 64)
    assert image.mode == "RGB"

