from pathlib import Path

from PIL import Image

from omni_zero.draft import DraftGenerator


def test_draft_generate_returns_rgb_image() -> None:
    result = DraftGenerator(size=96).generate("a futuristic city at sunrise", seed=42)
    assert result.image.mode == "RGB"
    assert result.image.size == (96, 96)
    assert result.seed == 42


def test_draft_edit_with_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "base.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (80, 80), (100, 100, 100)).save(image_path)
    mask = Image.new("L", (80, 80), 0)
    for x in range(20, 60):
        for y in range(20, 60):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    result = DraftGenerator(size=80).edit(
        image_path=image_path,
        mask_path=mask_path,
        prompt="make the center warm and cinematic",
        seed=4,
    )
    assert result.image.size == (80, 80)
    assert result.image.getpixel((5, 5)) == (100, 100, 100)
    assert result.image.getpixel((40, 40)) != (100, 100, 100)

