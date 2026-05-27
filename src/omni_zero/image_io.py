from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:  # pragma: no cover
    import torch


def open_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def open_luma(path: str | Path) -> Image.Image:
    return Image.open(path).convert("L")


def resize_square(image: Image.Image, size: int) -> Image.Image:
    if image.width == size and image.height == size:
        return image
    src = image.convert("RGB")
    scale = max(size / src.width, size / src.height)
    resized = src.resize((round(src.width * scale), round(src.height * scale)), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - size) // 2)
    top = max(0, (resized.height - size) // 2)
    return resized.crop((left, top, left + size, top + size))


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_image(image: Image.Image, path: str | Path) -> Path:
    target = ensure_parent(path)
    image.save(target)
    return target


def pil_to_tensor(image: Image.Image, size: int | None = None) -> "torch.Tensor":
    import numpy as np
    import torch

    src = resize_square(image, size) if size else image.convert("RGB")
    array = np.asarray(src).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.mul(2.0).sub(1.0)


def mask_to_tensor(mask: Image.Image, latent_size: int) -> "torch.Tensor":
    import numpy as np
    import torch

    resized = mask.convert("L").resize((latent_size, latent_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized).astype("float32") / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).clamp(0.0, 1.0)


def tensor_to_pil(tensor: "torch.Tensor") -> Image.Image:
    import numpy as np

    if tensor.ndim == 4:
        tensor = tensor[0]
    data = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    data = data.add(1.0).div(2.0).clamp(0.0, 1.0)
    array = data.permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")

