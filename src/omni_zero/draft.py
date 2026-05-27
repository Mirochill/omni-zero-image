from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .image_io import open_luma, open_rgb, resize_square


PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {
    "warm": ((255, 190, 110), (235, 95, 70), (75, 45, 55)),
    "cold": ((130, 210, 255), (70, 120, 210), (20, 30, 70)),
    "night": ((22, 26, 50), (55, 70, 130), (180, 210, 255)),
    "forest": ((58, 95, 60), (112, 154, 91), (230, 220, 170)),
    "mono": ((245, 245, 245), (135, 135, 135), (20, 20, 20)),
}


@dataclass(frozen=True)
class DraftResult:
    image: Image.Image
    seed: int
    palette: str


class DraftGenerator:
    """Instant deterministic renderer for end-to-end smoke tests.

    This is not the neural model. It gives the repository a zero-cost execution
    path and produces coherent placeholders that make CLIs, docs, CI, and UI
    integration testable before remote training has produced a real checkpoint.
    """

    def __init__(self, size: int = 768) -> None:
        if size < 64:
            raise ValueError("size must be at least 64")
        self.size = size

    def generate(self, prompt: str, seed: int | None = None) -> DraftResult:
        seed_value = seed if seed is not None else self._seed(prompt)
        rng = random.Random(seed_value)
        palette_name = self._palette_name(prompt)
        palette = PALETTES[palette_name]
        image = self._gradient(palette, rng)
        draw = ImageDraw.Draw(image, "RGBA")
        self._draw_scene(draw, prompt, palette, rng)
        image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=120, threshold=2))
        image = ImageEnhance.Contrast(image).enhance(1.06)
        return DraftResult(image=image.convert("RGB"), seed=seed_value, palette=palette_name)

    def edit(
        self,
        image_path: str | Path,
        prompt: str,
        mask_path: str | Path | None = None,
        strength: float = 0.55,
        seed: int | None = None,
    ) -> DraftResult:
        base = resize_square(open_rgb(image_path), self.size)
        generated = self.generate(prompt, seed=seed)
        strength = min(1.0, max(0.0, strength))
        generated_image = self._apply_prompt_grade(generated.image, prompt)
        if mask_path:
            mask = open_luma(mask_path).resize((self.size, self.size), Image.Resampling.BILINEAR)
            mask = ImageEnhance.Contrast(mask).enhance(1.5)
            overlay = Image.blend(base, generated_image, strength)
            edited = Image.composite(overlay, base, mask)
        else:
            edited = Image.blend(base, generated_image, strength)
        return DraftResult(image=edited.convert("RGB"), seed=generated.seed, palette=generated.palette)

    def _seed(self, prompt: str) -> int:
        digest = hashlib.blake2b((prompt or "").encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") & 0x7FFF_FFFF

    def _palette_name(self, prompt: str) -> str:
        text = prompt.lower()
        if any(word in text for word in ("night", "cyber", "neon", "space")):
            return "night"
        if any(word in text for word in ("forest", "nature", "garden", "jungle")):
            return "forest"
        if any(word in text for word in ("cold", "ice", "blue", "winter")):
            return "cold"
        if any(word in text for word in ("monochrome", "black", "white", "sketch")):
            return "mono"
        return "warm"

    def _gradient(
        self,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> Image.Image:
        width = height = self.size
        image = Image.new("RGB", (width, height))
        px = image.load()
        a, b, c = palette
        tilt = rng.uniform(-0.35, 0.35)
        for y in range(height):
            for x in range(width):
                t = (y / max(1, height - 1)) * 0.75 + (x / max(1, width - 1)) * tilt
                t = max(0.0, min(1.0, t))
                if t < 0.55:
                    local = t / 0.55
                    color = tuple(round(a[i] * (1 - local) + b[i] * local) for i in range(3))
                else:
                    local = (t - 0.55) / 0.45
                    color = tuple(round(b[i] * (1 - local) + c[i] * local) for i in range(3))
                px[x, y] = color
        return image.filter(ImageFilter.GaussianBlur(radius=max(1, self.size // 180)))

    def _draw_scene(
        self,
        draw: ImageDraw.ImageDraw,
        prompt: str,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        text = prompt.lower()
        if any(word in text for word in ("city", "building", "street", "tower")):
            self._draw_city(draw, palette, rng)
        elif any(word in text for word in ("product", "speaker", "phone", "chair", "object")):
            self._draw_product(draw, palette, rng)
        elif any(word in text for word in ("portrait", "person", "character", "face")):
            self._draw_portrait(draw, palette, rng)
        elif any(word in text for word in ("landscape", "mountain", "forest", "nature")):
            self._draw_landscape(draw, palette, rng)
        else:
            self._draw_abstract(draw, palette, rng)

    def _draw_city(
        self,
        draw: ImageDraw.ImageDraw,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        w = h = self.size
        horizon = round(h * rng.uniform(0.46, 0.62))
        draw.rectangle((0, horizon, w, h), fill=(*palette[2], 185))
        x = -round(w * 0.04)
        while x < w:
            bw = rng.randint(max(18, w // 28), max(28, w // 10))
            bh = rng.randint(max(55, h // 6), max(90, h // 2))
            y = horizon - bh + rng.randint(-12, 18)
            color = tuple(max(0, min(255, v + rng.randint(-22, 28))) for v in palette[2])
            draw.rounded_rectangle((x, y, x + bw, horizon + 10), radius=3, fill=(*color, 225))
            for wx in range(x + 8, x + bw - 5, max(8, bw // 4)):
                for wy in range(y + 10, horizon - 8, max(12, bh // 7)):
                    if rng.random() > 0.42:
                        draw.rectangle((wx, wy, wx + 3, wy + 5), fill=(*palette[0], 150))
            x += bw + rng.randint(3, 12)
        sun_x = round(w * rng.uniform(0.15, 0.85))
        sun_y = round(h * rng.uniform(0.13, 0.35))
        radius = round(w * rng.uniform(0.05, 0.09))
        draw.ellipse((sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius), fill=(*palette[0], 185))

    def _draw_landscape(
        self,
        draw: ImageDraw.ImageDraw,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        w = h = self.size
        for layer in range(4):
            y_base = h * (0.48 + layer * 0.11)
            points = [(0, h)]
            for i in range(9):
                x = round(w * i / 8)
                y = round(y_base + math.sin(i * 1.7 + layer) * h * 0.04 + rng.randint(-18, 18))
                points.append((x, y))
            points.append((w, h))
            color = tuple(max(0, min(255, palette[min(2, layer // 2)][i] - layer * 18)) for i in range(3))
            draw.polygon(points, fill=(*color, 165 + layer * 20))
        for _ in range(28):
            x = rng.randint(0, w)
            y = rng.randint(round(h * 0.56), h)
            trunk = rng.randint(8, 28)
            draw.line((x, y, x, y - trunk), fill=(55, 40, 30, 160), width=max(1, w // 220))
            crown = rng.randint(10, 26)
            draw.ellipse((x - crown, y - trunk - crown, x + crown, y - trunk + crown), fill=(*palette[1], 120))

    def _draw_product(
        self,
        draw: ImageDraw.ImageDraw,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        w = h = self.size
        cx = w // 2
        cy = round(h * 0.54)
        shadow = (cx - w // 4, cy + h // 7, cx + w // 4, cy + h // 5)
        draw.ellipse(shadow, fill=(0, 0, 0, 55))
        body = (cx - w // 7, cy - h // 5, cx + w // 7, cy + h // 5)
        draw.rounded_rectangle(body, radius=w // 18, fill=(*palette[0], 220), outline=(*palette[2], 190), width=3)
        for idx in range(6):
            y = cy - h // 7 + idx * h // 18
            draw.line((cx - w // 9, y, cx + w // 9, y), fill=(*palette[2], 90), width=max(1, w // 180))
        glow = rng.randint(24, 48)
        draw.ellipse((cx - glow, cy - glow, cx + glow, cy + glow), outline=(*palette[1], 190), width=3)

    def _draw_portrait(
        self,
        draw: ImageDraw.ImageDraw,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        w = h = self.size
        cx = w // 2
        head_r = w // 8
        head_y = round(h * 0.36)
        draw.ellipse((cx - head_r, head_y - head_r, cx + head_r, head_y + head_r), fill=(*palette[0], 220))
        draw.arc((cx - head_r // 2, head_y, cx + head_r // 2, head_y + head_r // 2), 10, 170, fill=(*palette[2], 160), width=3)
        eye_y = head_y - head_r // 5
        for dx in (-head_r // 3, head_r // 3):
            draw.ellipse((cx + dx - 4, eye_y - 4, cx + dx + 4, eye_y + 4), fill=(*palette[2], 210))
        shoulders = [
            (cx - w // 4, round(h * 0.78)),
            (cx - w // 7, round(h * 0.52)),
            (cx + w // 7, round(h * 0.52)),
            (cx + w // 4, round(h * 0.78)),
        ]
        draw.polygon(shoulders, fill=(*palette[2], 190))
        for _ in range(12):
            x = rng.randint(cx - w // 3, cx + w // 3)
            y = rng.randint(round(h * 0.2), round(h * 0.7))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(*palette[0], 120))

    def _draw_abstract(
        self,
        draw: ImageDraw.ImageDraw,
        palette: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
        rng: random.Random,
    ) -> None:
        w = h = self.size
        for idx in range(34):
            radius = rng.randint(max(12, w // 40), max(22, w // 7))
            x = rng.randint(-radius, w + radius)
            y = rng.randint(-radius, h + radius)
            color = palette[idx % len(palette)]
            alpha = rng.randint(45, 145)
            if rng.random() > 0.45:
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, alpha))
            else:
                draw.rounded_rectangle(
                    (x - radius, y - radius, x + radius, y + radius),
                    radius=max(4, radius // 4),
                    fill=(*color, alpha),
                )

    def _apply_prompt_grade(self, image: Image.Image, prompt: str) -> Image.Image:
        text = prompt.lower()
        result = image
        if any(word in text for word in ("warm", "sunset", "gold", "cinematic")):
            overlay = Image.new("RGB", image.size, (255, 190, 105))
            result = Image.blend(result, overlay, 0.12)
        if any(word in text for word in ("cold", "blue", "winter", "moon")):
            overlay = Image.new("RGB", image.size, (105, 160, 255))
            result = Image.blend(result, overlay, 0.12)
        if any(word in text for word in ("sharp", "detail", "crisp")):
            result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=2))
        return result
