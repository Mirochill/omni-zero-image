from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .image_io import pil_to_tensor
from .tokenizer import HashTokenizer


class JsonlImageDataset:
    """Captioned image dataset backed by JSONL records.

    Required fields:
    - image: path to an image, absolute or relative to the JSONL file
    - caption: text prompt
    """

    def __init__(self, path: str | Path, image_size: int, tokenizer: HashTokenizer) -> None:
        self.path = Path(path)
        self.root = self.path.parent
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.records = self._load_records()

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{self.path}:{line_no}: record must be an object")
            if "image" not in payload or "caption" not in payload:
                raise ValueError(f"{self.path}:{line_no}: record needs image and caption")
            records.append(payload)
        if not records:
            raise ValueError(f"No records found in {self.path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = Path(str(record["image"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = Image.open(image_path).convert("RGB")
        ids, mask = self.tokenizer.encode(str(record["caption"]))
        return {
            "image": pil_to_tensor(image, self.image_size)[0],
            "token_ids": ids,
            "token_mask": mask,
            "caption": str(record["caption"]),
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    import torch

    return {
        "image": torch.stack([item["image"] for item in batch]),  # type: ignore[list-item]
        "token_ids": torch.tensor([item["token_ids"] for item in batch], dtype=torch.long),
        "token_mask": torch.tensor([item["token_mask"] for item in batch], dtype=torch.bool),
        "caption": [item["caption"] for item in batch],
    }

