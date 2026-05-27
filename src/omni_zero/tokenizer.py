from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)?|[^\s]", re.IGNORECASE)


@dataclass(frozen=True)
class TokenBatch:
    ids: list[list[int]]
    masks: list[list[int]]


class HashTokenizer:
    """Small deterministic tokenizer with no hosted vocabulary dependency.

    It is not a replacement for a learned text encoder in a frontier model. It
    exists so the whole repo remains runnable offline and so training jobs can
    start without downloading a tokenizer.
    """

    def __init__(self, vocab_size: int = 32768, max_tokens: int = 64) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        if max_tokens < 4:
            raise ValueError("max_tokens must be at least 4")
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]

    def token_to_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return 4 + (value % (self.vocab_size - 4))

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        body = [self.token_to_id(token) for token in self.tokenize(text)]
        ids = [self.bos_id, *body[: self.max_tokens - 2], self.eos_id]
        mask = [1] * len(ids)
        while len(ids) < self.max_tokens:
            ids.append(self.pad_id)
            mask.append(0)
        return ids, mask

    def batch_encode(self, prompts: list[str]) -> TokenBatch:
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for prompt in prompts:
            prompt_ids, prompt_mask = self.encode(prompt)
            ids.append(prompt_ids)
            masks.append(prompt_mask)
        return TokenBatch(ids=ids, masks=masks)

