from omni_zero.tokenizer import HashTokenizer


def test_hash_tokenizer_is_deterministic() -> None:
    tokenizer = HashTokenizer(vocab_size=1024, max_tokens=16)
    ids_a, mask_a = tokenizer.encode("A red chair, cinematic light")
    ids_b, mask_b = tokenizer.encode("A red chair, cinematic light")
    assert ids_a == ids_b
    assert mask_a == mask_b
    assert len(ids_a) == 16
    assert mask_a[0] == 1
    assert mask_a[-1] == 0


def test_hash_tokenizer_respects_vocab_range() -> None:
    tokenizer = HashTokenizer(vocab_size=512, max_tokens=12)
    ids, _ = tokenizer.encode("futuristic glass city")
    assert min(ids) >= 0
    assert max(ids) < 512

