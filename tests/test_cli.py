from pathlib import Path

from omni_zero.cli import main


def test_cli_generate_draft(tmp_path: Path) -> None:
    output = tmp_path / "draft.png"
    code = main(
        [
            "generate",
            "--mode",
            "draft",
            "--prompt",
            "a product photo of a speaker",
            "--out",
            str(output),
            "--size",
            "96",
        ]
    )
    assert code == 0
    assert output.exists()


def test_cli_model_requires_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "model.png"
    code = main(
        [
            "generate",
            "--mode",
            "model",
            "--prompt",
            "a city",
            "--out",
            str(output),
        ]
    )
    assert code == 1
    assert not output.exists()

