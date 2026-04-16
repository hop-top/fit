"""Tests for fit.training.export — model export and model card generation."""
from __future__ import annotations

import json
from pathlib import Path

from fit.training.export import ModelExporter
from fit.training.grpo import TrainingResult


def _training_result(**overrides) -> TrainingResult:
    defaults = dict(
        model_path="/tmp/model",
        epochs_completed=3,
        final_loss=0.42,
        reward_stats={"mean": 0.85, "std": 0.1, "min": 0.5, "max": 1.0, "count": 100.0},
        training_metadata={"base_model": "Qwen/Qwen2-0.5B", "trainer": "simplified"},
    )
    defaults.update(overrides)
    return TrainingResult(**defaults)


class TestModelExporter:
    def test_init(self) -> None:
        exporter = ModelExporter("/tmp/model")
        assert exporter._model_path == Path("/tmp/model")

    def test_generate_model_card(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))
        result = _training_result()
        card = exporter.generate_model_card(result)

        assert card["base_model"] == "Qwen/Qwen2-0.5B"
        assert card["epochs"] == 3
        assert card["final_loss"] == 0.42
        assert card["trace_count"] == 100
        assert "export_timestamp" in card
        assert "reward_stats" in card

    def test_to_safetensors_missing_dep(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))

        # safetensors may or may not be installed
        try:
            exporter.to_safetensors(str(tmp_path / "out"))
        except ImportError as exc:
            assert "safetensors" in str(exc)

    def test_to_gguf_no_model(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))

        # Should either succeed (if gguf installed + no model) or raise ImportError
        try:
            exporter.to_gguf(str(tmp_path / "out.gguf"))
        except ImportError:
            pass  # expected when gguf not installed

    def test_push_to_hub(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))

        # push_to_hub requires valid HF credentials and repo;
        try:
            exporter.push_to_hub("test/repo", _training_result())
        except ImportError as exc:
            assert "huggingface" in str(exc).lower()
        except Exception as exc:
            message = str(exc).lower()
            expected_markers = (
                "401", "403", "404", "unauthorized", "forbidden",
                "not found", "credentials", "authentication", "token",
                "repo", "repository",
            )
            assert any(m in message for m in expected_markers), (
                f"Unexpected push_to_hub failure: {exc}"
            )

    def test_copy_artifacts(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}', encoding="utf-8")
        (model_dir / "tokenizer.json").write_text('{}', encoding="utf-8")

        exporter = ModelExporter(str(model_dir))
        dest = tmp_path / "export"
        dest.mkdir()
        exporter._copy_artifacts(dest)

        assert (dest / "config.json").exists()
        assert (dest / "tokenizer.json").exists()
        assert json.loads((dest / "config.json").read_text()) == {"test": True}


class TestPushToHubBytesIO:
    """Regression: push_to_hub must pass BytesIO (not raw bytes) to
    HfApi.upload_file for the model card upload."""

    def test_model_card_uploads_as_bytesio(
        self, tmp_path: Path
    ) -> None:
        import io
        from unittest.mock import MagicMock, patch

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))
        result = TrainingResult(
            model_path=str(model_dir),
            epochs_completed=1,
            final_loss=0.1,
            reward_stats={
                "mean": 0.5,
                "std": 0.0,
                "min": 0.5,
                "max": 0.5,
                "count": 1.0,
            },
        )

        captured_kwargs: dict = {}

        class FakeHfApi:
            def upload_folder(self, **kwargs: object) -> None:
                pass

            def upload_file(self, **kwargs: object) -> None:
                captured_kwargs.update(kwargs)

        fake_hf_module = MagicMock(HfApi=FakeHfApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf_module}):
            exporter.push_to_hub("test/repo", result)

        assert "path_or_fileobj" in captured_kwargs
        file_obj = captured_kwargs["path_or_fileobj"]
        assert isinstance(file_obj, io.BytesIO), (
            f"Bug: path_or_fileobj is {type(file_obj).__name__}, "
            "expected BytesIO. Raw bytes cause upload_file to fail."
        )
        # Verify the BytesIO content is valid JSON
        content = file_obj.read().decode()
        file_obj.seek(0)
        card = json.loads(content)
        assert "model_id" in card
        assert captured_kwargs["path_in_repo"] == "model_card.json"
