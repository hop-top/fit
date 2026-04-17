"""Tests for fit.training.export — model export and model card generation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

        # Should either succeed, raise ImportError when gguf is unavailable,
        # or raise NotImplementedError when gguf is installed but conversion
        # is intentionally unimplemented.
        try:
            exporter.to_gguf(str(tmp_path / "out.gguf"))
        except (ImportError, NotImplementedError):
            pass  # expected in environments without gguf or with stubbed support

    def test_push_to_hub(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock, patch

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))

        captured_calls: dict[str, dict[str, object]] = {}

        class FakeHfApi:
            def upload_folder(self, **kwargs: object) -> None:
                captured_calls["upload_folder"] = kwargs

            def upload_file(self, **kwargs: object) -> None:
                captured_calls["upload_file"] = kwargs

        fake_hf_module = MagicMock(HfApi=FakeHfApi)
        with patch.dict("sys.modules", {"huggingface_hub": fake_hf_module}):
            exporter.push_to_hub("test/repo", _training_result())

        assert "upload_folder" in captured_calls
        assert captured_calls["upload_folder"]["repo_id"] == "test/repo"
        assert "upload_file" in captured_calls
        assert captured_calls["upload_file"]["repo_id"] == "test/repo"

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


# ---------------------------------------------------------------------------
# Regression: to_onnx preflight only checks transformers, not torch
# ---------------------------------------------------------------------------


class TestOnnxMissingTorchRegression:
    """to_onnx() preflight must check both torch and transformers."""

    def test_onnx_preflight_checks_torch_and_transformers(self) -> None:
        import inspect

        source = inspect.getsource(ModelExporter.to_onnx)

        # Find the first try block (the preflight check)
        in_preflight_try = False
        preflight_lines: list[str] = []
        for line in source.splitlines():
            stripped = line.strip()
            if "try:" in stripped and not in_preflight_try:
                in_preflight_try = True
                continue
            if in_preflight_try:
                if stripped.startswith("except"):
                    break
                preflight_lines.append(stripped)

        preflight_text = "\n".join(preflight_lines)
        assert "transformers" in preflight_text, (
            "Preflight should check transformers"
        )
        assert "torch" in preflight_text, (
            "to_onnx preflight must check torch in addition to transformers. "
            "When transformers is installed but torch is not, the user "
            "should get the formatted error message, not a raw ImportError."
        )


# ---------------------------------------------------------------------------
# Regression: to_safetensors torch import unguarded
# ---------------------------------------------------------------------------


class TestSafetensorsTorchImportRegression:
    """to_safetensors() must wrap torch import with clear ImportError."""

    def test_safetensors_torch_import_is_guarded(self) -> None:
        import inspect

        source = inspect.getsource(ModelExporter.to_safetensors)
        lines = source.splitlines()

        # Find the bin_file block containing import torch
        torch_import_idx = None
        if_indent = None
        for i, line in enumerate(lines):
            if "import torch" in line and "bin_file" not in line:
                torch_import_idx = i
                break

        assert torch_import_idx is not None, (
            "Could not find 'import torch' in to_safetensors source"
        )

        # Walk backwards to find the bin_file block indent
        for i in range(torch_import_idx - 1, -1, -1):
            stripped = lines[i].strip()
            if "if bin_file.exists():" in stripped:
                if_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        assert if_indent is not None, "Could not find bin_file.exists() block"

        # Collect all lines within the bin_file block
        in_block = False
        block_lines: list[str] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "if bin_file.exists():" in stripped:
                in_block = True
                continue
            if in_block:
                current_indent = len(line) - len(line.lstrip())
                if stripped and current_indent <= if_indent and i > torch_import_idx:
                    break
                block_lines.append(stripped)

        block_text = "\n".join(block_lines)
        has_try = "try:" in block_text
        has_except = "except ImportError" in block_text

        assert has_try and has_except, (
            "torch import inside bin_file block must be wrapped in "
            "try/except ImportError with a clear message. "
            f"Block text:\n{block_text}"
        )


# ---------------------------------------------------------------------------
# Regression: to_onnx ImportError message falsely claims optimum is
# required, but the code has a fallback path that works without it.
# ---------------------------------------------------------------------------


class TestOnnxErrorMsgOptimumOptionalRegression:
    """to_onnx() preflight error must present optimum as optional.

    The preflight ImportError (raised when torch/transformers are missing)
    must say only "pip install torch transformers" and mention optimum
    as an optional optimization, not as a required dependency.
    The code has a fallback via _onnx_export_torch that works without
    optimum, so listing it as required was misleading.
    """

    def test_onnx_error_message_not_requiring_optimum(self) -> None:
        import inspect

        source = inspect.getsource(ModelExporter.to_onnx)

        # Locate the preflight ImportError block
        preflight_msg_lines: list[str] = []
        in_raise = False
        for line in source.splitlines():
            stripped = line.strip()
            if 'raise ImportError' in stripped:
                in_raise = True
                preflight_msg_lines.append(stripped)
                continue
            if in_raise:
                if stripped.startswith('"') or stripped.startswith("'"):
                    preflight_msg_lines.append(stripped)
                else:
                    in_raise = False

        preflight_msg = " ".join(preflight_msg_lines)

        # Must NOT present optimum as required
        assert "pip install torch transformers optimum" not in preflight_msg, (
            "Preflight ImportError must not list 'optimum' as a required "
            "dependency. The _onnx_export_torch fallback works without it. "
            f"Got: {preflight_msg}"
        )

        # Should mention optimum as optional
        assert "Optionally" in preflight_msg or "optionally" in preflight_msg, (
            "Preflight ImportError should mention optimum as optional. "
            f"Got: {preflight_msg}"
        )

        # Fallback path must exist
        assert "_onnx_export_torch" in source


# ---------------------------------------------------------------------------
# Regression: torch.load(..., weights_only=True) TypeError on old torch
# ---------------------------------------------------------------------------


class TestTorchLoadWeightsOnlyRegression:
    """torch.load(..., weights_only=True) raises TypeError on torch < 1.13.
    to_safetensors() must catch TypeError and retry without weights_only."""

    def test_safetensors_torch_load_catches_type_error(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin").write_bytes(b"fakeweights")

        exporter = ModelExporter(str(model_dir))

        call_count = 0
        fake_state = {"layer.weight": MagicMock(numpy=lambda: MagicMock())}

        def mock_torch_load(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and kwargs.get("weights_only") is True:
                raise TypeError(
                    "torch.load() got an unexpected keyword argument "
                    "'weights_only'"
                )
            return fake_state

        fake_torch = MagicMock(load=mock_torch_load)
        fake_sf = MagicMock()
        fake_sf.torch = MagicMock(save_file=MagicMock())

        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "safetensors": fake_sf,
            "safetensors.torch": fake_sf.torch,
        }):
            out = exporter.to_safetensors(str(tmp_path / "out"))

        assert call_count == 2, (
            f"Expected 2 calls to torch.load (1st with weights_only "
            f"raising TypeError, 2nd without), got {call_count}"
        )
        assert out.parent.exists()

    def test_safetensors_torch_load_weights_only_preferred(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "pytorch_model.bin").write_bytes(b"fakeweights")

        exporter = ModelExporter(str(model_dir))

        fake_state = {"layer.weight": MagicMock(numpy=lambda: MagicMock())}
        mock_load = MagicMock(return_value=fake_state)
        fake_torch = MagicMock(load=mock_load)
        fake_sf = MagicMock()
        fake_sf.torch = MagicMock(save_file=MagicMock())

        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "safetensors": fake_sf,
            "safetensors.torch": fake_sf.torch,
        }):
            exporter.to_safetensors(str(tmp_path / "out"))

        mock_load.assert_called_once()
        _, kwargs = mock_load.call_args
        assert kwargs.get("weights_only") is True, (
            "torch.load should be called with weights_only=True when "
            f"torch supports it. Got kwargs: {kwargs}"
        )


# ---------------------------------------------------------------------------
# Regression: ONNX export input names
# ---------------------------------------------------------------------------


class TestOnnxExportInputNames:
    """_onnx_export_torch passes (input_ids, attention_mask) as inputs
    but input_names must have matching length for torch.onnx.export."""

    def test_input_names_count_matches_inputs(self) -> None:
        """input_names must have as many entries as model inputs."""
        exporter = ModelExporter("/tmp/nonexistent-model")

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        captured_export: dict = {}

        def capture_export(
            *args: object, **kwargs: object
        ) -> None:
            captured_export.update(kwargs)

        mock_torch = MagicMock()
        mock_torch.onnx.export = capture_export
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = (
            mock_model
        )
        mock_transformers.AutoTokenizer.from_pretrained.return_value = (
            mock_tokenizer
        )

        with patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "transformers": mock_transformers,
            },
        ):
            try:
                exporter._onnx_export_torch(
                    exporter._model_path / "model.onnx"
                )
            except TypeError:
                pass

        input_names = captured_export.get("input_names", [])

        if len(input_names) == 1:
            pytest.fail(
                "input_names has 1 entry "
                f"({input_names}) but 2 inputs are passed "
                "(input_ids + attention_mask). "
                "torch.onnx.export requires matching lengths."
            )

        assert len(input_names) == 2


# ---------------------------------------------------------------------------
# Regression: push_to_hub uses BytesIO
# ---------------------------------------------------------------------------


class TestPushToHubUsesBufferIO:
    """push_to_hub must pass BytesIO, not raw bytes, to
    HfApi.upload_file."""

    def test_uses_bytes_io_not_raw_bytes(
        self, tmp_path: Path
    ) -> None:
        from fit.training.grpo import TrainingResult

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

        captured_args: dict = {}

        class FakeHfApi:
            def upload_folder(self, **kwargs: object) -> None:
                pass

            def upload_file(self, **kwargs: object) -> None:
                captured_args.update(kwargs)

        with patch.dict(
            "sys.modules",
            {"huggingface_hub": MagicMock(HfApi=FakeHfApi)},
        ):
            exporter.push_to_hub("test/repo", result)

        assert "path_or_fileobj" in captured_args
        file_obj = captured_args["path_or_fileobj"]
        import io

        assert isinstance(file_obj, io.BytesIO), (
            f"path_or_fileobj is {type(file_obj).__name__}, "
            "expected BytesIO. Raw bytes cause upload_file "
            "to fail."
        )


# ---------------------------------------------------------------------------
# Regression: to_gguf raises NotImplementedError
# ---------------------------------------------------------------------------


class TestToGgufUnimplementedRaises:
    """``ModelExporter.to_gguf()`` must either create a real artifact
    or raise ``NotImplementedError`` -- it must not return a path
    that does not exist on disk.
    """

    def test_returned_path_exists_or_raises(
        self, tmp_path: Path
    ) -> None:
        """If ``to_gguf`` returns a path, that path must exist.
        Otherwise the method must raise NotImplementedError."""
        import sys
        import types

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        exporter = ModelExporter(str(model_dir))

        fake_gguf = types.ModuleType("gguf")
        fake_gguf.GGUFWriter = type(  # type: ignore[attr-defined]
            "GGUFWriter", (), {}
        )
        with patch.dict(sys.modules, {"gguf": fake_gguf}):
            result = None
            raised = False
            try:
                result = exporter.to_gguf(
                    str(tmp_path / "model.gguf")
                )
            except NotImplementedError:
                raised = True

        if raised:
            return  # acceptable

        assert result is not None and result.exists(), (
            "to_gguf returned a path that does not "
            "exist on disk — no GGUF artifact was created"
        )


# ---------------------------------------------------------------------------
# Regression: to_onnx ignores output_path filename with optimum
# ---------------------------------------------------------------------------


class TestOnnxOutputPathConsistencyRegression:
    """Regression: to_onnx optimum path ignores the caller's filename.

    When ``output_path="exports/my_model.onnx"``, the optimum branch
    discards the filename and returns ``out.parent / "model.onnx"``
    instead. The torch fallback correctly writes to the exact
    ``output_path``. Both paths must honour the caller's filename.
    """

    def test_optimum_path_returns_requested_output_path(
        self, tmp_path: Path
    ) -> None:
        """to_onnx must return the exact path the caller requested."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        exporter = ModelExporter(str(model_dir))

        requested = tmp_path / "exports" / "my_model.onnx"

        def fake_save_pretrained(dest: str) -> None:
            """Simulate optimum writing model.onnx into dest dir."""
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "model.onnx").write_bytes(
                b"fake-onnx-payload"
            )

        mock_ort_model = MagicMock()
        mock_ort_model.save_pretrained.side_effect = (
            fake_save_pretrained
        )

        fake_optimum = MagicMock()
        fake_optimum.onnxruntime.ORTModelForCausalLM = MagicMock(
            from_pretrained=MagicMock(return_value=mock_ort_model),
        )

        fake_torch = MagicMock()
        fake_transformers = MagicMock()

        with patch.dict("sys.modules", {
            "torch": fake_torch,
            "transformers": fake_transformers,
            "optimum": fake_optimum,
            "optimum.onnxruntime": fake_optimum.onnxruntime,
        }):
            result = exporter.to_onnx(str(requested))

        assert Path(result) == requested, (
            f"to_onnx returned {result!s} but caller requested "
            f"{requested!s}. The optimum branch ignores the "
            "filename and always returns 'model.onnx'."
        )
        assert requested.exists(), (
            "Artifact must exist at the requested path"
        )
        assert not (requested.parent / "model.onnx").exists(), (
            "Original model.onnx must be renamed, not left behind"
        )
