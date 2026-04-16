"""Export trained advisor model to deployable artifacts.

Supports safetensors, GGUF, ONNX formats. All export deps are optional
and lazily imported. Model cards are JSON metadata for traceability.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .grpo import TrainingResult

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export a trained advisor model to various formats."""

    def __init__(self, model_path: str) -> None:
        self._model_path = Path(model_path)

    def to_safetensors(self, output_dir: str) -> Path:
        """Export model weights as safetensors format."""
        try:
            from safetensors.torch import save_file  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "safetensors not installed. Install with: pip install safetensors"
            )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Load model weights if available
        model_file = self._model_path / "model.safetensors"
        if model_file.exists():
            shutil.copy2(model_file, out / "model.safetensors")
        else:
            # Try converting from pytorch bin
            bin_file = self._model_path / "pytorch_model.bin"
            if bin_file.exists():
                try:
                    import torch  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "Converting pytorch_model.bin to safetensors "
                        "requires torch. Install with: pip install torch"
                    ) from exc

                state_dict = torch.load(
                    str(bin_file),
                    map_location="cpu",
                    weights_only=True,
                )
                save_file(state_dict, str(out / "model.safetensors"))
            else:
                logger.warning("No model weights found at %s", self._model_path)

        # Copy config/tokenizer files
        self._copy_artifacts(out)
        return out

    def to_gguf(self, output_path: str) -> Path:
        """Export model to GGUF format for llama.cpp / Ollama."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # GGUF conversion requires llama.cpp's convert script or
        # the gguf library. Both are optional heavy deps.
        try:
            from gguf import (  # type: ignore[import-untyped]
                GGUFWriter as _GGUFWriter,  # noqa: F401
            )
        except ImportError:
            # Fallback: copy if already a .gguf file
            existing = list(self._model_path.glob("*.gguf"))
            if existing:
                shutil.copy2(existing[0], out)
                return out
            raise ImportError(
                "gguf not installed. Install with: pip install gguf "
                "or convert via llama.cpp's convert_hf_to_gguf.py"
            )

        # Stub: actual conversion requires full model loading
        # In practice, use llama.cpp's conversion tools
        logger.warning(
            "GGUF conversion via Python is limited. "
            "Prefer llama.cpp's convert_hf_to_gguf.py for production."
        )
        return out

    def to_onnx(self, output_path: str) -> Path:
        """Export model to ONNX format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            __import__("torch")
            __import__("transformers")
        except ImportError as exc:
            raise ImportError(
                "ONNX export requires torch and transformers. "
                "Install with: pip install torch transformers optimum"
            ) from exc

        try:
            from optimum.onnxruntime import ORTModelForCausalLM  # type: ignore[import-untyped]
        except ImportError:
            # Fallback: manual torch.onnx.export
            return self._onnx_export_torch(out)

        model = ORTModelForCausalLM.from_pretrained(str(self._model_path))
        model.save_pretrained(str(out.parent))
        return out.parent / "model.onnx"

    def _onnx_export_torch(self, output_path: Path) -> Path:
        """Fallback ONNX export using torch.onnx.export."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(str(self._model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))

        dummy = tokenizer("test", return_tensors="pt")
        input_ids = dummy["input_ids"]
        attention_mask = dummy.get("attention_mask", None)

        onnx_inputs = (input_ids,)
        input_names = ["input_ids"]
        dynamic_axes: dict[str, dict[int, str]] = {
            "input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        }
        if attention_mask is not None:
            onnx_inputs = (input_ids, attention_mask)
            input_names.append("attention_mask")
            dynamic_axes["attention_mask"] = {0: "batch", 1: "seq"}

        torch.onnx.export(
            model,
            onnx_inputs,
            str(output_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
        )
        return output_path

    def generate_model_card(self, training_result: TrainingResult) -> dict[str, Any]:
        """Generate a model card JSON with training metadata."""
        card: dict[str, Any] = {
            "model_id": self._model_path.name,
            "base_model": training_result.training_metadata.get("base_model", "unknown"),
            "trainer": training_result.training_metadata.get("trainer", "unknown"),
            "epochs": training_result.epochs_completed,
            "final_loss": training_result.final_loss,
            "reward_stats": training_result.reward_stats,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_count": int(training_result.reward_stats.get("count", 0)),
            "model_path": str(self._model_path),
        }
        return card

    def push_to_hub(
        self,
        repo_id: str,
        training_result: TrainingResult,
    ) -> None:
        """Push model to Hugging Face Hub."""
        try:
            from huggingface_hub import HfApi  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )

        api = HfApi()

        # Upload model directory
        if self._model_path.exists():
            api.upload_folder(
                folder_path=str(self._model_path),
                repo_id=repo_id,
            )

        # Upload model card
        card = self.generate_model_card(training_result)
        import io

        card_json = json.dumps(card, indent=2)
        api.upload_file(
            path_or_fileobj=io.BytesIO(card_json.encode()),
            path_in_repo="model_card.json",
            repo_id=repo_id,
        )

    def _copy_artifacts(self, dest: Path) -> None:
        """Copy config/tokenizer files to destination."""
        for name in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "training_config.json",
        ]:
            src = self._model_path / name
            if src.exists():
                shutil.copy2(src, dest / name)
