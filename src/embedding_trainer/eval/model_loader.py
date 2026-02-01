"""Model loading utilities with auto-detection."""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _is_hf_checkpoint(model_path: Path) -> bool:
    """Check if the path contains a HuggingFace checkpoint."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    # HF models have 'architectures' or 'model_type' in config
    return "architectures" in config or "model_type" in config


def _load_hf_model(model_path: Path, device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a HuggingFace model checkpoint from a local path."""
    from transformers import AutoModelForMaskedLM

    logger.info(f"Loading HuggingFace model from {model_path}")
    model = AutoModelForMaskedLM.from_pretrained(str(model_path))
    model = model.to(device)
    model.eval()

    # Load config for metadata
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    metadata = {
        "model_type": "hf",
        "architecture": config.get("architectures", [config.get("model_type")])[0],
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("hidden_size"),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, metadata


def _load_hf_hub_model(model_id: str, device: str) -> tuple[nn.Module, dict[str, Any]]:
    """Load a model directly from HuggingFace Hub."""
    from transformers import AutoModelForMaskedLM

    logger.info(f"Loading HuggingFace Hub model: {model_id}")
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    # Extract metadata from model.config instead of local config.json
    config = model.config

    architectures = getattr(config, "architectures", None)
    if architectures:
        architecture = architectures[0]
    else:
        architecture = getattr(config, "model_type", "unknown")

    metadata = {
        "model_type": "hf_hub",
        "model_id": model_id,
        "architecture": architecture,
        "vocab_size": getattr(config, "vocab_size", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, metadata


def _load_custom_model(
    model_path: Path,
    model_class: str,
    device: str,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a custom model checkpoint."""
    logger.info(f"Loading custom model from {model_path}")

    # Parse model class path (e.g., "mymodule.MyModel")
    module_name, class_name = model_class.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Load checkpoint
    checkpoint_file = model_path if model_path.is_file() else model_path / "model.pt"
    if not checkpoint_file.exists():
        # Try pytorch_model.bin
        checkpoint_file = model_path / "pytorch_model.bin"

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {model_path}. "
            "Expected 'model.pt' or 'pytorch_model.bin'"
        )

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config = checkpoint.get("config", {})
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            config = checkpoint.get("config", {})
        else:
            # Assume it's just the state dict
            state_dict = checkpoint
            config = {}
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")

    # Instantiate model
    if config:
        model = cls(**config)
    else:
        model = cls()

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    metadata = {
        "model_type": "custom",
        "class": model_class,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, metadata


def load_model(
    model_path: str | Path,
    model_type: str = "auto",
    model_class: str | None = None,
    device: str = "cuda",
) -> tuple[nn.Module, dict[str, Any]]:
    """
    Load a model checkpoint with auto-detection.

    Args:
        model_path: Path to the model checkpoint, or a HuggingFace Hub model ID
        model_type: "auto", "hf", or "custom"
        model_class: For custom models, the class path (e.g., "mymodule.MyModel")
        device: Device to load model on

    Returns:
        Tuple of (model, metadata dict)
    """
    path = Path(model_path)
    is_local = path.exists()

    # If not local and model_type allows, try loading from HuggingFace Hub
    if not is_local:
        if model_type in ("hf", "auto"):
            return _load_hf_hub_model(str(model_path), device)
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

    # Local path handling
    # Auto-detect model type
    if model_type == "auto":
        if _is_hf_checkpoint(path):
            model_type = "hf"
        elif model_class is not None:
            model_type = "custom"
        else:
            raise ValueError(
                f"Cannot auto-detect model type for {model_path}. "
                "Specify --model_type or --model_class"
            )

    if model_type == "hf":
        return _load_hf_model(path, device)
    elif model_type == "custom":
        if model_class is None:
            raise ValueError("model_class is required for custom models")
        return _load_custom_model(path, model_class, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
