"""Precision contexts for mixed-precision training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast

if TYPE_CHECKING:
    from torch.optim import Optimizer


class PrecisionContext(ABC):
    """Abstract base class for precision contexts.

    Handles mixed-precision training operations including autocasting,
    loss scaling, and gradient unscaling.
    """

    @abstractmethod
    def autocast_context(self) -> Generator[None, None, None]:
        """Return a context manager for automatic mixed precision.

        Yields:
            Context manager that enables autocast for the forward pass.
        """
        ...

    @abstractmethod
    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale the loss for gradient computation.

        Args:
            loss: The loss tensor to scale.

        Returns:
            Scaled loss tensor (or original if no scaling needed).
        """
        ...

    @abstractmethod
    def unscale_and_step(self, optimizer: Optimizer) -> None:
        """Unscale gradients and step the optimizer.

        Args:
            optimizer: The optimizer to step.
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """Update the scaler after an optimizer step."""
        ...


class FP32Precision(PrecisionContext):
    """Full precision (FP32) context - no mixed precision."""

    @contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """No-op context manager for FP32 training."""
        yield

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Return loss unchanged."""
        return loss

    def unscale_and_step(self, optimizer: Optimizer) -> None:
        """Step optimizer directly without unscaling."""
        optimizer.step()

    def update(self) -> None:
        """No-op for FP32."""
        pass


class FP16Precision(PrecisionContext):
    """Mixed precision (FP16) context with gradient scaling."""

    def __init__(self, init_scale: float = 2.0**16) -> None:
        """Initialize FP16 precision context.

        Args:
            init_scale: Initial scale for gradient scaling.
        """
        self._scaler = GradScaler(init_scale=init_scale)

    @contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """Enable autocast for FP16."""
        with autocast(dtype=torch.float16):
            yield

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for FP16 training."""
        return self._scaler.scale(loss)

    def unscale_and_step(self, optimizer: Optimizer) -> None:
        """Unscale gradients and step optimizer."""
        self._scaler.unscale_(optimizer)
        self._scaler.step(optimizer)

    def update(self) -> None:
        """Update the gradient scaler."""
        self._scaler.update()


class BF16Precision(PrecisionContext):
    """BFloat16 precision context (no scaling needed)."""

    @contextmanager
    def autocast_context(self) -> Generator[None, None, None]:
        """Enable autocast for BF16."""
        with autocast(dtype=torch.bfloat16):
            yield

    def scale_loss(self, loss: Tensor) -> Tensor:
        """Return loss unchanged (BF16 doesn't need scaling)."""
        return loss

    def unscale_and_step(self, optimizer: Optimizer) -> None:
        """Step optimizer directly (no unscaling needed for BF16)."""
        optimizer.step()

    def update(self) -> None:
        """No-op for BF16."""
        pass


def get_precision_context(precision: str) -> PrecisionContext:
    """Get the appropriate precision context.

    Args:
        precision: One of "fp32", "fp16", or "bf16".

    Returns:
        The corresponding PrecisionContext instance.

    Raises:
        ValueError: If precision is not recognized.
    """
    precision = precision.lower()
    if precision == "fp32":
        return FP32Precision()
    elif precision == "fp16":
        return FP16Precision()
    elif precision == "bf16":
        return BF16Precision()
    else:
        raise ValueError(
            f"Unknown precision: {precision}. Use 'fp32', 'fp16', or 'bf16'."
        )
