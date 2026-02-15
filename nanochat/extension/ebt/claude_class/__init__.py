"""
Pure PyTorch implementations of pytorch_lightning components.

This package provides drop-in replacements for pytorch_lightning classes
without requiring the pytorch_lightning dependency.
"""

from .torchlightning_module import LightningModule, AttributeDict

__all__ = ['LightningModule', 'AttributeDict']
