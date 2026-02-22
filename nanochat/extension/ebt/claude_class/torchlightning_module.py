"""
Pure PyTorch implementation of pytorch_lightning.LightningModule.

This module provides a drop-in replacement for LightningModule without
depending on the pytorch_lightning package. It implements the core
functionality needed for training, validation, and testing workflows.

Usage:
    from claude_class.torchlightning_module import LightningModule

    class MyModel(LightningModule):
        def __init__(self, hparams):
            super().__init__()
            self.hparams.update(vars(hparams))
            # ... model definition

        def training_step(self, batch, batch_idx):
            # ... training logic
            return loss

        def configure_optimizers(self):
            return {'optimizer': optimizer, 'lr_scheduler': {...}}
"""

import torch
from torch import nn
from typing import Any, Dict, Optional, Union, List
from collections import OrderedDict


class AttributeDict(dict):
    """
    A dictionary that allows attribute-style access to its items.
    This mimics the behavior of Lightning's hparams object.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute '{key}'")

    def __contains__(self, key: str) -> bool:
        return super().__contains__(key)


class LightningModule(nn.Module):
    """
    Pure PyTorch implementation of pytorch_lightning.LightningModule.

    This class extends nn.Module with additional functionality for:
    - Hyperparameter management via self.hparams
    - Training/validation/test step hooks
    - Optimizer configuration
    - Metric logging
    - Checkpoint loading

    The trainer (Trainer class) is responsible for calling lifecycle hooks
    and managing the training loop.
    """

    def __init__(self) -> None:
        super().__init__()

        # Hyperparameters storage - mimics Lightning's hparams behavior
        self._hparams = AttributeDict()

        # Reference to the trainer - set by the Trainer when training starts
        self._trainer: Optional[Any] = None

        # Logged metrics for current step - cleared after each step
        self._logged_metrics: Dict[str, Any] = {}

        # Flag to track if we're in training mode (for logging purposes)
        self._in_training: bool = False

        # Whether validation is currently running (used to gate log buffering)
        self._is_validating: bool = False
        # Accumulates per-batch scalar values logged during validation;
        # flushed to wandb once at on_validation_epoch_end
        self._val_log_buffer: Dict[str, List] = {}

    @property
    def hparams(self) -> AttributeDict:
        """
        Returns the hyperparameters stored in this module.
        Hyperparameters can be accessed as attributes: self.hparams.learning_rate
        """
        return self._hparams

    @hparams.setter
    def hparams(self, value: Union[Dict, AttributeDict]) -> None:
        """Set hyperparameters from a dict or AttributeDict."""
        if isinstance(value, AttributeDict):
            self._hparams = value
        else:
            self._hparams = AttributeDict(value)

    @property
    def trainer(self) -> Optional[Any]:
        """
        Returns the trainer instance attached to this module.
        Set by the Trainer when fit/test/validate is called.
        """
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: Any) -> None:
        """Set the trainer reference."""
        self._trainer = trainer

    @property
    def logger(self) -> Optional[Any]:
        """
        Returns the logger from the trainer.
        Used for logging metrics to wandb, tensorboard, etc.
        """
        if self._trainer is not None:
            return getattr(self._trainer, 'logger', None)
        return None

    @property
    def global_step(self) -> int:
        """
        Returns the current global step count.
        This is the total number of optimizer steps taken across all epochs.
        """
        if self._trainer is not None:
            return getattr(self._trainer, 'global_step', 0)
        return 0

    @property
    def current_epoch(self) -> int:
        """
        Returns the current epoch number (0-indexed).
        """
        if self._trainer is not None:
            return getattr(self._trainer, 'current_epoch', 0)
        return 0

    @property
    def global_rank(self) -> int:
        """
        Returns the global rank in distributed training.
        Returns 0 for single-GPU or CPU training.
        """
        if self._trainer is not None:
            return getattr(self._trainer, 'global_rank', 0)
        return 0

    @property
    def device(self) -> torch.device:
        """
        Returns the device this module's parameters are on.
        Automatically detects from the first parameter.
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            # No parameters, return default device
            return torch.device('cpu')

    # ==================== Hyperparameter Management ====================

    def save_hyperparameters(
        self,
        *args,
        ignore: Optional[List[str]] = None,
        frame: Optional[Any] = None,
        logger: bool = True
    ) -> None:
        """
        Save hyperparameters to self.hparams.

        This is a simplified version - in the full Lightning implementation,
        it can automatically capture arguments from the calling frame.
        For this implementation, use self.hparams.update(vars(args)) instead.

        Args:
            *args: Ignored in this implementation
            ignore: List of parameter names to ignore
            frame: Ignored in this implementation
            logger: Whether to log hyperparameters
        """
        # This is a no-op in our simplified implementation
        # Users should use self.hparams.update(vars(hparams)) directly
        pass

    # ==================== Lifecycle Hooks ====================
    # These methods should be overridden by subclasses

    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass - must be implemented by subclass.

        This is the standard PyTorch forward method.
        """
        raise NotImplementedError("Subclass must implement forward()")

    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Training step - called for each batch during training.

        Args:
            batch: The current batch from the dataloader
            batch_idx: Index of the current batch

        Returns:
            Loss tensor to backpropagate, or None
        """
        raise NotImplementedError("Subclass must implement training_step()")

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Validation step - called for each batch during validation.

        Args:
            batch: The current batch from the dataloader
            batch_idx: Index of the current batch

        Returns:
            Optional loss or metric tensor
        """
        pass  # Optional - subclass may override

    def test_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        """
        Test step - called for each batch during testing.

        Args:
            batch: The current batch from the dataloader
            batch_idx: Index of the current batch

        Returns:
            Optional loss or metric tensor
        """
        pass  # Optional - subclass may override

    def configure_optimizers(self) -> Union[
        torch.optim.Optimizer,
        Dict[str, Any],
        List[torch.optim.Optimizer],
        None
    ]:
        """
        Configure optimizers and learning rate schedulers.

        Returns one of:
            - Single optimizer
            - Dictionary with 'optimizer' and optional 'lr_scheduler' keys
            - List of optimizers
            - None (if no optimization needed)

        Example return value:
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',  # or 'epoch'
                    'frequency': 1
                }
            }
        """
        raise NotImplementedError("Subclass must implement configure_optimizers()")

    def on_train_start(self) -> None:
        """
        Called at the beginning of training, before the first epoch.
        Override to add custom initialization logic.
        """
        pass

    def on_train_end(self) -> None:
        """
        Called at the end of training, after the last epoch.
        Override to add custom cleanup logic.
        """
        pass

    def on_train_epoch_start(self) -> None:
        """
        Called at the beginning of each training epoch.
        """
        pass

    def on_train_epoch_end(self) -> None:
        """
        Called at the end of each training epoch.
        """
        pass

    def on_validation_epoch_start(self) -> None:
        """Mark the start of validation; clear the per-epoch log buffer."""
        self._is_validating = True
        self._val_log_buffer.clear()

    def on_validation_epoch_end(self) -> None:
        """Flush buffered validation metrics to wandb as a single log call."""
        if self._val_log_buffer and self.logger is not None:
            avg_metrics: Dict[str, Any] = {}
            for name, values in self._val_log_buffer.items():
                numeric = [v.item() if isinstance(v, torch.Tensor) else float(v)
                           for v in values]
                avg_metrics[name] = sum(numeric) / len(numeric)
            if hasattr(self.logger, 'log_metrics'):
                self.logger.log_metrics(avg_metrics, step=self.global_step)
            elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                self.logger.experiment.log(avg_metrics, step=self.global_step)
        self._is_validating = False
        self._val_log_buffer.clear()

    def on_test_epoch_start(self) -> None:
        """
        Called at the beginning of testing.
        """
        pass

    def on_test_epoch_end(self) -> None:
        """
        Called at the end of testing.
        """
        pass

    def on_before_backward(self, loss: torch.Tensor) -> None:
        """
        Called before loss.backward().

        Args:
            loss: The loss tensor about to be backpropagated
        """
        pass

    def on_after_backward(self) -> None:
        """
        Called after loss.backward() but before optimizer.step().
        Useful for gradient inspection or modification.
        """
        pass

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Called before optimizer.step().

        Args:
            optimizer: The optimizer about to step
        """
        pass

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        Called at the beginning of each training batch.

        Args:
            batch: The current batch
            batch_idx: Index of the current batch
        """
        pass

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """
        Called at the end of each training batch.

        Args:
            outputs: Output from training_step
            batch: The current batch
            batch_idx: Index of the current batch
        """
        pass

    def on_validation_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        Called at the beginning of each validation batch.
        """
        pass

    def on_validation_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """
        Called at the end of each validation batch.
        """
        pass

    def on_test_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        Called at the beginning of each test batch.
        """
        pass

    def on_test_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int
    ) -> None:
        """
        Called at the end of each test batch.
        """
        pass

    # ==================== Dataloader Methods ====================
    # These should be overridden by subclasses that need custom dataloaders

    def train_dataloader(self) -> Any:
        """
        Returns the training dataloader.
        Override this method to provide a custom training dataloader.
        """
        raise NotImplementedError("Subclass must implement train_dataloader()")

    def val_dataloader(self) -> Any:
        """
        Returns the validation dataloader.
        Override this method to provide a custom validation dataloader.
        """
        raise NotImplementedError("Subclass must implement val_dataloader()")

    def test_dataloader(self) -> Any:
        """
        Returns the test dataloader.
        Override this method to provide a custom test dataloader.
        """
        raise NotImplementedError("Subclass must implement test_dataloader()")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called at the beginning of fit/test/validate.
        Use this to setup datasets or perform other initialization.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Called at the end of fit/test/validate.
        Use this to cleanup resources.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        pass

    # ==================== Logging Methods ====================

    def log(
        self,
        name: str,
        value: Any,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: str = 'mean',
        sync_dist: bool = False,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log a metric value.

        Args:
            name: Name of the metric
            value: Value to log (scalar tensor or number)
            prog_bar: Whether to show in progress bar
            logger: Whether to log to the logger (wandb, etc.)
            on_step: Whether to log at each step
            on_epoch: Whether to log at epoch end
            reduce_fx: Reduction function for distributed ('mean', 'sum', etc.)
            sync_dist: Whether to synchronize across distributed processes
            batch_size: Batch size for weighted averaging
        """
        # Convert tensor to scalar if needed
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.detach()
            # Keep multi-element tensors as-is for histogram logging

        # Store for per-batch retrieval (used by _run_validation's averaging logic)
        self._logged_metrics[name] = value

        if logger and self.logger is not None:
            if self._is_validating:
                # Buffer validation metrics; flushed once at on_validation_epoch_end
                if name not in self._val_log_buffer:
                    self._val_log_buffer[name] = []
                self._val_log_buffer[name].append(value)
            else:
                # Training: log immediately
                if hasattr(self.logger, 'log_metrics'):
                    self.logger.log_metrics({name: value}, step=self.global_step)
                elif hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                    self.logger.experiment.log({name: value}, step=self.global_step)

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        reduce_fx: str = 'mean',
        sync_dist: bool = False,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            dictionary: Dictionary of metric names to values
            prog_bar: Whether to show in progress bar
            logger: Whether to log to the logger
            on_step: Whether to log at each step
            on_epoch: Whether to log at epoch end
            reduce_fx: Reduction function for distributed
            sync_dist: Whether to synchronize across distributed processes
            batch_size: Batch size for weighted averaging
        """
        for name, value in dictionary.items():
            self.log(
                name,
                value,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                reduce_fx=reduce_fx,
                sync_dist=sync_dist,
                batch_size=batch_size,
                **kwargs
            )

    def get_logged_metrics(self) -> Dict[str, Any]:
        """
        Returns all metrics logged during the current step.
        """
        return self._logged_metrics.copy()

    def clear_logged_metrics(self) -> None:
        """
        Clears the logged metrics. Called by the trainer after each step.
        """
        self._logged_metrics.clear()

    # ==================== Checkpoint Methods ====================

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        map_location: Optional[Union[str, torch.device]] = None,
        hparams: Optional[Any] = None,
        strict: bool = True,
        **kwargs
    ) -> 'LightningModule':
        """
        Load a model from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
            map_location: Device to map the checkpoint to
            hparams: Optional hyperparameters to override
            strict: Whether to strictly enforce state_dict key matching
            **kwargs: Additional arguments passed to the model constructor

        Returns:
            Loaded model instance
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        # Get hyperparameters from checkpoint or use provided ones
        if hparams is not None:
            # Use provided hparams
            if hasattr(hparams, '__dict__'):
                hparams_dict = vars(hparams)
            elif isinstance(hparams, dict):
                hparams_dict = hparams
            else:
                hparams_dict = hparams
        elif 'hyper_parameters' in checkpoint:
            hparams_dict = checkpoint['hyper_parameters']
        elif 'hparams' in checkpoint:
            hparams_dict = checkpoint['hparams']
        else:
            hparams_dict = {}

        # Create model instance
        model = cls(hparams_dict, **kwargs)

        # Load state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=strict)
        else:
            model.load_state_dict(checkpoint, strict=strict)

        return model

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called when saving a checkpoint. Override to add custom data.

        Args:
            checkpoint: The checkpoint dictionary being saved
        """
        pass

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Called when loading a checkpoint. Override to load custom data.

        Args:
            checkpoint: The checkpoint dictionary being loaded
        """
        pass

    # ==================== Utility Methods ====================

    def freeze(self) -> None:
        """
        Freeze all parameters for inference.
        Sets requires_grad=False for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def unfreeze(self) -> None:
        """
        Unfreeze all parameters for training.
        Sets requires_grad=True for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def print(self, *args, **kwargs) -> None:
        """
        Print only on rank 0 in distributed training.
        """
        if self.global_rank == 0:
            print(*args, **kwargs)

    def to_torchscript(
        self,
        file_path: Optional[str] = None,
        method: str = 'script',
        example_inputs: Optional[Any] = None,
        **kwargs
    ) -> torch.jit.ScriptModule:
        """
        Convert the model to TorchScript.

        Args:
            file_path: Optional path to save the scripted model
            method: Either 'script' or 'trace'
            example_inputs: Example inputs for tracing

        Returns:
            TorchScript module
        """
        self.eval()

        if method == 'script':
            scripted = torch.jit.script(self, **kwargs)
        elif method == 'trace':
            if example_inputs is None:
                raise ValueError("example_inputs required for tracing")
            scripted = torch.jit.trace(self, example_inputs, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        if file_path is not None:
            torch.jit.save(scripted, file_path)

        return scripted

    def __repr__(self) -> str:
        """String representation showing model architecture."""
        return super().__repr__()
