"""
Pure PyTorch reimplementations of pytorch_lightning utility classes and functions.

This module provides drop-in replacements for:
    - pytorch_lightning.strategies.DDPStrategy
    - pytorch_lightning.seed_everything
    - pytorch_lightning.utilities.rank_zero.rank_zero_only
    - pytorch_lightning.loggers.WandbLogger
    - pytorch_lightning.callbacks.ModelCheckpoint

No pytorch_lightning imports are used anywhere in this file.
"""

import os
import re
import random
import functools
import warnings
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, List

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer

import ipdb

# ModelSummary is defined in torchlightning_trainer.py but imported from here
# for convenience (train.py imports it alongside ModelCheckpoint).
# Lazy import to avoid circular dependency at module level.
def __getattr__(name):
    if name == "ModelSummary":
        from claude_class.torchlightning_trainer import ModelSummary
        return ModelSummary
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the _DummyExperiment."""

# ============================================================================
# seed_everything
# ============================================================================

def seed_everything(seed: int = 42, workers: bool = False) -> int:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Mirrors pytorch_lightning.seed_everything. Seeds:
      - Python's built-in random module
      - NumPy's random generator
      - PyTorch CPU random generator
      - All CUDA device random generators (if available)

    Args:
        seed: The seed value. Will be clamped to a valid range.
        workers: If True, sets the PL_GLOBAL_SEED and PL_SEED_WORKERS
                 environment variables so that DataLoader workers can be
                 seeded deterministically via worker_init_fn.

    Returns:
        The seed value used.
    """
    # Clamp seed to valid 32-bit range (matching Lightning behavior)
    max_seed = np.iinfo(np.uint32).max
    min_seed = np.iinfo(np.uint32).min
    if not (min_seed <= seed <= max_seed):
        warnings.warn(f"seed {seed} is out of range [{min_seed}, {max_seed}], clipping")
        seed = max(min(seed, max_seed), min_seed)

    # Seed Python's built-in random module
    random.seed(seed)

    # Seed NumPy
    np.random.seed(seed)

    # Seed PyTorch CPU
    torch.manual_seed(seed)

    # Seed all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set environment variables for DataLoader worker seeding
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    if workers:
        os.environ["PL_SEED_WORKERS"] = "1"
    else:
        os.environ.pop("PL_SEED_WORKERS", None)

    return seed


# ============================================================================
# rank_zero_only
# ============================================================================

def _get_rank() -> int:
    """
    Determine the current process rank in distributed training.

    Checks environment variables in priority order:
      1. RANK (set by torchrun / torch.distributed.launch)
      2. SLURM_PROCID (set by SLURM)
      3. LOCAL_RANK (fallback)
    Returns 0 if no distributed environment is detected.
    """
    # Check standard PyTorch distributed rank variable first
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank)

    # Check SLURM environment
    rank = os.environ.get("SLURM_PROCID")
    if rank is not None:
        return int(rank)

    # Check local rank as fallback
    rank = os.environ.get("LOCAL_RANK")
    if rank is not None:
        return int(rank)

    return 0


class _RankZeroOnly:
    """
    Decorator class that ensures a function only executes on rank 0.

    On non-zero ranks the decorated function returns None immediately.
    This is used to prevent duplicate side effects (e.g., wandb init,
    printing) from occurring on every GPU during DDP training.
    """

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            rank = _get_rank()
            if rank == 0:
                return fn(*args, **kwargs)
            return None
        return wrapped_fn


# Singleton instance used as a decorator: @rank_zero_only
rank_zero_only = _RankZeroOnly()



# ============================================================================
# DDPStrategy
# ============================================================================

class DDPStrategy:
    """
    Configuration holder for PyTorch DistributedDataParallel (DDP) strategy.

    This is a simplified version of pytorch_lightning.strategies.DDPStrategy
    that stores DDP configuration parameters. The actual DDP wrapping is
    performed by the Trainer, which reads these settings.

    In this repository, it is used exclusively to set find_unused_parameters:
        DDPStrategy(find_unused_parameters=True)

    Args:
        find_unused_parameters: If True, DDP will detect parameters that
            do not receive gradients during backward and skip their
            gradient synchronization. Useful for debugging but adds
            overhead — should not be left on for production training.
        gradient_as_bucket_view: If True, gradients are views into
            allreduce communication buckets, reducing peak memory.
        static_graph: If True, tells DDP the computation graph is static
            across iterations, enabling optimizations.
    """

    def __init__(
        self,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False,
        **kwargs,
    ):
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph
        # Store any extra kwargs for forward compatibility
        self._kwargs = kwargs

    def __repr__(self) -> str:
        return (
            f"DDPStrategy("
            f"find_unused_parameters={self.find_unused_parameters}, "
            f"gradient_as_bucket_view={self.gradient_as_bucket_view}, "
            f"static_graph={self.static_graph})"
        )


# ============================================================================
# WandbLogger
# ============================================================================

class WandbLogger:
    """
    A lightweight Weights & Biases logger compatible with the Lightning API.

    Wraps a wandb.Run object and exposes it via the .experiment property.
    If an existing run is passed via the `experiment` parameter, that run
    is reused instead of creating a new one.

    Usage in this repository:
        wandb_logger = WandbLogger(
            save_dir="logs/", name="run_name",
            entity="team", project="EBT",
            offline=True, experiment=run
        )
        wandb_logger.experiment.tags = ["tag1"]
        wandb_logger.experiment.log({"metric": value})

    Args:
        save_dir: Directory to save wandb run files.
        name: Display name for the run.
        entity: Wandb team or user name.
        project: Wandb project name.
        offline: If True, run wandb in offline mode.
        experiment: An existing wandb.Run to reuse. If None, a new run
                    is initialized.
        **kwargs: Additional keyword arguments passed to wandb.init().
    """

    def __init__(
        self,
        save_dir: Optional[str] = None,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        project: Optional[str] = None,
        offline: bool = False,
        experiment: Optional[Any] = None,
        **kwargs,
    ):
        self._save_dir = save_dir
        self._name = name
        self._entity = entity
        self._project = project
        self._offline = offline
        self._experiment = experiment
        self._kwargs = kwargs

    @property
    def experiment(self) -> Any:
        """
        Returns the underlying wandb.Run object.

        If an existing run was passed at construction time, returns that.
        Otherwise lazily initializes a new wandb run on first access.
        """
        if self._experiment is not None:
            return self._experiment

        # Lazily initialize a new wandb run
        import wandb

        if wandb.run is not None:
            # Reuse the currently active global run
            self._experiment = wandb.run
        else:
            mode = "offline" if self._offline else "online"
            self._experiment = wandb.init(
                dir=self._save_dir,
                name=self._name,
                entity=self._entity,
                project=self._project,
                mode=mode,
                **self._kwargs,
            )
        return self._experiment

    @property
    def save_dir(self) -> Optional[str]:
        """Returns the save directory for wandb files."""
        return self._save_dir

    @property
    def name(self) -> Optional[str]:
        """Returns the run name."""
        return self._name

    @property
    def version(self) -> Optional[str]:
        """Returns the wandb run ID (version string)."""
        exp = self.experiment
        return exp.id if exp is not None else None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dictionary of scalar metrics to wandb.

        Args:
            metrics: Dictionary mapping metric names to scalar values.
            step: Optional global step number for the x-axis.
        """
        exp = self.experiment
        if exp is not None:
            if step is not None:
                exp.log(metrics, step=step)
            else:
                exp.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters to wandb config.

        Args:
            params: Dictionary of hyperparameter names to values.
        """
        exp = self.experiment
        if exp is not None:
            exp.config.update(params)

    def watch(
        self,
        model: "nn.Module",
        log: Optional[str] = "gradients",
        log_freq: int = 100,
        log_graph: bool = True,
    ) -> None:
        """
        Watch a model for gradient and parameter logging via wandb.

        Delegates to ``wandb.watch()`` on the underlying experiment.
        Call ``self.experiment.unwatch(model)`` to stop watching.

        Args:
            model: The model to watch.
            log: What to log. One of ``"gradients"``, ``"parameters"``,
                ``"all"``, or ``None``.
            log_freq: How often (in steps) to log.
            log_graph: Whether to log the computational graph.
        """
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    def finalize(self, status: str = "success") -> None:
        """
        Finalize the wandb run (mark as finished).

        Args:
            status: Final status of the run.
        """
        import wandb
        if self._experiment is not None:
            wandb.finish()

    def __repr__(self) -> str:
        return (
            f"WandbLogger(name={self._name!r}, project={self._project!r}, "
            f"entity={self._entity!r}, offline={self._offline})"
        )

# ============================================================================
# ModelCheckpoint
# ============================================================================

class ModelCheckpoint:
    """
    Callback that saves model checkpoints based on a monitored metric.

    Mimics pytorch_lightning.callbacks.ModelCheckpoint. Keeps track of the
    top-k checkpoints ranked by the monitored quantity, and optionally
    saves a "last.ckpt" after every validation epoch.

    Usage in this repository:
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_loss", mode="min", save_top_k=10,
            save_last=True,
            dirpath="./logs/checkpoints/run_name_2024-01-01/",
            filename="epoch={epoch}-step={step}-valid_loss={valid_loss:.4f}",
            verbose=True,
        )
        # After training:
        best_path = checkpoint_callback.best_model_path

    Args:
        monitor: Metric name to monitor (e.g. "valid_loss").
        mode: One of "min" or "max". Determines whether lower or higher
              values of the monitored metric are better.
        save_top_k: Number of best checkpoints to keep on disk. Set to
                    -1 to keep all checkpoints.
        save_last: If True, always saves a "last.ckpt" regardless of
                   the monitored metric.
        dirpath: Directory to save checkpoint files.
        filename: Checkpoint filename template. Supports {epoch}, {step},
                  and any logged metric names as format keys, e.g.
                  "epoch={epoch}-step={step}-val_loss={val_loss:.4f}".
        verbose: If True, prints a message each time a checkpoint is saved.
        every_n_epochs: Save a checkpoint every N epochs (default 1).
        every_n_train_steps: Save every N training steps (0 = disabled).
    """

    # Sentinel for "no metric observed yet"
    _INIT_BEST = {
        "min": float("inf"),
        "max": float("-inf"),
    }

    def __init__(
        self,
        monitor: Optional[str] = None,
        mode: str = "min",
        save_top_k: int = 1,
        save_last: bool = False,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
        verbose: bool = False,
        every_n_epochs: int = 1,
        every_n_train_steps: int = 0,
    ):
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"

        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.dirpath = dirpath
        self.filename = filename or "{epoch}-{step}"
        self.verbose = verbose
        self.every_n_epochs = every_n_epochs
        self.every_n_train_steps = every_n_train_steps

        # Best metric value seen so far
        self.best_model_score: Optional[float] = self._INIT_BEST[mode]

        # Path to the best checkpoint
        self.best_model_path: str = ""

        # Path to the last saved checkpoint
        self.last_model_path: str = ""

        # List of (score, path) tuples for top-k tracking, sorted best-first
        self._top_k_checkpoints: List[tuple] = []

        # Comparison operator based on mode
        if mode == "min":
            self._is_better = lambda current, best: current < best
        else:
            self._is_better = lambda current, best: current > best

    def _format_filename(self, metrics: Dict[str, Any]) -> str:
        """
        Format the filename template with metric values.

        Handles Lightning-style templates like:
            "epoch={epoch}-step={step}-valid_loss={valid_loss:.4f}"

        The template uses Python str.format style, but metric names
        inside braces act as keys into the metrics dict.
        """
        filename = self.filename

        # Replace Lightning-style {metric_name} with actual values
        # First, find all format placeholders like {key} or {key:.4f}
        pattern = r"\{(\w+)(?::([^}]*))?\}"

        def _replace(match):
            key = match.group(1)
            fmt_spec = match.group(2)
            if key in metrics:
                val = metrics[key]
                if fmt_spec:
                    return f"{val:{fmt_spec}}"
                else:
                    return str(val)
            # If key not found in metrics, leave the placeholder unchanged
            return match.group(0)

        formatted = re.sub(pattern, _replace, filename)
        return formatted

    def _save_checkpoint(
        self,
        model,
        filepath: str,
        epoch: int,
        global_step: int,
        optimizer_states: Optional[list] = None,
        lr_scheduler_states: Optional[list] = None,
    ) -> None:
        """
        Save a checkpoint to disk.

        The checkpoint dictionary follows the Lightning convention:
            - 'state_dict': model parameters
            - 'hyper_parameters': model hyperparameters (from hparams)
            - 'epoch': current epoch
            - 'global_step': current global step
            - 'optimizer_states': list of optimizer state dicts
            - 'lr_scheduler_states': list of scheduler state dicts

        Args:
            model: The LightningModule to save.
            filepath: Full path where the checkpoint will be written.
            epoch: Current epoch number.
            global_step: Current global step count.
            optimizer_states: Optional list of optimizer state dicts.
            lr_scheduler_states: Optional list of scheduler state dicts.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Build checkpoint dict
        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }

        # Save hyperparameters if available
        if hasattr(model, "hparams") and model.hparams:
            checkpoint["hyper_parameters"] = dict(model.hparams)

        # Save optimizer states
        if optimizer_states is not None:
            checkpoint["optimizer_states"] = optimizer_states

        # Save scheduler states
        if lr_scheduler_states is not None:
            checkpoint["lr_scheduler_states"] = lr_scheduler_states

        # Call model hook if it exists
        if hasattr(model, "on_save_checkpoint"):
            model.on_save_checkpoint(checkpoint)

        torch.save(checkpoint, filepath)

    def on_validation_end(
        self,
        model,
        metrics: Dict[str, Any],
        epoch: int,
        global_step: int,
        optimizer_states: Optional[list] = None,
        lr_scheduler_states: Optional[list] = None,
    ) -> None:
        """
        Called at the end of validation to potentially save a checkpoint.

        This is the main entry point invoked by the Trainer after each
        validation pass. It checks whether the current metric is among
        the top-k, saves the checkpoint if so, and prunes old checkpoints.

        Args:
            model: The LightningModule being trained.
            metrics: Dictionary of all logged metrics for this validation.
            epoch: Current epoch number.
            global_step: Current global training step.
            optimizer_states: Optional optimizer state dicts for resumability.
            lr_scheduler_states: Optional scheduler state dicts.
        """
        # Build metrics dict for filename formatting
        fmt_metrics = {**metrics, "epoch": epoch, "step": global_step}

        # Save "last.ckpt" if configured
        if self.save_last:
            last_path = os.path.join(self.dirpath, "last.ckpt")
            self._save_checkpoint(
                model, last_path, epoch, global_step,
                optimizer_states, lr_scheduler_states,
            )
            self.last_model_path = last_path

        # If no monitor is set, just save every time
        if self.monitor is None:
            filepath = os.path.join(
                self.dirpath,
                self._format_filename(fmt_metrics) + ".ckpt",
            )
            self._save_checkpoint(
                model, filepath, epoch, global_step,
                optimizer_states, lr_scheduler_states,
            )
            self.best_model_path = filepath
            return

        # Get current metric value
        current = metrics.get(self.monitor)
        if current is None:
            warnings.warn(
                f"ModelCheckpoint: monitored metric '{self.monitor}' not found "
                f"in logged metrics. Available: {list(metrics.keys())}. "
                f"Skipping checkpoint."
            )
            return

        # Convert tensor to float
        if isinstance(current, torch.Tensor):
            current = current.item()

        filepath = os.path.join(
            self.dirpath,
            self._format_filename(fmt_metrics) + ".ckpt",
        )

        # Determine if we should save this checkpoint
        if self.save_top_k == -1:
            # Save all checkpoints
            self._save_checkpoint(
                model, filepath, epoch, global_step,
                optimizer_states, lr_scheduler_states,
            )
            if self._is_better(current, self.best_model_score):
                self.best_model_score = current
                self.best_model_path = filepath
            self._top_k_checkpoints.append((current, filepath))
            if self.verbose:
                print(
                    f"ModelCheckpoint: saved checkpoint '{filepath}' "
                    f"({self.monitor}={current:.4f})"
                )
            return

        if self.save_top_k == 0:
            # Don't save any metric-based checkpoints
            return

        # Check if this score makes it into the top-k
        should_save = len(self._top_k_checkpoints) < self.save_top_k
        if not should_save:
            # Find the worst checkpoint in top-k
            worst_idx = self._find_worst_index()
            worst_score = self._top_k_checkpoints[worst_idx][0]
            should_save = self._is_better(current, worst_score)

        if should_save:
            self._save_checkpoint(
                model, filepath, epoch, global_step,
                optimizer_states, lr_scheduler_states,
            )

            if self.verbose:
                print(
                    f"ModelCheckpoint: saved checkpoint '{filepath}' "
                    f"({self.monitor}={current:.4f})"
                )

            # Update best if this is the best score
            if self._is_better(current, self.best_model_score):
                self.best_model_score = current
                self.best_model_path = filepath

            # Add to top-k list
            self._top_k_checkpoints.append((current, filepath))

            # If we've exceeded top-k, remove the worst checkpoint
            if len(self._top_k_checkpoints) > self.save_top_k:
                worst_idx = self._find_worst_index()
                _, worst_path = self._top_k_checkpoints.pop(worst_idx)

                # Delete the worst checkpoint file from disk
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    if self.verbose:
                        print(
                            f"ModelCheckpoint: removed old checkpoint "
                            f"'{worst_path}'"
                        )

    def _find_worst_index(self) -> int:
        """
        Find the index of the worst checkpoint in the top-k list.

        "Worst" means the highest score when mode="min", or the lowest
        score when mode="max".
        """
        if self.mode == "min":
            # Worst = largest value
            return max(
                range(len(self._top_k_checkpoints)),
                key=lambda i: self._top_k_checkpoints[i][0],
            )
        else:
            # Worst = smallest value
            return min(
                range(len(self._top_k_checkpoints)),
                key=lambda i: self._top_k_checkpoints[i][0],
            )

    def __repr__(self) -> str:
        return (
            f"ModelCheckpoint(monitor={self.monitor!r}, mode={self.mode!r}, "
            f"save_top_k={self.save_top_k}, save_last={self.save_last}, "
            f"dirpath={self.dirpath!r})"
        )
