"""
Pure PyTorch reimplementation of pytorch_lightning.Trainer.

Provides a drop-in replacement for Lightning's Trainer class, supporting:
  - Single-GPU and multi-GPU (DDP via torchrun) training
  - Mixed-precision training (32-true, 16-mixed, bf16-mixed)
  - Gradient accumulation and gradient clipping
  - Validation at configurable intervals
  - Checkpoint saving/loading and training resumption
  - max_steps-based training termination
  - Overfit-batches, fast-dev-run, limit-batches, and other debug modes
  - LightningModule lifecycle hooks
  - WandbLogger and ModelCheckpoint callback integration

No pytorch_lightning imports are used anywhere in this file.
"""

import os
import sys
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Union
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm

from claude_class.torchlightning_function import DDPStrategy, ModelCheckpoint


# ============================================================================
# ModelSummary callback
# ============================================================================

class ModelSummary:
    """
    Callback that prints a summary of the model architecture.

    Mimics pytorch_lightning.callbacks.ModelSummary. Prints layer names,
    types, parameter counts, and shapes.

    Args:
        max_depth: How deep to recurse into nested modules.
                   -1 means unlimited depth.
    """

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth

    def on_fit_start(self, model: torch.nn.Module) -> None:
        """Print model summary at the start of training."""
        self._print_summary(model)

    def _print_summary(self, model: torch.nn.Module) -> None:
        """Print a tabular model summary with parameter counts."""
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0

        print("=" * 80)
        print(f"{'Layer':<50} {'Type':<20} {'Params':>10}")
        print("-" * 80)

        for name, module in model.named_modules():
            # Respect max_depth: count dots in name to determine depth
            depth = name.count('.') + 1 if name else 0
            if self.max_depth != -1 and depth > self.max_depth:
                continue

            # Count only direct parameters (not from sub-modules)
            direct_params = sum(p.numel() for p in module.parameters(recurse=False))
            if direct_params > 0 or depth == 0:
                type_name = type(module).__name__
                display_name = name if name else "(root)"
                print(f"{display_name:<50} {type_name:<20} {direct_params:>10,}")

        for p in model.parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
            else:
                non_trainable_params += p.numel()

        print("-" * 80)
        print(f"Total params:         {total_params:>12,}")
        print(f"Trainable params:     {trainable_params:>12,}")
        print(f"Non-trainable params: {non_trainable_params:>12,}")
        print("=" * 80)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    """
    Pure PyTorch reimplementation of pytorch_lightning.Trainer.

    Manages the full training, validation, and testing lifecycle for a
    LightningModule. Handles DDP distribution, mixed precision, gradient
    accumulation, checkpointing, and logging.

    Constructor args match the subset used by this repository's train.py.
    """

    def __init__(
        self,
        accelerator: str = "auto",
        devices: Any = "auto",
        num_nodes: int = 1,
        precision: str = "32-true",
        max_steps: int = -1,
        max_epochs: int = 1000,
        logger: Any = None,
        enable_model_summary: bool = True,
        callbacks: Optional[List[Any]] = None,
        strategy: Any = "ddp",
        enable_checkpointing: bool = True,
        fast_dev_run: bool = False,
        num_sanity_val_steps: int = 0,
        limit_train_batches: Union[int, float] = 1.0,
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        detect_anomaly: bool = False,
        gradient_clip_val: Optional[float] = None,
        overfit_batches: Union[int, float] = 0,
        profiler: Optional[str] = None,
        val_check_interval: Union[int, float] = 1.0,
        check_val_every_n_epoch: int = 1,
        deterministic: bool = False,
        log_every_n_steps: int = 50,
        accumulate_grad_batches: int = 1,
        inference_mode: bool = True,
    ):
        # ---- Store all configuration ----
        self.accelerator = accelerator
        self._devices_arg = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.logger = logger
        self.enable_model_summary = enable_model_summary
        self.callbacks = callbacks or []
        self.strategy = strategy
        self.enable_checkpointing = enable_checkpointing
        self.fast_dev_run = fast_dev_run
        self.num_sanity_val_steps = num_sanity_val_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.limit_test_batches = limit_test_batches
        self.detect_anomaly = detect_anomaly
        self.gradient_clip_val = gradient_clip_val
        self.overfit_batches = overfit_batches
        self.profiler = profiler
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.deterministic = deterministic
        self.log_every_n_steps = log_every_n_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.inference_mode = inference_mode

        # ---- Runtime state (set during fit/test) ----
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.global_rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.optimizers: List[torch.optim.Optimizer] = []
        self._schedulers: List[Dict[str, Any]] = []
        self._model = None          # The unwrapped LightningModule
        self._ddp_model = None      # DDP-wrapped model (or same as _model)
        self._device = None
        self._should_stop = False

        # ---- DDP configuration from strategy ----
        self._find_unused_parameters = False
        if isinstance(self.strategy, DDPStrategy):
            self._find_unused_parameters = self.strategy.find_unused_parameters

        # ---- Precision / autocast setup ----
        self._autocast_dtype = None  # None means no autocast
        self._use_grad_scaler = False
        if precision in ("16-mixed", "16"):
            self._autocast_dtype = torch.float16
            self._use_grad_scaler = True
        elif precision in ("bf16-mixed", "bf16"):
            self._autocast_dtype = torch.bfloat16
            self._use_grad_scaler = False  # bf16 doesn't need loss scaling

        # ---- Determinism ----
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ---- Anomaly detection ----
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # Override: fast_dev_run forces 1 train + 1 val batch
        if self.fast_dev_run:
            self.max_steps = 1
            self.limit_train_batches = 1
            self.limit_val_batches = 1
            self.limit_test_batches = 1
            self.num_sanity_val_steps = 0
            self.max_epochs = 1

    # ====================================================================
    # Distributed setup helpers
    # ====================================================================

    def _setup_distributed(self) -> None:
        """
        Detect and initialize distributed training environment.

        If launched via torchrun, environment variables RANK, LOCAL_RANK,
        and WORLD_SIZE are already set. We initialize the NCCL process
        group and set rank/device information accordingly.
        """
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.global_rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.world_size = int(os.environ["WORLD_SIZE"])

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            torch.cuda.set_device(self.local_rank)
            self._device = torch.device("cuda", self.local_rank)
        else:
            # Single-process mode
            self.global_rank = 0
            self.local_rank = 0
            self.world_size = 1
            if torch.cuda.is_available():
                self._device = torch.device("cuda", 0)
            else:
                self._device = torch.device("cpu")

    @property
    def is_global_zero(self) -> bool:
        """True if this is the rank-0 process (or single-GPU)."""
        return self.global_rank == 0

    def _print_rank0(self, *args, **kwargs) -> None:
        """Print only on the master process."""
        if self.is_global_zero:
            print(*args, **kwargs)

    # ====================================================================
    # Precision context manager
    # ====================================================================

    def _autocast_ctx(self):
        """
        Returns an autocast context manager for mixed-precision training.

        For "32-true" precision, returns a no-op context.
        For "16-mixed", returns torch.amp.autocast(float16).
        For "bf16-mixed", returns torch.amp.autocast(bfloat16).
        """
        if self._autocast_dtype is not None:
            return torch.amp.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return nullcontext()

    # ====================================================================
    # Dataloader utilities
    # ====================================================================

    def _limit_batches(self, dataloader: DataLoader, limit: Union[int, float]) -> int:
        """
        Compute the effective number of batches given a limit.

        Args:
            dataloader: The dataloader to limit.
            limit: If float <= 1.0, treated as a fraction of total batches.
                   If int > 1, treated as an absolute batch count.

        Returns:
            The number of batches to iterate over.
        """
        total = len(dataloader)
        if isinstance(limit, float) and limit <= 1.0:
            return max(1, int(total * limit))
        else:
            return min(int(limit), total)

    def _wrap_dataloader_for_ddp(self, dataloader: DataLoader) -> DataLoader:
        """
        Wrap a dataloader with DistributedSampler for DDP training.

        Creates a new DataLoader that uses a DistributedSampler, preserving
        the original batch_size, num_workers, collate_fn, etc.
        """
        if self.world_size <= 1:
            return dataloader

        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
        )

        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            persistent_workers=dataloader.persistent_workers
            if dataloader.num_workers > 0
            else False,
            prefetch_factor=dataloader.prefetch_factor
            if dataloader.num_workers > 0
            else None,
        )

    # ====================================================================
    # Optimizer / Scheduler configuration
    # ====================================================================

    def _configure_optimizers(self, model) -> None:
        """
        Call model.configure_optimizers() and parse the result.

        Supports the dict format returned by this repository:
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        """
        opt_config = model.configure_optimizers()

        if isinstance(opt_config, torch.optim.Optimizer):
            self.optimizers = [opt_config]
            self._schedulers = []
        elif isinstance(opt_config, dict):
            self.optimizers = [opt_config["optimizer"]]
            if "lr_scheduler" in opt_config:
                sched_cfg = opt_config["lr_scheduler"]
                if isinstance(sched_cfg, dict):
                    self._schedulers = [sched_cfg]
                else:
                    self._schedulers = [{"scheduler": sched_cfg, "interval": "epoch", "frequency": 1}]
            else:
                self._schedulers = []
        elif isinstance(opt_config, (list, tuple)):
            if len(opt_config) == 2 and isinstance(opt_config[0], list):
                self.optimizers = opt_config[0]
                self._schedulers = [
                    {"scheduler": s, "interval": "epoch", "frequency": 1}
                    for s in opt_config[1]
                ]
            else:
                self.optimizers = list(opt_config)
                self._schedulers = []
        else:
            raise ValueError(f"Unsupported return type from configure_optimizers: {type(opt_config)}")

    def _step_schedulers(self, interval: str) -> None:
        """
        Step all LR schedulers that match the given interval ('step' or 'epoch').
        """
        for sched_cfg in self._schedulers:
            if sched_cfg.get("interval", "epoch") == interval:
                freq = sched_cfg.get("frequency", 1)
                if interval == "step" and self.global_step % freq == 0:
                    sched_cfg["scheduler"].step()
                elif interval == "epoch":
                    sched_cfg["scheduler"].step()

    # ====================================================================
    # Checkpoint resume
    # ====================================================================

    def _resume_from_checkpoint(self, model, ckpt_path: str) -> None:
        """
        Resume training from a saved checkpoint.

        Restores model state_dict, optimizer states, scheduler states,
        epoch, and global_step from the checkpoint file.

        Args:
            model: The LightningModule being trained.
            ckpt_path: Path to the checkpoint file.
        """
        self._print_rank0(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        # Restore model weights
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        # Restore optimizer states
        if "optimizer_states" in checkpoint:
            for opt, opt_state in zip(self.optimizers, checkpoint["optimizer_states"]):
                opt.load_state_dict(opt_state)

        # Restore scheduler states
        if "lr_scheduler_states" in checkpoint:
            for sched_cfg, sched_state in zip(self._schedulers, checkpoint["lr_scheduler_states"]):
                sched_cfg["scheduler"].load_state_dict(sched_state)

        # Restore epoch and step
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        # Call model hook
        if hasattr(model, "on_load_checkpoint"):
            model.on_load_checkpoint(checkpoint)

        self._print_rank0(
            f"Resumed at epoch={self.current_epoch}, global_step={self.global_step}"
        )

    # ====================================================================
    # Validation loop
    # ====================================================================

    def _run_validation(self, model, val_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Run the full validation loop.

        Calls model.validation_step() for each batch and collects logged
        metrics. EBT's forward() uses torch.set_grad_enabled(True)
        internally for MCMC gradient computation, so we do NOT use
        torch.no_grad() here — the model manages its own grad context.

        Args:
            model: The LightningModule (unwrapped).
            val_dataloader: The validation DataLoader.

        Returns:
            Dictionary of aggregated metric values from the validation run.
        """
        model.eval()
        model.on_validation_epoch_start()

        num_val_batches = self._limit_batches(val_dataloader, self.limit_val_batches)

        all_metrics = {}
        count = 0

        for batch_idx, batch in enumerate(val_dataloader):
            if batch_idx >= num_val_batches:
                break

            batch = self._move_batch_to_device(batch)
            model.on_validation_batch_start(batch, batch_idx)

            # Do NOT wrap in no_grad: EBT's forward() needs gradients for
            # the MCMC optimization loop (it uses torch.set_grad_enabled
            # and torch.autograd.grad internally). The model itself handles
            # gradient context.
            output = model.validation_step(batch, batch_idx)

            model.on_validation_batch_end(output, batch, batch_idx)

            # Collect logged metrics
            step_metrics = model.get_logged_metrics()
            for k, v in step_metrics.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    v = v.item()
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
            count += 1
            model.clear_logged_metrics()

        # Average the collected metrics over batches
        if count > 0:
            for k in all_metrics:
                all_metrics[k] /= count

        model.on_validation_epoch_end()
        model.train()

        return all_metrics

    # ====================================================================
    # Test loop
    # ====================================================================

    def test(self, model, datamodule=None) -> None:
        """
        Run the test loop.

        Calls model.setup("test"), gets the test dataloader, and runs
        model.test_step() for each batch. For EBT models,
        inference_mode=False means we don't wrap in no_grad/inference_mode
        to allow gradient computation for MCMC.

        Args:
            model: The LightningModule to test.
            datamodule: Unused, kept for interface compatibility.
        """
        self._setup_distributed()

        # Attach trainer reference to model
        model.trainer = self
        model.to(self._device)
        self._model = model

        # Setup and get dataloader
        model.setup("test")
        test_dl = model.test_dataloader()
        if self.world_size > 1:
            test_dl = self._wrap_dataloader_for_ddp(test_dl)

        num_test_batches = self._limit_batches(test_dl, self.limit_test_batches)

        # During testing, optimizers list is empty
        self.optimizers = []

        model.on_test_epoch_start()

        self._print_rank0(f"Testing: {num_test_batches} batches")

        # inference_mode=False means DON'T restrict gradients during test
        # (needed for EBT's MCMC gradient computation)
        if self.inference_mode:
            ctx = torch.inference_mode()
        else:
            ctx = nullcontext()

        with ctx:
            for batch_idx, batch in enumerate(tqdm(
                test_dl,
                total=num_test_batches,
                desc="Testing",
                disable=not self.is_global_zero,
            )):
                if batch_idx >= num_test_batches:
                    break

                batch = self._move_batch_to_device(batch)
                model.on_test_batch_start(batch, batch_idx)
                output = model.test_step(batch, batch_idx)
                model.on_test_batch_end(output, batch, batch_idx)
                model.clear_logged_metrics()

        model.on_test_epoch_end()

        self._print_rank0("Testing complete.")

    # ====================================================================
    # Main training entry point
    # ====================================================================

    def fit(self, model, datamodule=None, ckpt_path: Optional[str] = None) -> None:
        """
        Run the full training loop.

        This is the main entry point, equivalent to Lightning's trainer.fit().

        Lifecycle:
          1. Setup distributed environment
          2. Move model to device, wrap in DDP if needed
          3. Configure optimizers and schedulers
          4. Resume from checkpoint if specified
          5. Run optional sanity validation
          6. Training loop: epoch -> batch -> backward -> optimizer step
          7. Periodic validation with checkpoint saving

        Args:
            model: The LightningModule to train.
            datamodule: Unused, kept for interface compatibility.
            ckpt_path: Optional path to a checkpoint to resume from.
        """
        self._setup_distributed()

        # Attach trainer reference so model can access self.trainer.*
        model.trainer = self
        model.to(self._device)
        self._model = model

        # Print model summary if enabled
        if self.enable_model_summary and self.is_global_zero:
            for cb in self.callbacks:
                if isinstance(cb, ModelSummary):
                    cb.on_fit_start(model)
                    break

        # Setup dataloaders
        model.setup("fit")
        train_dl = model.train_dataloader()
        val_dl = model.val_dataloader()

        # Wrap dataloaders for DDP
        train_dl = self._wrap_dataloader_for_ddp(train_dl)
        if self.world_size > 1:
            val_dl = self._wrap_dataloader_for_ddp(val_dl)

        # Handle overfit_batches: restrict training to a small subset
        if self.overfit_batches > 0:
            if isinstance(self.overfit_batches, float) and self.overfit_batches < 1.0:
                n = max(1, int(len(train_dl.dataset) * self.overfit_batches))
            else:
                n = int(self.overfit_batches)
            # Limit the number of batches
            self.limit_train_batches = n

        # Configure optimizers and schedulers from the model
        self._configure_optimizers(model)

        # Initialize GradScaler for mixed precision if needed
        scaler = None
        if self._use_grad_scaler:
            scaler = torch.amp.GradScaler("cuda")

        # Wrap model in DDP if distributed
        if self.world_size > 1:
            self._ddp_model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=self._find_unused_parameters,
            )
        else:
            self._ddp_model = model

        # Resume from checkpoint if provided
        if ckpt_path is not None:
            self._resume_from_checkpoint(model, ckpt_path)

        # ----- Sanity validation -----
        if self.num_sanity_val_steps > 0 and self.limit_val_batches != 0:
            self._print_rank0(f"Running {self.num_sanity_val_steps} sanity val steps...")
            saved_limit = self.limit_val_batches
            self.limit_val_batches = self.num_sanity_val_steps
            self._run_validation(model, val_dl)
            self.limit_val_batches = saved_limit

        # ----- Training loop -----
        model.train()
        model.on_train_start()

        self._should_stop = False
        start_epoch = self.current_epoch

        for epoch in range(start_epoch, self.max_epochs):
            if self._should_stop:
                break

            self.current_epoch = epoch
            model.on_train_epoch_start()

            # Set epoch on DistributedSampler for shuffling
            if hasattr(train_dl, "sampler") and isinstance(train_dl.sampler, DistributedSampler):
                train_dl.sampler.set_epoch(epoch)

            num_train_batches = self._limit_batches(train_dl, self.limit_train_batches)

            # Determine when to run validation within the epoch
            # val_check_interval can be a float (fraction of epoch)
            # or int (number of training steps)
            if isinstance(self.val_check_interval, float) and self.val_check_interval <= 1.0:
                val_every_n_batches = max(1, int(num_train_batches * self.val_check_interval))
            else:
                val_every_n_batches = int(self.val_check_interval)

            # Zero gradients at the start
            for opt in self.optimizers:
                opt.zero_grad()

            accum_count = 0  # Counts batches within an accumulation window

            for batch_idx, batch in enumerate(train_dl):
                if batch_idx >= num_train_batches:
                    break

                batch = self._move_batch_to_device(batch)
                model.on_train_batch_start(batch, batch_idx)

                # Forward pass with optional autocast
                with self._autocast_ctx():
                    loss = model.training_step(batch, batch_idx)

                if loss is None:
                    continue

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.accumulate_grad_batches

                # Backward pass
                model.on_before_backward(scaled_loss)
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                model.on_after_backward()

                accum_count += 1

                # Optimizer step at the end of each accumulation window
                if accum_count >= self.accumulate_grad_batches:
                    # Gradient clipping
                    if self.gradient_clip_val is not None:
                        if scaler is not None:
                            scaler.unscale_(self.optimizers[0])
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), self.gradient_clip_val
                        )

                    model.on_before_optimizer_step(self.optimizers[0])

                    # Optimizer step
                    if scaler is not None:
                        for opt in self.optimizers:
                            scaler.step(opt)
                        scaler.update()
                    else:
                        for opt in self.optimizers:
                            opt.step()

                    # Zero gradients for next accumulation window
                    for opt in self.optimizers:
                        opt.zero_grad()

                    # Step per-step LR schedulers
                    self._step_schedulers("step")

                    self.global_step += 1
                    accum_count = 0

                # Call on_train_batch_end hook
                model.on_train_batch_end(loss, batch, batch_idx)
                model.clear_logged_metrics()

                # Check val_check_interval within epoch
                # Only validate if enough batches have passed since last check
                effective_batch = batch_idx + 1
                if (
                    self.limit_val_batches != 0
                    and effective_batch % val_every_n_batches == 0
                    and effective_batch < num_train_batches  # Don't double-validate at end
                    and epoch % self.check_val_every_n_epoch == 0
                ):
                    val_metrics = self._run_validation(model, val_dl)
                    self._try_checkpoint(model, val_metrics)
                    model.train()

                # Check max_steps termination
                if self.max_steps > 0 and self.global_step >= self.max_steps:
                    self._should_stop = True
                    break

            # End-of-epoch hooks
            model.on_train_epoch_end()
            self._step_schedulers("epoch")

            # End-of-epoch validation
            if (
                self.limit_val_batches != 0
                and epoch % self.check_val_every_n_epoch == 0
            ):
                val_metrics = self._run_validation(model, val_dl)
                self._try_checkpoint(model, val_metrics)
                model.train()

            # Check max_steps termination at epoch boundary too
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                self._should_stop = True

        model.on_train_end()
        self._print_rank0(
            f"Training complete. Final step={self.global_step}, epoch={self.current_epoch}"
        )

    # ====================================================================
    # Checkpoint helper
    # ====================================================================

    def _try_checkpoint(self, model, val_metrics: Dict[str, Any]) -> None:
        """
        Invoke all ModelCheckpoint callbacks with the current validation metrics.

        Passes optimizer and scheduler state_dicts for full resumability.

        Args:
            model: The LightningModule being trained.
            val_metrics: Dictionary of validation metrics from the last run.
        """
        if not self.enable_checkpointing:
            return
        if not self.is_global_zero:
            return

        opt_states = [opt.state_dict() for opt in self.optimizers]
        sched_states = [s["scheduler"].state_dict() for s in self._schedulers]

        for cb in self.callbacks:
            if isinstance(cb, ModelCheckpoint):
                cb.on_validation_end(
                    model=model,
                    metrics=val_metrics,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    optimizer_states=opt_states,
                    lr_scheduler_states=sched_states,
                )

    # ====================================================================
    # Utility
    # ====================================================================

    def _move_batch_to_device(self, batch: Any) -> Any:
        """
        Recursively move a batch of data to the training device.

        Handles tensors, dicts, lists, and tuples.

        Args:
            batch: The data batch from the DataLoader.

        Returns:
            The batch with all tensors moved to self._device.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self._device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            moved = [self._move_batch_to_device(item) for item in batch]
            return type(batch)(moved)
        return batch



class IterableTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)
    
    def _run_validation(self, *args, **kwargs):
        return super()._run_validation(*args, **kwargs)

    def test(self, *args, **kwargs):
        return super().test(*args, **kwargs)

    def _resume_from_checkpoint(self, *args, **kwargs):
        return super()._resume_from_checkpoint(*args, **kwargs)




