"""
IterableTrainer: A standalone trainer for PyTorch IterableDataset.

Does not inherit from Trainer. Designed exclusively for infinite streaming
datasets (e.g., nanochat's tokenizing data loader wrapped in IterableDataset).

Design principles:
  - Training time is governed entirely by optimizer steps, not epochs.
  - Termination:  global_step >= max_steps
  - Validation:   global_step % val_every_n_step == 0  (step-based)
  - IterableDataset handles DDP data sharding internally; no DistributedSampler
    is applied to any dataloader.
  - global_step counts optimizer steps (incremented after gradient accumulation).
  - current_epoch is always 0; the epoch concept is not used.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Union
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from claude_class.torchlightning_function import DDPStrategy, ModelCheckpoint
from claude_class.torchlightning_trainer import ModelSummary


class IterableTrainer:
    """
    Standalone step-based trainer for IterableDataset.

    Supports:
      - Single-GPU and multi-GPU (DDP via torchrun) training
      - Mixed-precision training (32-true, 16-mixed, bf16-mixed)
      - Gradient accumulation and gradient clipping
      - Step-based validation and checkpoint saving
      - Checkpoint save/load and training resumption
      - fast_dev_run and other debug modes
      - LightningModule lifecycle hooks
      - WandbLogger and ModelCheckpoint callback integration

    Constructor args are a superset of those accepted by train.py's set_trainer().
    """

    def __init__(
        self,
        accelerator: str = "auto",
        devices: Any = "auto",
        num_nodes: int = 1,
        precision: str = "32-true",
        max_steps: int = -1,
        logger: Any = None,
        enable_model_summary: bool = True,
        callbacks: Optional[List[Any]] = None,
        strategy: Any = "ddp",
        enable_checkpointing: bool = True,
        fast_dev_run: bool = False,
        num_sanity_val_steps: int = 0,
        limit_train_batches: Union[int, float] = 1.0,  # accepted but not used in training loop
        limit_val_batches: Union[int, float] = 1.0,
        limit_test_batches: Union[int, float] = 1.0,
        detect_anomaly: bool = False,
        gradient_clip_val: Optional[float] = None,
        overfit_batches: Union[int, float] = 0,
        profiler: Optional[str] = None,
        val_every_n_step: Union[int] = 15000,
        val_after_n_step: int = 0,  
        deterministic: bool = False,
        log_every_n_steps: int = 50,
        accumulate_grad_batches: int = 1,
        inference_mode: bool = True,
    ):
        # ---- Store configuration ----
        self.accelerator = accelerator
        self._devices_arg = devices
        self.num_nodes = num_nodes
        self.precision = precision
        self.max_steps = max_steps
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
        self.val_every_n_step = val_every_n_step
        self.val_after_n_step = val_after_n_step
        self.deterministic = deterministic
        self.log_every_n_steps = log_every_n_steps
        self.accumulate_grad_batches = accumulate_grad_batches
        self.inference_mode = inference_mode

        # ---- Runtime state ----
        self.global_step: int = 0
        self.current_epoch: int = 0  # always 0; epoch tracking is not used
        self.global_rank: int = 0
        self.local_rank: int = 0
        self.world_size: int = 1
        self.optimizers: List[torch.optim.Optimizer] = []
        self._schedulers: List[Dict[str, Any]] = []
        self._model = None       # unwrapped LightningModule
        self._ddp_model = None   # DDP-wrapped model (or same as _model)
        self._device = None
        self._should_stop = False

        # ---- DDP configuration from strategy ----
        self._find_unused_parameters = False
        if isinstance(self.strategy, DDPStrategy):
            self._find_unused_parameters = self.strategy.find_unused_parameters

        # ---- Precision / autocast setup ----
        self._autocast_dtype = None
        self._use_grad_scaler = False
        if precision in ("16-mixed", "16"):
            self._autocast_dtype = torch.float16
            self._use_grad_scaler = True
        elif precision in ("bf16-mixed", "bf16"):
            self._autocast_dtype = torch.bfloat16
            self._use_grad_scaler = False  # bf16 does not need loss scaling

        # ---- Determinism ----
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # ---- Anomaly detection ----
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        # ---- fast_dev_run: 1 train step + 1 val batch, then stop ----
        if self.fast_dev_run:
            self.max_steps = 1
            self.limit_val_batches = 1
            self.limit_test_batches = 1
            self.num_sanity_val_steps = 0

    # ====================================================================
    # Distributed setup
    # ====================================================================

    def _setup_distributed(self) -> None:
        """
        Detect and initialize distributed training environment.

        If launched via torchrun, RANK / LOCAL_RANK / WORLD_SIZE are already
        set in the environment. Initializes the NCCL process group and sets
        rank / device information accordingly.
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
            self.global_rank = 0
            self.local_rank = 0
            self.world_size = 1
            self._device = (
                torch.device("cuda", 0)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

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
        """Return an autocast context for mixed-precision, or a no-op context."""
        if self._autocast_dtype is not None:
            return torch.amp.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return nullcontext()

    # ====================================================================
    # Batch utilities
    # ====================================================================

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Recursively move tensors in a batch to the training device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self._device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            moved = [self._move_batch_to_device(item) for item in batch]
            return type(batch)(moved)
        return batch

    def _limit_batches(self, dataloader: DataLoader, limit: Union[int, float]) -> int:
        """
        Compute the effective number of batches for validation or test.

        If the dataset implements __len__, applies the limit as a fraction
        (float <= 1.0) or absolute count (int). If the dataset has no __len__
        (bare IterableDataset without __len__), treats float as unlimited
        (sys.maxsize) and int as an absolute count.
        """
        try:
            total = len(dataloader)
            if isinstance(limit, float) and limit <= 1.0:
                return max(1, int(total * limit))
            else:
                return min(int(limit), total)
        except TypeError:
            # Dataset does not implement __len__
            if isinstance(limit, float):
                return sys.maxsize
            return int(limit)

    # ====================================================================
    # Optimizer / scheduler configuration
    # ====================================================================

    def _configure_optimizers(self, model) -> None:
        """
        Call model.configure_optimizers() and parse the result.

        Supports the dict format used by this repository:
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
                    self._schedulers = [
                        {"scheduler": sched_cfg, "interval": "epoch", "frequency": 1}
                    ]
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
            raise ValueError(
                f"Unsupported return type from configure_optimizers: {type(opt_config)}"
            )

    def _step_schedulers(self, interval: str) -> None:
        """Step all LR schedulers that match the given interval ('step' or 'epoch')."""
        for sched_cfg in self._schedulers:
            if sched_cfg.get("interval", "epoch") == interval:
                freq = sched_cfg.get("frequency", 1)
                if interval == "step" and self.global_step % freq == 0:
                    sched_cfg["scheduler"].step()
                elif interval == "epoch":
                    sched_cfg["scheduler"].step()

    # ====================================================================
    # Checkpoint helpers
    # ====================================================================

    def _resume_from_checkpoint(self, model, ckpt_path: str) -> None:
        """
        Resume training from a saved checkpoint.

        Restores model weights, optimizer states, scheduler states, and
        global_step. current_epoch is not restored because epoch tracking
        is not used.
        """
        self._print_rank0(f"Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self._device, weights_only=False)

        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        if "optimizer_states" in checkpoint:
            for opt, opt_state in zip(self.optimizers, checkpoint["optimizer_states"]):
                opt.load_state_dict(opt_state)

        if "lr_scheduler_states" in checkpoint:
            for sched_cfg, sched_state in zip(
                self._schedulers, checkpoint["lr_scheduler_states"]
            ):
                sched_cfg["scheduler"].load_state_dict(sched_state)

        self.global_step = checkpoint.get("global_step", 0)

        if hasattr(model, "on_load_checkpoint"):
            model.on_load_checkpoint(checkpoint)

        self._print_rank0(f"Resumed at global_step={self.global_step}")

    def _try_checkpoint(self, model, val_metrics: Dict[str, Any]) -> None:
        """
        Invoke all ModelCheckpoint callbacks with the current validation metrics.

        Only runs on rank 0 to avoid duplicate checkpoint writes in DDP.
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
    # Validation loop
    # ====================================================================

    def _run_validation(self, model, val_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Run the full validation loop for IterableDataset.

        Uses iter()/next() instead of a for-loop. If the iterator is
        exhausted before num_val_batches is reached, it is re-created
        so that validation always processes exactly num_val_batches batches.

        EBT models require gradients during validation (MCMC optimization), so
        this method does NOT wrap in no_grad or inference_mode; the model manages
        its own gradient context via torch.set_grad_enabled internally.
        """
        model.eval()
        model.on_validation_epoch_start()

        num_val_batches = self._limit_batches(val_dataloader, self.limit_val_batches)

        all_metrics: Dict[str, float] = {}
        count = 0

        val_iter = iter(val_dataloader)

        for batch_idx in range(num_val_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                # Re-create the iterator if the val dataset wraps around
                val_iter = iter(val_dataloader)
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break

            batch = self._move_batch_to_device(batch)
            model.on_validation_batch_start(batch, batch_idx)

            output = model.validation_step(batch, batch_idx)

            model.on_validation_batch_end(output, batch, batch_idx)

            step_metrics = model.get_logged_metrics()
            for k, v in step_metrics.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    v = v.item()
                if isinstance(v, (int, float)):
                    all_metrics[k] = all_metrics.get(k, 0.0) + v
            count += 1
            model.clear_logged_metrics()

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
        Run the test loop for IterableDataset.

        Uses iter()/next() with a manual tqdm progress bar to avoid
        calling len() on the DataLoader. When inference_mode=False
        (required for EBT's MCMC gradient computation), no gradient
        restriction is applied.
        """
        self._setup_distributed()

        model.trainer = self
        model.to(self._device)
        self._model = model

        model.setup("test")
        test_dl = model.test_dataloader()
        # IterableDataset handles DDP sharding internally; no sampler wrapping needed

        num_test_batches = self._limit_batches(test_dl, self.limit_test_batches)

        self.optimizers = []  # no optimizer needed during testing

        model.on_test_epoch_start()
        self._print_rank0(f"Testing: {num_test_batches} batches")

        ctx = torch.inference_mode() if self.inference_mode else nullcontext()

        with ctx:
            test_iter = iter(test_dl)
            pbar = tqdm(
                total=num_test_batches,
                desc="Testing",
                disable=not self.is_global_zero,
            )

            for batch_idx in range(num_test_batches):
                try:
                    batch = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_dl)
                    try:
                        batch = next(test_iter)
                    except StopIteration:
                        break

                batch = self._move_batch_to_device(batch)
                model.on_test_batch_start(batch, batch_idx)
                output = model.test_step(batch, batch_idx)
                model.on_test_batch_end(output, batch, batch_idx)
                model.clear_logged_metrics()
                pbar.update(1)

            pbar.close()

        model.on_test_epoch_end()
        self._print_rank0("Testing complete.")

    # ====================================================================
    # Training entry point
    # ====================================================================

    def fit(self, model, datamodule=None, ckpt_path: Optional[str] = None) -> None:
        """
        Run the full step-based training loop.

        Lifecycle:
          1. Setup distributed environment.
          2. Move model to device; wrap in DDP if needed.
          3. Configure optimizers and schedulers.
          4. Optionally resume from checkpoint.
          5. Optionally run sanity validation.
          6. Main loop: fetch -> forward -> backward -> optimizer step.
             Validation is triggered every val_every_n_step optimizer steps.
             Training stops when global_step >= max_steps.
          7. Final validation and checkpoint after the loop ends.

        The training dataloader wraps an infinite IterableDataset. StopIteration
        from the iterator is treated as an emergency exit (it should never happen
        under normal usage with nanochat's streaming generator).
        """
        self._setup_distributed()

        model.trainer = self
        model.to(self._device)
        self._model = model

        # Print model summary on rank 0
        if self.enable_model_summary and self.is_global_zero:
            for cb in self.callbacks:
                if isinstance(cb, ModelSummary):
                    cb.on_fit_start(model)
                    break

        # Get dataloaders; IterableDataset handles DDP sharding internally
        model.setup("fit")
        train_dl = model.train_dataloader()
        val_dl = model.val_dataloader()

        # Configure optimizers and schedulers from the model
        self._configure_optimizers(model)

        # GradScaler for fp16 mixed precision
        scaler = None
        if self._use_grad_scaler:
            scaler = torch.amp.GradScaler("cuda")

        # Wrap model in DDP for multi-GPU training
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

        # Sanity validation before training starts
        if self.num_sanity_val_steps > 0 and self.limit_val_batches != 0:
            self._print_rank0(
                f"Running {self.num_sanity_val_steps} sanity val steps..."
            )
            saved_limit = self.limit_val_batches
            self.limit_val_batches = self.num_sanity_val_steps
            self._run_validation(model, val_dl)
            self.limit_val_batches = saved_limit

        # val_every_n_step is always interpreted as an integer optimizer-step count
        val_every_n_steps = int(self.val_every_n_step)

        # ---- Training loop ----
        model.train()
        model.on_train_start()
        model.on_train_epoch_start()  # called once at the start; no epoch concept

        self._should_stop = False
        accum_count = 0  # micro-batches accumulated since last optimizer step
        batch_idx = 0    # total micro-batches processed (used in hook signatures)

        for opt in self.optimizers:
            opt.zero_grad()

        train_iter = iter(train_dl)

        while not self._should_stop:

            # ----------------------------------------------------------
            # Fetch next micro-batch from the infinite stream
            # ----------------------------------------------------------
            try:
                batch = next(train_iter)
            except StopIteration:
                # Emergency exit: nanochat's generator should never exhaust
                self._print_rank0(
                    "Warning: training iterator exhausted unexpectedly. Stopping."
                )
                break

            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------
            batch = self._move_batch_to_device(batch)
            model.on_train_batch_start(batch, batch_idx)

            with self._autocast_ctx():
                loss = model.training_step(batch, batch_idx)
                print("IterableTrainer: ", loss)

            if loss is None:
                batch_idx += 1
                continue

            # ----------------------------------------------------------
            # Backward pass with gradient accumulation scaling
            # ----------------------------------------------------------
            scaled_loss = loss / self.accumulate_grad_batches

            model.on_before_backward(scaled_loss)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            model.on_after_backward()

            accum_count += 1

            # ----------------------------------------------------------
            # Optimizer step after accumulating enough micro-batches
            # ----------------------------------------------------------
            did_step = False
            if accum_count >= self.accumulate_grad_batches:
                if self.gradient_clip_val is not None:
                    if scaler is not None:
                        scaler.unscale_(self.optimizers[0])
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.gradient_clip_val
                    )

                model.on_before_optimizer_step(self.optimizers[0])

                if scaler is not None:
                    for opt in self.optimizers:
                        scaler.step(opt)
                    scaler.update()
                else:
                    for opt in self.optimizers:
                        opt.step()

                for opt in self.optimizers:
                    opt.zero_grad()

                self.global_step += 1 # global_step += 1 should before _step_schedulers
                self._step_schedulers("step")
                accum_count = 0
                did_step = True

            # ----------------------------------------------------------
            # Post-batch hooks
            # ----------------------------------------------------------
            model.on_train_batch_end(loss, batch, batch_idx)
            model.clear_logged_metrics()
            batch_idx += 1

            # ----------------------------------------------------------
            # Step-based validation
            # ----------------------------------------------------------
            if (
                did_step
                and self.limit_val_batches != 0
                and self.global_step >= self.val_after_n_step
                and self.global_step % val_every_n_steps == 0
            ):
                val_metrics = self._run_validation(model, val_dl)
                self._try_checkpoint(model, val_metrics)
                model.train()

            # ----------------------------------------------------------
            # Step-based termination
            # ----------------------------------------------------------
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                self._should_stop = True

        # ---- End of training ----
        model.on_train_end()

        # Final validation after training completes
        if self.limit_val_batches != 0:
            val_metrics = self._run_validation(model, val_dl)
            self._try_checkpoint(model, val_metrics)

        self._print_rank0(f"Training complete. Final step={self.global_step}")
