"""
IterableTrainer: Trainer subclass for PyTorch IterableDataset.

Inherits from Trainer and adapts the training loop for infinite streaming
datasets (e.g., nanochat's tokenizing data loader wrapped in IterableDataset).

Key differences from Trainer:
  - No DistributedSampler wrapping (IterableDataset handles distribution internally)
  - Step-based while-loop instead of epoch-based for-loop
  - val_check_interval is always interpreted as a step count (integer)
  - Epochs are tracked by counting DataLoader iterator exhaustions or
    when limit_train_batches is reached per epoch

All hooks, gradient accumulation, mixed precision, checkpointing,
and validation logic are preserved identically to Trainer.
"""

import sys
from typing import Any, Dict, List, Optional, Union
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from claude_class.torchlightning_trainer import Trainer, ModelSummary
from claude_class.torchlightning_function import ModelCheckpoint


class IterableTrainer(Trainer):
    """
    Trainer subclass that supports PyTorch IterableDataset.

    The only behavioral difference from Trainer is how the training loop
    iterates over data:
      - Trainer uses epoch-based for-loops with DistributedSampler.
      - IterableTrainer uses a step-based while-loop, re-creating the
        DataLoader iterator when it is exhausted (StopIteration).

    All constructor arguments are identical to Trainer.
    """

    # ================================================================
    # Override: DDP dataloader wrapping
    # ================================================================

    def _wrap_dataloader_for_ddp(self, dataloader: DataLoader) -> DataLoader:
        """
        Skip DistributedSampler wrapping for IterableDataset.

        IterableDataset (e.g., nanochat's streaming data loader) handles
        distributed data sharding internally. DistributedSampler is
        incompatible with IterableDataset.

        For map-style datasets, delegates to the parent implementation.
        """
        if isinstance(dataloader.dataset, IterableDataset):
            return dataloader
        return super()._wrap_dataloader_for_ddp(dataloader)

    # ================================================================
    # Override: batch limiting
    # ================================================================

    def _limit_batches(self, dataloader: DataLoader, limit: Union[int, float]) -> int:
        """
        Compute effective batch count, handling datasets without __len__.

        If the dataset defines __len__, delegates to the parent implementation.
        Otherwise, treats the limit as an absolute batch count; a float >= 1.0
        means unlimited (sys.maxsize).
        """
        try:
            return super()._limit_batches(dataloader, limit)
        except TypeError:
            # Dataset does not implement __len__
            if isinstance(limit, float):
                return sys.maxsize
            return int(limit)

    # ================================================================
    # Override: training loop
    # ================================================================

    def fit(self, model, datamodule=None, ckpt_path: Optional[str] = None) -> None:
        """
        Run the full training loop for IterableDataset.

        Functionally identical to Trainer.fit() with these adaptations:
          - Uses a step-based while-loop instead of nested epoch/batch for-loops.
          - When the DataLoader iterator is exhausted (StopIteration) or
            limit_train_batches is reached, epoch hooks fire and the iterator
            is re-created.
          - val_check_interval is interpreted as a global step count.
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

        # Wrap dataloaders for DDP (no-op for IterableDataset)
        train_dl = self._wrap_dataloader_for_ddp(train_dl)
        if self.world_size > 1:
            val_dl = self._wrap_dataloader_for_ddp(val_dl)

        # Handle overfit_batches: restrict training to a small subset
        if self.overfit_batches > 0:
            if isinstance(self.overfit_batches, float) and self.overfit_batches < 1.0:
                if hasattr(train_dl.dataset, '__len__'):
                    n = max(1, int(len(train_dl.dataset) * self.overfit_batches))
                else:
                    n = int(self.overfit_batches)
            else:
                n = int(self.overfit_batches)
            self.limit_train_batches = n

        # Configure optimizers and schedulers from the model
        self._configure_optimizers(model)

        # Initialize GradScaler for mixed precision if needed
        scaler = None
        if self._use_grad_scaler:
            scaler = torch.amp.GradScaler("cuda")

        # Wrap model in DDP if distributed
        if self.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
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

        # ----- Determine validation interval (always step-based) -----
        val_every_n_steps = int(self.val_check_interval)

        # ----- Compute per-epoch batch limit -----
        num_train_batches = self._limit_batches(train_dl, self.limit_train_batches)

        # ----- Training loop -----
        model.train()
        model.on_train_start()

        self._should_stop = False
        accum_count = 0
        batch_idx = 0

        # Zero gradients at the start
        for opt in self.optimizers:
            opt.zero_grad()

        # Create initial DataLoader iterator
        train_iter = iter(train_dl)
        model.on_train_epoch_start()

        while not self._should_stop:

            # ----------------------------------------------------------
            # Check if per-epoch batch limit is reached
            # ----------------------------------------------------------
            epoch_exhausted = (batch_idx >= num_train_batches)

            # ----------------------------------------------------------
            # Fetch next batch, or handle epoch boundary
            # ----------------------------------------------------------
            if epoch_exhausted:
                batch = None
            else:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    batch = None

            if batch is None:
                # --- Epoch boundary ---
                model.on_train_epoch_end()
                self._step_schedulers("epoch")

                # End-of-epoch validation
                if (
                    self.limit_val_batches != 0
                    and self.current_epoch % self.check_val_every_n_epoch == 0
                ):
                    val_metrics = self._run_validation(model, val_dl)
                    self._try_checkpoint(model, val_metrics)
                    model.train()

                # Advance epoch
                self.current_epoch += 1
                batch_idx = 0

                # Check max_epochs termination
                if self.current_epoch >= self.max_epochs:
                    self._should_stop = True
                    break

                # Re-create iterator for the new epoch
                train_iter = iter(train_dl)
                model.on_train_epoch_start()
                continue

            # ----------------------------------------------------------
            # Training step (identical logic to Trainer)
            # ----------------------------------------------------------
            batch = self._move_batch_to_device(batch)
            model.on_train_batch_start(batch, batch_idx)

            # Forward pass with optional autocast
            with self._autocast_ctx():
                loss = model.training_step(batch, batch_idx)
                print(loss)

            if loss is None:
                batch_idx += 1
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

            # ----------------------------------------------------------
            # Optimizer step at the end of each accumulation window
            # ----------------------------------------------------------
            did_step = False
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
                did_step = True

            # ----------------------------------------------------------
            # on_train_batch_end hook
            # ----------------------------------------------------------
            model.on_train_batch_end(loss, batch, batch_idx)
            model.clear_logged_metrics()

            batch_idx += 1

            # ----------------------------------------------------------
            # Periodic validation (step-based, only right after a step)
            # ----------------------------------------------------------
            if (
                did_step
                and self.limit_val_batches != 0
                and self.global_step % val_every_n_steps == 0
                and self.current_epoch % self.check_val_every_n_epoch == 0
            ):
                val_metrics = self._run_validation(model, val_dl)
                self._try_checkpoint(model, val_metrics)
                model.train()

            # ----------------------------------------------------------
            # Check max_steps termination
            # ----------------------------------------------------------
            if self.max_steps > 0 and self.global_step >= self.max_steps:
                self._should_stop = True

        # ----- End of training -----
        model.on_train_end()

        # Final validation after training completes
        if self.limit_val_batches != 0:
            val_metrics = self._run_validation(model, val_dl)
            self._try_checkpoint(model, val_metrics)

        self._print_rank0(
            f"Training complete. Final step={self.global_step}, epoch={self.current_epoch}"
        )

    # ================================================================
    # Override: validation loop for IterableDataset
    # ================================================================

    def _run_validation(self, model, val_dataloader: DataLoader) -> Dict[str, Any]:
        """
        Run the full validation loop, adapted for IterableDataset.

        Uses iter()/next() instead of for-loop enumeration. When the
        iterator is exhausted before reaching num_val_batches, it is
        re-created to continue accumulating batches.

        All hooks and metric collection are preserved identically to
        Trainer._run_validation().
        """
        model.eval()
        model.on_validation_epoch_start()

        num_val_batches = self._limit_batches(val_dataloader, self.limit_val_batches)

        all_metrics = {}
        count = 0

        val_iter = iter(val_dataloader)

        for batch_idx in range(num_val_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_dataloader)
                try:
                    batch = next(val_iter)
                except StopIteration:
                    break

            batch = self._move_batch_to_device(batch)
            model.on_validation_batch_start(batch, batch_idx)

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

    # ================================================================
    # Override: test loop for IterableDataset
    # ================================================================

    def test(self, model, datamodule=None) -> None:
        """
        Run the test loop, adapted for IterableDataset.

        Uses iter()/next() instead of for-loop enumeration with tqdm.
        A manual tqdm progress bar avoids calling len() on the DataLoader.

        inference_mode context is preserved: when inference_mode=False
        (needed for EBT's MCMC gradient computation), no grad restriction
        is applied.
        """
        self._setup_distributed()

        model.trainer = self
        model.to(self._device)
        self._model = model

        model.setup("test")
        test_dl = model.test_dataloader()
        if self.world_size > 1:
            test_dl = self._wrap_dataloader_for_ddp(test_dl)

        num_test_batches = self._limit_batches(test_dl, self.limit_test_batches)

        self.optimizers = []

        model.on_test_epoch_start()

        self._print_rank0(f"Testing: {num_test_batches} batches")

        if self.inference_mode:
            ctx = torch.inference_mode()
        else:
            ctx = nullcontext()

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
