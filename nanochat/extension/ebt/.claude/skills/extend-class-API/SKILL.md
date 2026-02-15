---
name: Extend a Class to Add a Requested Function Using PyTorch
description: Use this skill when asked to extend an existing class to implement a requested function using pure PyTorch (no pytorch_lightning dependency).
---

## Inputs You Will Receive

1) **Target class path**
   Example:
   - `torchlightning_function.WandbLogger`

2) **Requested function(s) to add**
   Example:
   - Add `.watch(...)`

3) **How the class is used (call-site snippet)**
   Example:
   ```python
   wandb_logger = WandbLogger(...)
   wandb_logger.watch(..., log="all", log_freq=...)

## Your Goal

Extend the given class so that it supports the requested function(s) with the same user-facing behavior as the official PyTorch Lightning version where practical, but implemented using:

- PyTorch + standard Python libraries only

- No imports from pytorch_lightning (or lightning)

- Keep changes minimal and localized to the target class/module

## Constraints

- Do not change existing public behavior unless required to support the new function.

- If the official implementation relies on Lightning-specific utilities, recreate small substitutes (helpers) locally in the same file/module.


## Step 1 — Study the Official Implementation

First, carefully study the official implementation of the requested function.

For example, the official implementation of torchlightning_function.WandbLogger has been given in:
https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/loggers/wandb.py

Your goal in this step is to understand:

- Core responsibilities
- Public API (constructor arguments and methods)
- Expected behaviors
- Internal logic flow
- Required components and abstractions
- Dependencies that need replacement

## Step 2 — Implement the Extension

Based on your understanding of the official implementation, extend the class to implement the requested function using PyTorch. 

You may need to:

- Keep the implementation readable with English comments

- Preserve naming and signatures as much as possible

- Add any minimal helper functions/classes needed

- Avoid unnecessary refactors

Deliverable for this step:

- The updated class code (and any helper code you added)

## Step 3 — Test the Extended function

After extending the class, test it thoroughly to ensure it meets the expected behavior and requirements. You may need to:

- Create instances of the class with different configurations
