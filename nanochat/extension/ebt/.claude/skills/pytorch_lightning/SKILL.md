---
name: Implementing pytorch_lightning Function or Class Using PyTorch
description: Use this skill when asked to reimplement a pytorch_lightning function or class purely using PyTorch (without any Lightning dependency).
---

You will be given the name of a pytorch_lightning function or class, for example:

    pytorch_lightning.Trainer

Your task is to reimplement this function or class purely using PyTorch, without importing or depending on pytorch_lightning.

Your implementation must be functional and compatible with the usage across this repository.

## Part 1 — Study the Official Implementation

First, carefully study the official implementation of the requested function or class.

Source code:
https://github.com/Lightning-AI/pytorch-lightning/tree/master

For example:
pytorch_lightning.Trainer:
https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/trainer.py

Also review the official documentation:
https://lightning.ai/docs/pytorch/stable/

Your goal in this step is to understand:

- Core responsibilities
- Public API (constructor arguments and methods)
- Expected behaviors
- Internal logic flow
- Required components and abstractions
- Dependencies that need replacement

You do NOT need to replicate Lightning’s entire internal ecosystem — only the parts required for correct usage in this repository.


## Part 2 — Analyze Usage in this Repository

Next, search across this repository and locate all usages of the requested function or class.

Example:

```python
import pytorch_lightning as L

trainer = L.Trainer(
    accelerator="auto",
    devices=args.gpus,
    ...
)
```

You must identify:

- Which constructor arguments are actually used

- Which methods are called, for example, fit, test, validate, etc.

- What behaviors are relied upon

- What callbacks or hooks are expected

- What return values (if any) are used

Focus only on features required by the repository. Do NOT over-implement unused Lightning features.

## Part 3 — Define the Required Functional Scope

Before coding, clearly determine:

- Which arguments must be supported

- Which methods must be implemented

- Which behaviors are required

- Which features can be safely omitted

Your implementation should:

- Preserve the external interface expected by the repository

- Match input signatures used in the codebase

- Provide compatible return types

- Reproduce expected logic flow

You are not required to reproduce Lightning’s internal architecture exactly. You only need functional equivalence for this repository.

## Part 4 — Implement Using Pure PyTorch

Now implement the class or function using only PyTorch APIs and Standard Python libraries.

DO NOT import:

- pytorch_lightning

- lightning

- Any Lightning internal utilities

You may define helper classes or utilities if necessary.

Design Principles:

- Keep implementation minimal but correct

- Avoid unnecessary abstractions

- Prefer clarity over architectural complexity

- Ensure compatibility with repository usage

- Make the implementation self-contained

## Part 5 — Verify Correctness

After implementation, verify:

- Python syntax is correct

- Constructor arguments match repository usage

- Methods expected in this repository exist

- Input/output types are consistent

- Training/validation logic functions as expected

- Edge cases required by this repository are handled

Specifically confirm:

- Method signatures match usage

- No missing required arguments

- No unused Lightning-only concepts remain

- All dependencies are replaced with PyTorch equivalents

## Part 6 — Final Output Requirements

Your output must be in the requested file:

- Contain only the implemented class/function

- Include clear English comments explaining behavior

- Be self-contained and runnable

- Not reference pytorch_lightning anywhere

The implementation must be production-quality and compatible with the repository.








