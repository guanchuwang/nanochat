---
name: Verify the correctness of a given class, function, or Python file
description: Use this skill to verify the correctness, consistency, and repository compatibility of a given class, function, or Python file.
---


You will be given a class, function, or Python file path, for example:

    ./claude_class/torchlightning_trainer.py

Your task is to verify its correctness within the context of this repository. You should ignore any issues that are unrelated to the given class, function, or file.

---

# Verification Procedure

Follow the steps below in order. 

If an issue is detected at any step:

- Immediately stop the verification process after completing that step.

- Report the issue clearly.

- Do NOT continue to later steps.

- Do NOT search for additional issues.

---

## Step 1: Verify Python Syntax

Ensure the file contains valid Python syntax.

You may use:
```bash
python -m py_compile <file>
```
If syntax errors exist:

- Clearly identify the exact line(s)

- Explain the issue

- Propose a minimal fix

## Step 2: Locate Repository Usage

Search the entire repository and identify:

- All import locations

- All instantiations of the class

- All function calls

- All attribute or method accesses

List the relevant files and describe how the target component is used in each location.

## Step 3: Verify Interface Consistency

Across all usage locations, verify:

1. Constructor Compatibility

- Constructor arguments match how the class is instantiated

- No missing required parameters

- No unexpected parameters used

2. Method Availability

- All methods referenced in the repository exist

- Method names match exactly

- No deprecated or removed methods are being used

3. Signature Consistency

- Method signatures match how they are called

- Positional vs keyword arguments are compatible

- Default values behave as expected

## Step 4: Verify Behavioral Correctness

Evaluate whether the implementation logically satisfies repository expectations:

- Training and validation logic works as required

- Expected return types are correct

- Input/output shapes or types are consistent

- Error handling aligns with repository assumptions

- Required edge cases are handled

## Step 5: Dependency and Refactoring Check

Verify:

- All required dependencies are properly imported

- No unresolved symbols remain

- No leftover framework-specific concepts (e.g., Lightning-only abstractions) remain if the repository is pure PyTorch

- Only dependencies relevant to this class/function/file are considered (ignore unrelated modules)

---

## Expected Output Format

Your verification report should include:

- Syntax Status

- Usage Summary

- Interface Issues (if any)

- Behavioral Issues (if any)

- Dependency Issues (if any)

- Suggested Fixes

Be precise and minimal. Do not rewrite the full file unless necessary. Focus only on correctness and repository compatibility.










