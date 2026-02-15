from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from tokenizer import AutoTokenizer

@dataclass
class DataCollatorWithPadding:
    """
    Minimal pure-PyTorch version of transformers.DataCollatorWithPadding.

    This collator pads variable-length tokenized samples into a batch tensor.

    Expected each feature (sample) to be a dict like:
      {
        "input_ids": List[int],
        "attention_mask": Optional[List[int]],
        "token_type_ids": Optional[List[int]],
        "labels": Optional[Union[int, float, List[int]]],
        ...
      }

    Notes:
    - This is a minimal implementation. The real HF collator supports more dtypes,
      numpy, tensors, and tokenizer.pad() behavior.
    - We handle the most common training pipeline use-cases.
    """

    tokenizer: Any
    padding: Union[bool, str] = True  # True/"longest"/"max_length"/False
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id: int = -100  # common ignore_index for CE loss

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if self.return_tensors != "pt":
            raise ValueError("This minimal collator only supports return_tensors='pt'.")

        if not features:
            raise ValueError("features must be a non-empty list of samples.")

        # ---- Determine padding length ----
        lengths = [len(f["input_ids"]) for f in features]
        if self.padding is False:
            target_len = max(lengths)  # still need rectangular tensors
        elif self.padding is True or self.padding == "longest":
            target_len = max(lengths)
        elif self.padding == "max_length":
            if self.max_length is None:
                raise ValueError("padding='max_length' requires max_length.")
            target_len = self.max_length
        else:
            raise ValueError("padding must be False/True/'longest'/'max_length'.")

        # Optionally round target_len up to a multiple (Tensor Core friendliness)
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            m = self.pad_to_multiple_of
            target_len = ((target_len + m - 1) // m) * m

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        if padding_side not in ("left", "right"):
            padding_side = "right"

        # ---- Helper for padding 1D sequences ----
        def pad_1d(seq: List[int], pad_value: int, tgt_len: int) -> List[int]:
            if len(seq) > tgt_len:
                # Truncate if needed (collator can be used after tokenizer truncation,
                # but we keep this defensive behavior).
                seq = seq[:tgt_len]
            pad_len = tgt_len - len(seq)
            if pad_len <= 0:
                return seq
            if padding_side == "right":
                return seq + [pad_value] * pad_len
            else:
                return [pad_value] * pad_len + seq

        # ---- Figure out which keys are present ----
        # We always pad input_ids; others are conditional.
        has_attention_mask = any("attention_mask" in f for f in features)
        has_token_type_ids = any("token_type_ids" in f for f in features)
        has_labels = any("labels" in f for f in features)

        # ---- Build padded tensors ----
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_token_type_ids: List[List[int]] = []

        # For labels, we support:
        # 1) scalar labels (classification) -> stack into shape [B]
        # 2) sequence labels (token classification / LM) -> pad to [B, T]
        labels_are_sequence = False
        if has_labels:
            for f in features:
                if "labels" in f and isinstance(f["labels"], list):
                    labels_are_sequence = True
                    break

        batch_labels_scalar: List[Any] = []
        batch_labels_seq: List[List[int]] = []

        for f in features:
            ids = f["input_ids"]
            batch_input_ids.append(pad_1d(ids, pad_id, target_len))

            if has_attention_mask:
                if "attention_mask" in f and f["attention_mask"] is not None:
                    am = f["attention_mask"]
                else:
                    # If missing, infer from input length (1 for tokens, 0 for pads)
                    am = [1] * len(ids)
                batch_attention_mask.append(pad_1d(am, 0, target_len))

            if has_token_type_ids:
                if "token_type_ids" in f and f["token_type_ids"] is not None:
                    tt = f["token_type_ids"]
                else:
                    tt = [0] * len(ids)
                batch_token_type_ids.append(pad_1d(tt, 0, target_len))

            if has_labels and "labels" in f:
                lab = f["labels"]
                if labels_are_sequence:
                    # Missing labels in some samples -> pad as ignore_index
                    if lab is None:
                        lab_seq = []
                    else:
                        if not isinstance(lab, list):
                            raise ValueError("Mixed label types in batch: expected list labels for all samples.")
                        lab_seq = lab
                    batch_labels_seq.append(pad_1d(lab_seq, self.label_pad_token_id, target_len))
                else:
                    # Scalar / non-list labels
                    if isinstance(lab, list):
                        raise ValueError("Mixed label types in batch: expected scalar labels, got list.")
                    batch_labels_scalar.append(lab)

        # ---- Convert to torch tensors ----
        batch: Dict[str, torch.Tensor] = {}
        batch["input_ids"] = torch.tensor(batch_input_ids, dtype=torch.long)

        if has_attention_mask:
            batch["attention_mask"] = torch.tensor(batch_attention_mask, dtype=torch.long)

        if has_token_type_ids:
            batch["token_type_ids"] = torch.tensor(batch_token_type_ids, dtype=torch.long)

        if has_labels:
            if labels_are_sequence:
                batch["labels"] = torch.tensor(batch_labels_seq, dtype=torch.long)
            else:
                # Choose dtype based on label type (int -> long, float -> float)
                if batch_labels_scalar and isinstance(batch_labels_scalar[0], float):
                    batch["labels"] = torch.tensor(batch_labels_scalar, dtype=torch.float)
                else:
                    batch["labels"] = torch.tensor(batch_labels_scalar, dtype=torch.long)

        # ---- Pass through any extra keys that are already tensors or scalars ----
        # If you have extra metadata keys, you can decide to keep/drop them.
        # Here we drop unknown variable-length lists to avoid shape issues.
        for k in features[0].keys():
            if k in batch:
                continue
            vals = [f.get(k, None) for f in features]
            # Keep only if all are scalars or tensors of the same shape
            if all(isinstance(v, (int, float)) or v is None for v in vals):
                # None will become nan for float, or error for int; so we skip None mixed.
                if any(v is None for v in vals):
                    continue
                dtype = torch.float if isinstance(vals[0], float) else torch.long
                batch[k] = torch.tensor(vals, dtype=dtype)
            elif all(isinstance(v, torch.Tensor) for v in vals):
                # Stack tensors if same shape
                try:
                    batch[k] = torch.stack(vals, dim=0)
                except Exception:
                    pass

        return batch


class NLP_HF_Collator:
    def __init__(self, hparams):
        self.hparams = hparams
        self.max_length = hparams.context_length+1
        self.tokenizer = None  # Will be initialized in __call__
        self.data_collator = None

    def __call__(self, batch):
        padding = "max_length" if self.hparams.mcmc_replay_buffer else True # for replay buffer need to pad to max since all elements in replay buffer need same seq dim
        if self.hparams.pretokenize_dataset:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces=False)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # is token 0, was right padding things
                self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding=padding, max_length=self.max_length)
            return self.data_collator(batch)
        else:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, clean_up_tokenization_spaces = False)
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # is token 0, was right padding things
            if self.hparams.execution_mode == "inference":
                questions, answers = zip(*batch)
                return self.tokenizer(questions, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length), self.tokenizer(answers, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            
            tokens = self.tokenizer(batch, return_tensors="pt", padding=padding, truncation=True, max_length=self.max_length)
            return tokens
        

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    class DummyTokenizer:
        pad_token_id = 0
        padding_side = "right"

    collator = DataCollatorWithPadding(
        tokenizer=DummyTokenizer(),
        padding=True,              # pad to longest in batch
        pad_to_multiple_of=8,      # optional
        label_pad_token_id=-100,
    )

    features = [
        {"input_ids": [101, 2003, 102], "attention_mask": [1, 1, 1], "labels": 1},
        {"input_ids": [101, 2023, 2003, 1037, 7953, 102], "labels": 0},
    ]

    batch = collator(features)
    for k, v in batch.items():
        print(k, v.shape, v.dtype)
