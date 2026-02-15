import sys
sys.path.append("../../../")

from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit

import torch
from torch.utils.data import IterableDataset, DataLoader


class IterableDataset(IterableDataset):
    """
    Wraps tokenizing_distributed_data_loader_with_state_bos_bestfit
    into a PyTorch IterableDataset.

    This keeps:
    - infinite streaming
    - resume state support
    - no padding
    - distributed compatibility
    """

    def __init__(
        self,
        tokenizer,
        B,
        T,
        split,
        max_iter,
        # tokenizer_threads=4,
        # tokenizer_batch_size=128,
        device="cuda",
        resume_state_dict=None,
        # buffer_size=1000,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.max_iter = max_iter
        # self.tokenizer_threads = tokenizer_threads
        # self.tokenizer_batch_size = tokenizer_batch_size
        self.device = device
        self.resume_state_dict = resume_state_dict
        # self.buffer_size = buffer_size

    def __iter__(self):
        return tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            # tokenizer_threads=self.tokenizer_threads,
            # tokenizer_batch_size=self.tokenizer_batch_size,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
            # buffer_size=1000,
        )
    
    def __len__(self):
        return self.max_iter


def generate_dataloader(tokenizer, batch_size, max_seq_length, max_iter, split, device, resume_state_dict=None):

    dataset = IterableDataset(
        tokenizer=tokenizer,
        B=batch_size, 
        T=max_seq_length,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,      # IMPORTANT
        shuffle=False,
        num_workers=0,        # keep 0 for stateful streaming
        pin_memory=False      # already handled internally
    )
    return dataloader