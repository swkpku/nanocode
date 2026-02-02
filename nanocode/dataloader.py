"""
Distributed data loading with FIM integration for code pretraining.
Adapts nanochat's BOS-aligned streaming dataloader, adding FIM transform
before tokenization and multi-language sampling via dataset.py.
"""

import random
from collections import deque

import torch

from nanocode.common import get_dist_info
from nanocode.dataset import multi_language_iter_batched, parquets_iter_batched
from nanocode.tokenizer import get_tokenizer
from nanocode.fim import apply_fim, tokenize_fim_segments


def tokenizing_distributed_data_loader(B, T, split, fim_rate=0.5, fim_spm_rate=0.5,
                                        tokenizer_threads=4, tokenizer_batch_size=128):
    """
    Stream pretraining code from parquet files, optionally apply FIM, tokenize, yield training batches.

    Args:
        B: batch size per device
        T: sequence length
        split: "train" or "val"
        fim_rate: probability of applying FIM to each document (0.0 to disable)
        fim_spm_rate: when FIM applied, probability of SPM vs PSM format
        tokenizer_threads: threads for batch tokenization
        tokenizer_batch_size: batch size for tokenizer encoding
    """
    assert split in ["train", "val"]
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1  # +1 for target at last position

    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    fim_rng = random.Random(42 + ddp_rank)  # per-rank FIM randomness

    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    # Infinite iterator over document batches
    def document_batches():
        while True:
            for batch in multi_language_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    while True:
        # Accumulate enough tokens for one batch
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)

            for doc in doc_batch:
                # Apply FIM transformation (pre-tokenization)
                segments, is_fim = apply_fim(doc, fim_rate=fim_rate, fim_spm_rate=fim_spm_rate, rng=fim_rng)

                # Prepend BOS token
                token_buffer.append(bos_token)

                if is_fim:
                    # Tokenize each FIM segment independently
                    fim_ids = tokenize_fim_segments(segments, tokenizer)
                    token_buffer.extend(fim_ids)
                else:
                    # Standard tokenization
                    text = segments[0][1]  # (None, text) pair
                    ids = tokenizer.encode(text)
                    token_buffer.extend(ids)

        # Move tokens from deque into scratch buffer
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()

        # Create inputs/targets
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(B, T).to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device="cuda", dtype=torch.int64, non_blocking=True)
        yield inputs, targets
