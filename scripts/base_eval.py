"""
Base model evaluation: val perplexity + standard/FIM samples.

Usage:
    python -m scripts.base_eval
    torchrun --nproc_per_node=8 -m scripts.base_eval
"""

import os
import torch

from nanocode.common import compute_init, compute_cleanup, print0, get_base_dir
from nanocode.tokenizer import get_tokenizer, get_token_bytes
from nanocode.checkpoint_manager import load_model
from nanocode.dataloader import tokenizing_distributed_data_loader
from nanocode.loss_eval import evaluate_bpb
from nanocode.engine import Engine


def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Load model
    model, tokenizer, meta = load_model("base", device, phase="eval")
    token_bytes = get_token_bytes(device=device)

    # Evaluate validation bpb
    print0("Evaluating validation bpb...")
    device_batch_size = 32
    max_seq_len = model.config.sequence_len
    val_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", fim_rate=0.0)
    eval_tokens = 20 * 524288
    eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)

    with autocast_ctx:
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Validation bpb: {val_bpb:.4f}")

    # Standard samples
    if ddp_rank == 0:
        engine = Engine(model, tokenizer)
        print0("\n--- Standard completion samples ---")
        prompts = [
            "def quicksort(arr):",
            "class BinaryTree:",
            "# HTTP server in Python\nimport socket\n",
            "fn main() {\n    let",
        ]
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=128, temperature=0.8, top_k=50)
            print0(tokenizer.decode(sample[0]))
            print0("-" * 40)

        # FIM samples
        print0("\n--- FIM completion samples ---")
        fim_prefix = tokenizer.encode_special("<|fim_prefix|>")
        fim_suffix = tokenizer.encode_special("<|fim_suffix|>")
        fim_middle = tokenizer.encode_special("<|fim_middle|>")
        bos = tokenizer.get_bos_token_id()

        fim_examples = [
            ("def greet(name):\n    ", '\n    print(message)\n\ngreet("World")'),
            ("for i in range(10):\n    ", "\n    results.append(x)"),
        ]
        for prefix_text, suffix_text in fim_examples:
            fim_tokens = [bos, fim_prefix] + tokenizer.encode(prefix_text) + \
                         [fim_suffix] + tokenizer.encode(suffix_text) + [fim_middle]
            with autocast_ctx:
                sample, _ = engine.generate_batch(fim_tokens, num_samples=1, max_tokens=64, temperature=0.8, top_k=50)
            middle_text = tokenizer.decode(sample[0][len(fim_tokens):])
            print0(f"Prefix: {prefix_text!r}")
            print0(f"Suffix: {suffix_text!r}")
            print0(f"Middle: {middle_text!r}")
            print0("-" * 40)

    compute_cleanup()


if __name__ == "__main__":
    main()
