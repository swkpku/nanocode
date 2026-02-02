"""
Train a BPE tokenizer on code from all configured languages.
Produces a 32768-vocab tokenizer with 12 special tokens (including FIM).

Usage:
    python -m scripts.tok_train --max_chars=2000000000 --vocab_size=32768
"""
import os
import time
import argparse
import torch

from nanocode.tokenizer import HuggingFaceTokenizer, TiktokenTokenizer, SPECIAL_TOKENS
from nanocode.common import get_base_dir
from nanocode.dataset import multi_language_iter_batched

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a BPE tokenizer on code')
parser.add_argument('--max_chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc_cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab_size', type=int, default=32768, help='Vocabulary size (default: 32768)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# Text iterator over code from all languages
def text_iterator():
    nchars = 0
    for batch in multi_language_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# Train the tokenizer
print("Training tokenizer...")
t0 = time.time()
hf_tokenizer = HuggingFaceTokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# Save the HuggingFace tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
hf_tokenizer.save(tokenizer_dir)

# Convert to tiktoken format for efficient inference
print("Converting to tiktoken format...")
tiktoken_tokenizer = TiktokenTokenizer.from_hf_tokenizer(hf_tokenizer)
tiktoken_tokenizer.save(tokenizer_dir)

# Quick sanity check with code
test_text = '''def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test it
print(fibonacci(10))  # 55
'''
encoded = tiktoken_tokenizer.encode(test_text)
decoded = tiktoken_tokenizer.decode(encoded)
assert decoded == test_text, f"Round-trip failed!\nExpected: {test_text!r}\nGot: {decoded!r}"
print(f"Sanity check passed: {len(test_text)} chars -> {len(encoded)} tokens (ratio: {len(test_text)/len(encoded):.2f})")

# Verify FIM special tokens exist
for token_name in ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]:
    token_id = tiktoken_tokenizer.encode_special(token_name)
    print(f"  {token_name} -> id {token_id}")

# Cache token_bytes mapping for bits-per-byte evaluation
vocab_size = tiktoken_tokenizer.get_vocab_size()
special_set = set(tiktoken_tokenizer.get_special_tokens())
token_bytes = []
for token_id in range(vocab_size):
    token_str = tiktoken_tokenizer.decode([token_id])
    if token_str in special_set:
        token_bytes.append(0)
    else:
        id_bytes = len(token_str.encode("utf-8"))
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")
print(f"Vocab size: {vocab_size}")
print(f"Special tokens: {len(special_set)}")
