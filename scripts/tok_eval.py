"""
Evaluate compression ratio of the tokenizer across different code languages.
"""

from nanocode.tokenizer import get_tokenizer, TiktokenTokenizer

# Test texts for different languages
python_text = r'''import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x
'''

javascript_text = r'''async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return {
            name: data.name,
            email: data.email,
            createdAt: new Date(data.created_at),
        };
    } catch (error) {
        console.error('Failed to fetch user:', error);
        return null;
    }
}
'''

go_text = r'''package main

import (
    "fmt"
    "sync"
)

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

func main() {
    var wg sync.WaitGroup
    results := make(chan int, 10)

    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func(n int) {
            defer wg.Done()
            results <- fibonacci(n)
        }(i)
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    for r := range results {
        fmt.Println(r)
    }
}
'''

rust_text = r'''use std::collections::HashMap;

fn word_frequency(text: &str) -> HashMap<&str, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        *freq.entry(word).or_insert(0) += 1;
    }
    freq
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";
    let freq = word_frequency(text);
    let mut sorted: Vec<_> = freq.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (word, count) in sorted {
        println!("{}: {}", word, count);
    }
}
'''

all_text = [
    ("python", python_text),
    ("javascript", javascript_text),
    ("go", go_text),
    ("rust", rust_text),
]

# Evaluate our tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print(f"\nVocab size: {vocab_size}")
print(f"Special tokens: {len(tokenizer.get_special_tokens())}")

# Compare with GPT-4 tokenizer
tokenizer_results = {}
vocab_sizes = {}

for tokenizer_name in ["gpt4", "ours"]:
    if tokenizer_name == "gpt4":
        tok = TiktokenTokenizer.from_pretrained("cl100k_base")
    else:
        tok = tokenizer

    vocab_sizes[tokenizer_name] = tok.get_vocab_size()
    tokenizer_results[tokenizer_name] = {}

    for name, text in all_text:
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text, f"Round-trip failed for {tokenizer_name}/{name}"
        encoded_bytes = text.encode('utf-8')
        ratio = len(encoded_bytes) / len(encoded)
        tokenizer_results[tokenizer_name][name] = {
            'bytes': len(encoded_bytes),
            'tokens': len(encoded),
            'ratio': ratio,
        }

# Print comparison
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

print(f"\nComparison with GPT-4 tokenizer:")
print("=" * 85)
print(f"{'Language':<12} {'Bytes':<8} {'GPT-4':<15} {'Ours':<15} {'Relative':<12}")
print(f"{'':12} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
print("-" * 85)

for name, text in all_text:
    gpt4_data = tokenizer_results['gpt4'][name]
    ours_data = tokenizer_results['ours'][name]
    relative_diff = ((gpt4_data['tokens'] - ours_data['tokens']) / gpt4_data['tokens']) * 100
    if ours_data['ratio'] >= gpt4_data['ratio']:
        ours_color, diff_color = GREEN, GREEN
    else:
        ours_color, diff_color = RED, RED
    print(f"{name:<12} {gpt4_data['bytes']:<8} "
          f"{gpt4_data['tokens']:<7} {gpt4_data['ratio']:<7.2f} "
          f"{ours_color}{ours_data['tokens']:<7}{RESET} "
          f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
          f"{diff_color}{relative_diff:+7.1f}%{RESET}")
