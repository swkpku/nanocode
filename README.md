# nanocode

The simplest experimental harness for training a coding LLM. Modeled after [nanochat](https://github.com/karpathy/nanochat) but focused on code. Runs on a single 8xH100 GPU node.

## Quick Start

```bash
bash runs/speedrun.sh
```

This runs the full pipeline: data download -> tokenizer -> pretrain -> eval -> SFT -> eval -> agent demo.

## What's Inside

| Stage | Script | Description |
|-------|--------|-------------|
| Data | `nanocode/dataset.py` | Download The Stack v2 (Python/JS/TS/Java/C/C++/Go/Rust) |
| Tokenizer | `scripts/tok_train.py` | Train 32K vocab BPE on code with FIM special tokens |
| Pretrain | `scripts/base_train.py` | Next-token + 50% FIM, 124M params, Muon + AdamW |
| Eval | `scripts/code_eval.py` | HumanEval + MBPP pass@k |
| SFT | `scripts/code_sft.py` | CodeAlpaca + Code-Feedback + GSM8K + synthetic tool-use |
| Agent | `scripts/code_cli.py` | CLI with bash, read_file, write_file, edit_file tools |

## Architecture

124M parameter Transformer (12 layers, 768 dim, 6 heads) with:
- Rotary embeddings (RoPE)
- QK normalization
- ReLU^2 MLP activation
- Logit softcapping at 15
- No bias, no learnable norm params

## Special Tokens

```
 0  <|bos|>               Document delimiter
 1  <|user_start|>        SFT + Agent
 2  <|user_end|>          SFT + Agent
 3  <|assistant_start|>   SFT + Agent
 4  <|assistant_end|>     SFT + Agent (stop token)
 5  <|tool_start|>        Tool call begin
 6  <|tool_end|>          Triggers tool execution
 7  <|output_start|>      Tool result begin
 8  <|output_end|>        Tool result end
 9  <|fim_prefix|>        FIM pretraining
10  <|fim_middle|>        FIM pretraining
11  <|fim_suffix|>        FIM pretraining
```

## FIM (Fill-in-the-Middle)

During pretraining, 50% of documents are transformed into FIM format:
- **PSM**: `<|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle`
- **SPM**: `<|fim_prefix|> <|fim_suffix|> suffix <|fim_middle|> prefix+middle`

Split at random character positions, 50/50 PSM/SPM. Applied pre-tokenization.

## Agent Tool Format

```
<|assistant_start|>
Let me check the files.
<|tool_start|>{"name": "bash", "arguments": {"command": "ls -la"}}<|tool_end|>
<|output_start|>total 8
-rw-r--r-- 1 user staff 42 main.py<|output_end|>
I can see main.py.<|assistant_end|>
```

## Individual Commands

```bash
# Download data
python -m nanocode.dataset -l python,javascript -n 10

# Train tokenizer
python -m scripts.tok_train --vocab_size=32768

# Pretrain (single GPU)
python -m scripts.base_train --depth=12 --fim_rate=0.5

# Pretrain (8 GPU)
torchrun --nproc_per_node=8 -m scripts.base_train -- --depth=12

# Evaluate
torchrun --nproc_per_node=8 -m scripts.code_eval -- -i base -a humaneval

# SFT
torchrun --nproc_per_node=8 -m scripts.code_sft

# Chat
python -m scripts.code_cli -p "Write a sorting algorithm"
python -m scripts.code_cli  # interactive mode
```
