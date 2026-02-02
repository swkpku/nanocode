"""
Minimal coding agent CLI.
System prompt describes 4 tools: bash, read_file, write_file, edit_file.
Conversation loop with tool execution. Max 10 tool turns per response.

Usage:
    python -m scripts.code_cli
    python -m scripts.code_cli -i base
    python -m scripts.code_cli -p "Write a fibonacci function in Python"
"""

import argparse
import torch
from nanocode.common import compute_init
from nanocode.engine import Engine
from nanocode.checkpoint_manager import load_model

SYSTEM_PROMPT = """You are a helpful coding assistant. You have access to the following tools:

1. **bash** - Execute shell commands
   Usage: {"name": "bash", "arguments": {"command": "<shell command>"}}

2. **read_file** - Read file contents
   Usage: {"name": "read_file", "arguments": {"path": "<file path>"}}

3. **write_file** - Create or overwrite a file
   Usage: {"name": "write_file", "arguments": {"path": "<file path>", "content": "<file content>"}}

4. **edit_file** - Replace text in a file
   Usage: {"name": "edit_file", "arguments": {"path": "<file path>", "old_string": "<text to find>", "new_string": "<replacement text>"}}

To use a tool, output the JSON between tool markers. You can use multiple tools in a single response. Think step by step and explain what you're doing."""

parser = argparse.ArgumentParser(description='Code CLI agent')
parser.add_argument('-i', '--source', type=str, default="sft", help="Model source: base|sft")
parser.add_argument('-g', '--model-tag', type=str, default=None)
parser.add_argument('-s', '--step', type=int, default=None)
parser.add_argument('-p', '--prompt', type=str, default='', help='Single prompt mode')
parser.add_argument('-t', '--temperature', type=float, default=0.6)
parser.add_argument('-k', '--top-k', type=int, default=50)
parser.add_argument('-m', '--max-tokens', type=int, default=2048)
args = parser.parse_args()

# Init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                     model_tag=args.model_tag, step=args.step)

# Special tokens
bos = tokenizer.get_bos_token_id()
user_start = tokenizer.encode_special("<|user_start|>")
user_end = tokenizer.encode_special("<|user_end|>")
assistant_start = tokenizer.encode_special("<|assistant_start|>")
assistant_end = tokenizer.encode_special("<|assistant_end|>")
tool_start = tokenizer.encode_special("<|tool_start|>")
tool_end = tokenizer.encode_special("<|tool_end|>")
output_start = tokenizer.encode_special("<|output_start|>")
output_end = tokenizer.encode_special("<|output_end|>")

engine = Engine(model, tokenizer)

print("\nNanoCode Interactive Agent")
print("-" * 50)
print("Type 'quit' or 'exit' to end")
print("Type 'clear' to start fresh")
print("-" * 50)

# Start conversation with system prompt baked into first user message
conversation_tokens = [bos]

# Track if system prompt has been injected
system_prompt_injected = False

while True:
    if args.prompt:
        user_input = args.prompt
    else:
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        system_prompt_injected = False
        print("Conversation cleared.")
        continue

    if not user_input:
        continue

    # Inject system prompt on first user message
    if not system_prompt_injected:
        full_input = SYSTEM_PROMPT + "\n\n" + user_input
        system_prompt_injected = True
    else:
        full_input = user_input

    # Add user message
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(full_input))
    conversation_tokens.append(user_end)

    # Start assistant response
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]
            response_tokens.append(token)

            # Decode and print, but handle special tokens nicely
            if token == tool_start:
                print("\n[Tool Call] ", end="", flush=True)
            elif token == tool_end:
                print(" [Executing...]", end="", flush=True)
            elif token == output_start:
                print("\n[Output] ", end="", flush=True)
            elif token == output_end:
                print("\n", end="", flush=True)
            elif token == assistant_end:
                pass
            else:
                token_text = tokenizer.decode([token])
                print(token_text, end="", flush=True)

    print()

    # Ensure assistant_end is present
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In prompt mode, exit after one response
    if args.prompt:
        break
