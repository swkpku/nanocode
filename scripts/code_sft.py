"""
SFT on code instructions. Finetune a pretrained base model for code tasks.

Loads CodeAlpaca + Code-Feedback + GSM8K + synthetic tool-use examples.
Conversation format with special tokens. Supervision mask: only assistant tokens.

Usage:
    python -m scripts.code_sft
    torchrun --standalone --nproc_per_node=8 -m scripts.code_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import copy

import wandb
import torch
import torch.distributed as dist

from nanocode.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanocode.checkpoint_manager import load_model, save_checkpoint
from nanocode.engine import Engine
from tasks.common import TaskMixture, Task

# -----------------------------------------------------------------------------
# SFT Dataset classes

class CodeAlpaca(Task):
    """CodeAlpaca-20k instruction dataset."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        self.ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        prompt = row['instruction']
        if row.get('input'):
            prompt += f"\n\nInput: {row['input']}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row['output']},
        ]
        return {"messages": messages}


class CodeFeedback(Task):
    """Code-Feedback instruction dataset (multi-turn)."""
    def __init__(self, max_rows=66000, **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        ds = load_dataset("m-a-p/Code-Feedback", split="train")
        self._data = []
        for row in ds:
            messages = []
            for msg in row['messages']:
                role = msg['role']
                if role == 'system':
                    continue
                content = msg['content']
                if role in ['user', 'assistant']:
                    messages.append({"role": role, "content": content})
            if len(messages) >= 2 and messages[0]['role'] == 'user':
                self._data.append({"messages": messages})
            if len(self._data) >= max_rows:
                break

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self._data)

    def get_example(self, index):
        return self._data[index]


class GSM8KSft(Task):
    """GSM8K math dataset formatted for SFT."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from datasets import load_dataset
        self.ds = load_dataset("openai/gsm8k", "main", split="train")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": row['answer']},
        ]
        return {"messages": messages}


class SyntheticToolUse(Task):
    """
    Synthetic tool-use training examples.
    Teaches the model to use bash, read_file, write_file, edit_file tools.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._examples = self._generate_examples()

    def _generate_examples(self):
        examples = []

        # bash tool examples
        bash_examples = [
            ("List the files in the current directory.",
             [{"type": "text", "text": "Let me list the files."},
              {"type": "tool_call", "text": '{"name": "bash", "arguments": {"command": "ls -la"}}'},
              {"type": "tool_output", "text": "total 8\ndrwxr-xr-x 3 user staff 96 Jan 1 12:00 .\n-rw-r--r-- 1 user staff 42 Jan 1 12:00 main.py"},
              {"type": "text", "text": "The directory contains a `main.py` file."}]),
            ("What Python version is installed?",
             [{"type": "text", "text": "Let me check the Python version."},
              {"type": "tool_call", "text": '{"name": "bash", "arguments": {"command": "python3 --version"}}'},
              {"type": "tool_output", "text": "Python 3.11.5"},
              {"type": "text", "text": "Python 3.11.5 is installed."}]),
            ("Count the number of lines in main.py",
             [{"type": "text", "text": "I'll count the lines."},
              {"type": "tool_call", "text": '{"name": "bash", "arguments": {"command": "wc -l main.py"}}'},
              {"type": "tool_output", "text": "      42 main.py"},
              {"type": "text", "text": "main.py has 42 lines."}]),
        ]

        # read_file examples
        read_examples = [
            ("Show me the contents of config.json",
             [{"type": "text", "text": "Let me read the file."},
              {"type": "tool_call", "text": '{"name": "read_file", "arguments": {"path": "config.json"}}'},
              {"type": "tool_output", "text": '{"debug": true, "port": 8080}'},
              {"type": "text", "text": 'The config has debug mode enabled on port 8080.'}]),
        ]

        # write_file examples
        write_examples = [
            ("Create a hello world Python script.",
             [{"type": "text", "text": "I'll create the script."},
              {"type": "tool_call", "text": '{"name": "write_file", "arguments": {"path": "hello.py", "content": "def main():\\n    print(\\"Hello, World!\\")\\n\\nif __name__ == \\"__main__\\":\\n    main()\\n"}}'},
              {"type": "tool_output", "text": "Successfully wrote 73 chars to hello.py"},
              {"type": "text", "text": "I've created `hello.py` with a main function that prints Hello, World!"}]),
        ]

        # edit_file examples
        edit_examples = [
            ("Change the port from 8080 to 3000 in config.json",
             [{"type": "text", "text": "I'll update the port."},
              {"type": "tool_call", "text": '{"name": "read_file", "arguments": {"path": "config.json"}}'},
              {"type": "tool_output", "text": '{"debug": true, "port": 8080}'},
              {"type": "text", "text": "Now I'll change the port."},
              {"type": "tool_call", "text": '{"name": "edit_file", "arguments": {"path": "config.json", "old_string": "\\"port\\": 8080", "new_string": "\\"port\\": 3000"}}'},
              {"type": "tool_output", "text": "Successfully edited config.json"},
              {"type": "text", "text": "I've updated the port from 8080 to 3000."}]),
        ]

        all_tool_examples = bash_examples + read_examples + write_examples + edit_examples

        for user_msg, assistant_parts in all_tool_examples:
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_parts},
            ]
            # Duplicate each example ~200 times to reach ~2K total
            for _ in range(200):
                examples.append({"messages": messages})

        return examples

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self._examples)

    def get_example(self, index):
        return self._examples[index]


# -----------------------------------------------------------------------------
# Hyperparameters
run = "dummy"
source = "base"
model_tag = None
step = None
dtype = "bfloat16"
device_batch_size = 4
num_epochs = 1
max_iterations = -1
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
eval_every = 100
eval_steps = 100
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanocode', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanocode-sft", name=run, config=user_config)

# Load model
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model
engine = Engine(model, tokenizer)

# Task mixture
print0("Loading SFT datasets...")
train_ds = TaskMixture([
    CodeAlpaca(),          # ~20K rows
    CodeFeedback(),        # ~66K rows
    GSM8KSft(),            # ~7.5K rows
    SyntheticToolUse(),    # ~2K rows
])
print0(f"Total training examples: {len(train_ds)}")

# DataLoader
def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        return inputs.to(device), targets.to(device)

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
assert target_examples_per_step % examples_per_step == 0
grad_accum_steps = target_examples_per_step // examples_per_step
num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    num_iterations = max_iterations
print0(f"Grad accum steps: {grad_accum_steps}, iterations: {num_iterations}")

train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)

# Optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr, embedding_lr=embedding_lr,
    matrix_lr=matrix_lr, weight_decay=weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# LR schedule (linear decay)
def get_lr_multiplier(it):
    return 1.0 - it / num_iterations

# Training loop
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # Val loss
    if last_step or step % eval_every == 0:
        model.eval()
        # Quick val on a few batches of training data (no separate val set for code SFT)
        val_losses = []
        val_iter = iter(sft_data_generator(train_ds, batch_size=device_batch_size))
        for _ in range(min(eval_steps, 20)):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            val_losses.append(loss)
        val_loss = torch.stack(val_losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Val loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "val_loss": val_loss})
        model.train()

    if last_step:
        break

    # Training step
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    print0(f"Step {step:05d}/{num_iterations:05d} | loss: {train_loss.item():.6f} | lrm: {lrm:.6f} | tokens: {num_tokens.item():,}")
    wandb_run.log({"step": step, "train_loss": train_loss.item(), "lrm": lrm})

# Save checkpoint
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag_out = f"d{depth}"
    checkpoint_dir = os.path.join(base_dir, "codesft_checkpoints", model_tag_out)
    model_config_kwargs = model.config.__dict__
    save_checkpoint(
        checkpoint_dir, step, model.state_dict(), None,
        {"step": step, "val_loss": val_loss, "model_config": model_config_kwargs}
    )
    print0(f"Saved SFT checkpoint to {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
