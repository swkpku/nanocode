"""
Pretrain model with FIM objective. Run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch

from nanocode.gpt import GPT, GPTConfig
from nanocode.dataloader import tokenizing_distributed_data_loader
from nanocode.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, get_gpu_flops
from nanocode.tokenizer import get_tokenizer, get_token_bytes
from nanocode.checkpoint_manager import save_checkpoint
from nanocode.loss_eval import evaluate_bpb
from nanocode.engine import Engine
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name ("dummy" = skip wandb)
# Model architecture
depth = 12 # transformer depth, rest derived
max_seq_len = 2048 # context length
# Training horizon
num_iterations = -1 # explicit steps (-1 = disable)
target_flops = -1.0 # target FLOPs (-1 = disable)
target_param_data_ratio = 20 # Chinchilla ratio (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size
total_batch_size = 524288 # total batch size in tokens
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0
# FIM
fim_rate = 0.5 # probability of FIM per document
fim_spm_rate = 0.5 # probability of SPM vs PSM when FIM applied
# Evaluation
eval_every = 250
eval_tokens = 20 * 524288
sample_every = 2000
# Output
model_tag = ""
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanocode', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanocode", name=run, config=user_config)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model architecture derived from depth
num_layers = depth
model_dim = depth * 64 # aspect ratio 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads
print0(f"num_layers: {num_layers}, model_dim: {model_dim}, num_heads: {num_heads}")

# Gradient accumulation
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Initialize Model
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers,
                           n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device="cuda")
model.init_weights()
orig_model = model
model = torch.compile(model, dynamic=False)

# Warmup: trigger compilation on all ranks simultaneously before any collective ops
print0("Compiling model (this may take several minutes)...")
with autocast_ctx:
    _warmup_x = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device="cuda", dtype=torch.int32)
    _warmup_y = torch.randint(0, vocab_size, (device_batch_size, max_seq_len), device="cuda", dtype=torch.int64)
    _warmup_loss = model(_warmup_x, _warmup_y)
    _warmup_loss.backward()
    model.zero_grad(set_to_none=True)
    del _warmup_x, _warmup_y, _warmup_loss
print0("Compilation complete.")

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()

# Calculate iterations
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Iterations from data:param ratio: {num_iterations:,}")
total_tokens = total_batch_size * num_iterations
print0(f"Total training tokens: {total_tokens:,} (ratio: {total_tokens/num_params:.2f})")

# Initialize Optimizers
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr,
                                     matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# Initialize DataLoaders
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train",
                                                   fim_rate=fim_rate, fim_spm_rate=fim_spm_rate)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val",
                                                               fim_rate=0.0)  # no FIM for val
x, y = next(train_loader)

# LR schedule
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

    # Evaluate val bpb
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({"step": step, "val/bpb": val_bpb, "total_training_time": total_training_time})
        model.train()

    # Sample from model
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        engine = Engine(orig_model, tokenizer)
        # Standard completion samples
        prompts = [
            "def fibonacci(n):",
            "# Binary search implementation\ndef binary_search(",
            "class LinkedList:",
            "import torch\nimport torch.nn as nn\n\nclass",
        ]
        print0("--- Standard samples ---")
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=64, temperature=0)
            print0(tokenizer.decode(sample[0]))
            print0()
        # FIM completion sample
        fim_prefix = tokenizer.encode_special("<|fim_prefix|>")
        fim_suffix = tokenizer.encode_special("<|fim_suffix|>")
        fim_middle = tokenizer.encode_special("<|fim_middle|>")
        bos = tokenizer.get_bos_token_id()
        prefix_text = "def add(a, b):\n    "
        suffix_text = "\n    return result"
        fim_tokens = [bos, fim_prefix] + tokenizer.encode(prefix_text) + \
                     [fim_suffix] + tokenizer.encode(suffix_text) + [fim_middle]
        print0("--- FIM sample ---")
        print0(f"Prefix: {prefix_text!r}")
        print0(f"Suffix: {suffix_text!r}")
        with autocast_ctx:
            sample, _ = engine.generate_batch(fim_tokens, num_samples=1, max_tokens=32, temperature=0)
        print0(f"Middle: {tokenizer.decode(sample[0][len(fim_tokens):])!r}")
        print0()
        model.train()

    # Save checkpoint
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(get_base_dir(), "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir, step, orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {"step": step, "val_bpb": val_bpb, "model_config": model_config_kwargs,
             "user_config": user_config, "fim_rate": fim_rate, "fim_spm_rate": fim_spm_rate}
        )

    if last_step:
        break

    # Training step
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops = get_gpu_flops() * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops
    if step > 10:
        total_training_time += dt
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}")
    if step % 100 == 0:
        wandb_run.log({"step": step, "train/loss": debiased_smooth_loss, "train/mfu": mfu,
                        "total_training_time": total_training_time})

print0(f"Peak memory: {torch.cuda.max_memory_allocated()/1024/1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Min val bpb: {min_val_bpb:.4f}")

wandb_run.finish()
compute_cleanup()
