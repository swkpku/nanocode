"""
Engine for efficient inference with KV-cache and tool execution.

The engine coordinates:
1. KV-cached autoregressive generation
2. Tool call detection and dispatch (on <|tool_end|> token)
3. Tool result injection as forced tokens

Tool format: JSON between <|tool_start|> and <|tool_end|>
  {"name": "bash", "arguments": {"command": "ls -la"}}
"""

import json
import torch
import torch.nn.functional as F
from collections import deque

from nanocode.common import compute_init
from nanocode.checkpoint_manager import load_model
from nanocode.execution import execute_bash, execute_code


# -----------------------------------------------------------------------------
class KVCache:
    """KV cache that works hand-in-hand with the GPT model."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        assert self.kv_cache is None
        assert other.kv_cache is not None
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                assert dim1 == dim2
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1
            elif ix == 4:
                assert dim1 >= dim2
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024
            t_needed = (t_needed + 1023) & ~1023
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from logits of shape (B, vocab_size)."""
    assert temperature >= 0.0
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------
# Tool execution

def execute_tool(tool_call_json):
    """
    Parse and execute a tool call.
    Expected format: {"name": "bash"|"read_file"|"write_file"|"edit_file", "arguments": {...}}
    Returns the tool output as a string.
    """
    try:
        call = json.loads(tool_call_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in tool call: {e}"

    name = call.get("name", "")
    args = call.get("arguments", {})

    if name == "bash":
        command = args.get("command", "")
        result = execute_bash(command, timeout=30.0)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.error:
            output += f"\nERROR: {result.error}"
        return output.strip() or "(no output)"

    elif name == "read_file":
        path = args.get("path", "")
        try:
            with open(path, "r") as f:
                content = f.read()
            return content[:50000]  # cap length
        except Exception as e:
            return f"Error reading {path}: {e}"

    elif name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote {len(content)} chars to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"

    elif name == "edit_file":
        path = args.get("path", "")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")
        try:
            with open(path, "r") as f:
                content = f.read()
            if old_string not in content:
                return f"Error: old_string not found in {path}"
            content = content.replace(old_string, new_string, 1)
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing {path}: {e}"

    else:
        return f"Error: Unknown tool '{name}'"


import os  # needed for write_file tool


# -----------------------------------------------------------------------------
class RowState:
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()
        self.in_tool_block = False
        self.tool_call_tokens = []
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int)
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
        tool_start = get_special("<|tool_start|>")
        tool_end = get_special("<|tool_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # 1) Batch-1 prefill
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(batch_size=1, seq_len=len(tokens), **kv_model_kwargs)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate KV cache
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(batch_size=num_samples, seq_len=kv_length_hint, **kv_model_kwargs)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Initialize row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]
                next_ids = sample_next_token(logits, rng, temperature, top_k)
                sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                if next_token == assistant_end or next_token == bos:
                    state.completed = True

                # Tool call state machine
                if next_token == tool_start:
                    state.in_tool_block = True
                    state.tool_call_tokens = []
                elif next_token == tool_end and state.in_tool_block:
                    state.in_tool_block = False
                    if state.tool_call_tokens:
                        tool_json = self.tokenizer.decode(state.tool_call_tokens)
                        result = execute_tool(tool_json)
                        result_tokens = self.tokenizer.encode(result)
                        state.forced_tokens.append(output_start)
                        state.forced_tokens.extend(result_tokens)
                        state.forced_tokens.append(output_end)
                    state.tool_call_tokens = []
                elif state.in_tool_block:
                    state.tool_call_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation returning final token sequences."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
