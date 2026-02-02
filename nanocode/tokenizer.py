"""
BPE Tokenizer for code with FIM special tokens.
Adapted from nanochat. Uses HuggingFace Tokenizers for training, tiktoken for inference.
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    "<|bos|>",              # 0: document delimiter
    "<|user_start|>",       # 1: SFT + Agent
    "<|user_end|>",         # 2: SFT + Agent
    "<|assistant_start|>",  # 3: SFT + Agent
    "<|assistant_end|>",    # 4: SFT + Agent (stop token)
    "<|tool_start|>",       # 5: SFT + Agent (tool call begin)
    "<|tool_end|>",         # 6: SFT + Agent (triggers execution)
    "<|output_start|>",     # 7: SFT + Agent (tool result begin)
    "<|output_end|>",       # 8: SFT + Agent (tool result end)
    "<|fim_prefix|>",       # 9: FIM pretraining
    "<|fim_middle|>",       # 10: FIM pretraining
    "<|fim_suffix|>",       # 11: FIM pretraining
]

# Code-aware BPE split pattern: preserves indentation, handles common code constructs
# Differences from GPT-4 pattern: \p{N}{1,2} instead of \p{N}{1,3} to save vocab for small models
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# HuggingFace Tokenizer wrapper
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False,
        ))
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# tiktoken-based tokenizer for efficient inference

import pickle
import tiktoken

class TiktokenTokenizer:
    """Wrapper around tiktoken for efficient inference. Train with HuggingFace, infer with tiktoken."""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_hf_tokenizer(cls, hf_tokenizer):
        """Convert a trained HuggingFace tokenizer to tiktoken format."""
        # Build reverse mapping: ByteLevel unicode char -> byte value
        # HuggingFace ByteLevel maps each byte (0-255) to a unicode character
        byte_level_chars = pre_tokenizers.ByteLevel.alphabet()
        # The alphabet is ordered: printable ASCII chars first, then shifted chars for others
        # Build the reverse: unicode char -> byte value
        char_to_byte = {}
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        for b, c in zip(bs, cs):
            char_to_byte[chr(c)] = b

        vocab = hf_tokenizer.tokenizer.get_vocab()
        mergeable_ranks = {}
        special_tokens = {}
        special_set = set(SPECIAL_TOKENS)
        for token_str, token_id in sorted(vocab.items(), key=lambda x: x[1]):
            if token_str in special_set:
                special_tokens[token_str] = token_id
            else:
                # Convert ByteLevel-encoded string back to raw bytes
                token_bytes = bytes([char_to_byte[c] for c in token_str])
                mergeable_ranks[token_bytes] = token_id
        enc = tiktoken.Encoding(
            name="nanocode",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single conversation.
        Returns:
        - ids: list[int] token ids
        - mask: list[int] mask = 1 for tokens that are supervised (assistant output)
        """
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        tool_start, tool_end = self.encode_special("<|tool_start|>"), self.encode_special("<|tool_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str)
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "tool_call":
                            add_tokens(tool_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(tool_end, 1)
                        elif part["type"] == "tool_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        """Render conversation priming the assistant for completion."""
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant"
        messages.pop()
        ids, mask = self.render_conversation(conversation)
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

    def visualize_tokenization(self, ids, mask):
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
        return '|'.join(tokens)


# -----------------------------------------------------------------------------
# Convenience functions

def _get_base_dir():
    """Lazy import to avoid circular dependency with common.py."""
    from nanocode.common import get_base_dir
    return get_base_dir()

def get_tokenizer():
    base_dir = _get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    return TiktokenTokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    base_dir = _get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? Run tok_train.py first"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
