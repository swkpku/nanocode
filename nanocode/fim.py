"""
Fill-in-the-Middle (FIM) transformation for code pretraining.

Implements the FIM objective from "Efficient Training of Language Models to Fill in the Middle"
(Bavarian et al., 2022). At a configurable rate (default 50%), documents are split at random
character positions and rearranged into PSM (prefix-suffix-middle) or SPM (suffix-prefix-middle)
format with FIM special tokens.

The transformation operates on raw text (pre-tokenization) so each segment tokenizes independently.
"""

import random


def apply_fim(text, fim_rate=0.5, fim_spm_rate=0.5, rng=None):
    """
    Apply FIM transformation to a code document.

    Args:
        text: Raw code document string.
        fim_rate: Probability of applying FIM (default 0.5 = 50%).
        fim_spm_rate: When FIM is applied, probability of SPM vs PSM (default 0.5).
        rng: random.Random instance for reproducibility.

    Returns:
        If FIM applied: tuple of (segments, is_fim) where segments is a list of
            (special_token_or_none, text_content) pairs and is_fim=True.
        If not applied: tuple of ([(None, text)], False).

    FIM formats:
        PSM: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
        SPM: <|fim_prefix|> <|fim_suffix|> suffix <|fim_middle|> prefix middle
             (note: SPM puts prefix content after fim_middle, concatenated with middle)
    """
    if rng is None:
        rng = random.Random()

    # Decide whether to apply FIM at all
    if rng.random() >= fim_rate:
        return [(None, text)], False

    # Pick a random split point (character level)
    # We split the document into prefix, middle, suffix
    # by choosing two random cut points
    n = len(text)
    if n < 3:
        return [(None, text)], False

    # Choose two random positions and sort them
    pos1 = rng.randint(0, n)
    pos2 = rng.randint(0, n)
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1

    prefix = text[:pos1]
    middle = text[pos1:pos2]
    suffix = text[pos2:]

    # Decide PSM vs SPM
    use_spm = rng.random() < fim_spm_rate

    if use_spm:
        # SPM format: <|fim_prefix|> <|fim_suffix|> suffix <|fim_middle|> prefix middle
        segments = [
            ("<|fim_prefix|>", ""),
            ("<|fim_suffix|>", suffix),
            ("<|fim_middle|>", prefix + middle),
        ]
    else:
        # PSM format: <|fim_prefix|> prefix <|fim_suffix|> suffix <|fim_middle|> middle
        segments = [
            ("<|fim_prefix|>", prefix),
            ("<|fim_suffix|>", suffix),
            ("<|fim_middle|>", middle),
        ]

    return segments, True


def tokenize_fim_segments(segments, tokenizer):
    """
    Tokenize FIM segments, encoding each text segment independently.

    Args:
        segments: List of (special_token_name_or_none, text) pairs.
        tokenizer: Tokenizer with encode() and encode_special() methods.

    Returns:
        List of token ids.
    """
    ids = []
    for special_token, text in segments:
        if special_token is not None:
            ids.append(tokenizer.encode_special(special_token))
        if text:
            ids.extend(tokenizer.encode(text))
    return ids
