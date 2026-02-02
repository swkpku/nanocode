"""
Code dataset download and language filtering for pretraining.

Primary source: codeparrot/github-code-clean (ungated, no auth needed).
Downloads parquet files directly from HuggingFace Hub, filters by quality,
writes to local parquet shards.
Languages: Python, JavaScript, TypeScript, Java, C, C++, Go, Rust.
"""

import os
import argparse
import time
import logging
import requests
import pyarrow as pa
import pyarrow.parquet as pq

from nanocode.common import get_base_dir, setup_default_logging

setup_default_logging()
logger = logging.getLogger(__name__)

# Supported languages and their dataset-specific names
LANGUAGES = ["python", "javascript", "typescript", "java", "c", "c++", "go", "rust"]

# Language name mapping for codeparrot/github-code-clean
# Note: Go is labeled "GO" (uppercase) in the all-all config
CODEPARROT_LANG_MAP = {
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "java": "Java",
    "c": "C",
    "c++": "C++",
    "go": "GO",
    "rust": "Rust",
}

# Sampling weights for round-robin across languages
LANGUAGE_WEIGHTS = {
    "python": 0.25,
    "javascript": 0.15,
    "typescript": 0.10,
    "java": 0.12,
    "c": 0.10,
    "c++": 0.10,
    "go": 0.08,
    "rust": 0.10,
}

# Quality filtering thresholds
MIN_FILE_CHARS = 50
MAX_FILE_CHARS = 100_000
MAX_LINE_LENGTH = 1000
MAX_AVG_LINE_LENGTH = 200

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "code_data")
os.makedirs(DATA_DIR, exist_ok=True)


def passes_quality_filter(content):
    """Basic quality heuristics for code files."""
    if not content or not isinstance(content, str):
        return False
    n = len(content)
    if n < MIN_FILE_CHARS or n > MAX_FILE_CHARS:
        return False
    lines = content.split('\n')
    if len(lines) == 0:
        return False
    max_line = max(len(line) for line in lines)
    if max_line > MAX_LINE_LENGTH:
        return False
    avg_line = n / len(lines)
    if avg_line > MAX_AVG_LINE_LENGTH:
        return False
    return True


def list_parquet_files(language=None, data_dir=None):
    """List all parquet files, optionally for a specific language."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    if language:
        lang_dir = os.path.join(data_dir, language)
        if not os.path.exists(lang_dir):
            return []
        parquet_files = sorted([
            os.path.join(lang_dir, f) for f in os.listdir(lang_dir)
            if f.endswith('.parquet') and not f.endswith('.tmp')
        ])
    else:
        parquet_files = []
        for lang in LANGUAGES:
            parquet_files.extend(list_parquet_files(language=lang, data_dir=data_dir))
    return parquet_files


def parquets_iter_batched(split="train", start=0, step=1, language=None):
    """
    Iterate through code dataset, yielding batches of code content strings.
    Applies quality filtering.
    """
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files(language=language)
    if not parquet_paths:
        return
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            if 'content' in rg.column_names:
                texts = rg.column('content').to_pylist()
            elif 'code' in rg.column_names:
                texts = rg.column('code').to_pylist()
            elif 'text' in rg.column_names:
                texts = rg.column('text').to_pylist()
            else:
                continue
            filtered = [t for t in texts if passes_quality_filter(t)]
            if filtered:
                yield filtered


def multi_language_iter_batched(split="train", start=0, step=1):
    """Weighted round-robin sampling across all languages."""
    import random
    lang_iters = {}
    for lang in LANGUAGES:
        lang_iters[lang] = parquets_iter_batched(split=split, start=start, step=step, language=lang)

    rng = random.Random(42 + start)
    languages = list(LANGUAGE_WEIGHTS.keys())
    weights = [LANGUAGE_WEIGHTS[l] for l in languages]

    exhausted = set()
    while len(exhausted) < len(languages):
        lang = rng.choices(languages, weights=weights, k=1)[0]
        if lang in exhausted:
            continue
        try:
            batch = next(lang_iters[lang])
            yield batch
        except StopIteration:
            exhausted.add(lang)
            continue


# Language configs available as dedicated subsets in codeparrot/github-code-clean
# Languages not in this map need to be extracted from the "all-all" config
CODEPARROT_CONFIGS = {
    "python": "Python-all",
    "javascript": "JavaScript-all",
    "java": "Java-all",
    "c": "C-all",
    "c++": "C++-all",
}


def _get_parquet_urls(language):
    """Get parquet file URLs for a language from the HuggingFace API."""
    repo = "codeparrot/github-code-clean"
    config = CODEPARROT_CONFIGS.get(language)
    if config:
        api_url = f"https://huggingface.co/api/datasets/{repo}/parquet/{config}/train"
    else:
        # Go, Rust, TypeScript: use all-all and filter by language column
        api_url = f"https://huggingface.co/api/datasets/{repo}/parquet/all-all/train"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    return resp.json(), config is not None


def _download_and_filter_parquet(url, language, needs_lang_filter):
    """Download a remote parquet, filter by language+quality, return content list."""
    import io
    fname = url.split('/')[-1]
    logger.info(f"  Fetching {fname}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    table = pq.read_table(io.BytesIO(resp.content))
    codes = table.column('code').to_pylist()
    if needs_lang_filter:
        hf_lang = CODEPARROT_LANG_MAP[language]
        langs = table.column('language').to_pylist()
        return [c for c, l in zip(codes, langs) if l == hf_lang and passes_quality_filter(c)]
    else:
        return [c for c in codes if passes_quality_filter(c)]


def download_from_huggingface(language, num_shards=10, source="codeparrot"):
    """Download code data for a specific language from codeparrot/github-code-clean."""
    lang_dir = os.path.join(DATA_DIR, language)
    os.makedirs(lang_dir, exist_ok=True)

    existing = [f for f in os.listdir(lang_dir) if f.endswith('.parquet')]
    if len(existing) >= num_shards:
        logger.info(f"Already have {len(existing)} shards for {language}, skipping")
        return

    logger.info(f"Downloading {language} from codeparrot/github-code-clean...")
    parquet_urls, has_dedicated_config = _get_parquet_urls(language)
    needs_lang_filter = not has_dedicated_config

    batch = []
    batch_size = 10000
    shard_idx = len(existing)
    total_docs = 0

    for url in parquet_urls:
        if shard_idx >= num_shards:
            break
        contents = _download_and_filter_parquet(url, language, needs_lang_filter)
        batch.extend(contents)
        total_docs += len(contents)
        logger.info(f"  {language}: {total_docs:,} docs so far")

        while len(batch) >= batch_size and shard_idx < num_shards:
            table = pa.table({"content": batch[:batch_size]})
            shard_path = os.path.join(lang_dir, f"shard_{shard_idx:05d}.parquet")
            pq.write_table(table, shard_path)
            logger.info(f"  {language}: wrote shard {shard_idx}")
            batch = batch[batch_size:]
            shard_idx += 1

    # Write remaining batch
    if batch and shard_idx < num_shards:
        table = pa.table({"content": batch})
        shard_path = os.path.join(lang_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path)
        shard_idx += 1

    logger.info(f"  {language}: done ({shard_idx} shards, {total_docs:,} docs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download code data")
    parser.add_argument("-l", "--languages", type=str, default="all",
                        help="Comma-separated languages or 'all' (default: all)")
    parser.add_argument("-n", "--num-shards", type=int, default=10,
                        help="Shards per language (default: 10)")
    args = parser.parse_args()

    if args.languages == "all":
        langs = list(LANGUAGES)
    else:
        langs = [l.strip() for l in args.languages.split(",")]

    print(f"Downloading {len(langs)} languages, {args.num_shards} shards each")
    print(f"Target directory: {DATA_DIR}")

    for lang in langs:
        print(f"\n--- {lang} ---")
        download_from_huggingface(lang, num_shards=args.num_shards)

    print("\nDone!")
