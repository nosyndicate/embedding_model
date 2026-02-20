#!/usr/bin/env python3
"""
Pre-tokenization script for FineWeb-Edu dataset (RoBERTa-style).

Produces a flat token stream with [SEP] as document boundary markers.
No [CLS] tokens are inserted. The resulting shards are sequence-length-agnostic:
the same pre-tokenized data can be used with any max_seq_length at training time.

We usually do not need BOS tokens, in BERT, this is [CLS], in RoBERTa, this is <s>.
We only need EOS tokens, in BERT, this is [SEP], in RoBERTa, this is </s> as boundary markers.

The script is designed to satisfy the following conditions:
- Pretokenize once only but support different seq length samples
- Allow cross document boundaries
- Support dynamic masking

Specifically, when we say allow cross document boundaries (FULL SENTENCE method in RoBERTa), we mean two situations
- A sample may contain multiple documents
  - No strong evidence suggest that attention cross the document boundary would hurt the result on short sequences.
    Longer sequence might suffer from the packing multiple documents into a single sample (Llama 3).
  - Since we allow allow attention to cross document boundaries, we don't need to reset position id between documents.
  - In future, we can try to implement diagonal attention makes to prevent attention across different documents.

- A document can span two samples
  - This means we don't pad at the end of a sample, instead, we break the next document into two piece and have
    the first part in the current sample and the remainder in the next sample.
  - The next sample might lose some context since the first part of the document is in the previous sample.
  - This is mostly a computation resouce optimization.
  - But always have the position 0 being the first token of a document might introduce unintended biases.
  - To avoid the concern of losing context, we can do two things:
    - Shift segmentation each epoch: instead of always using windows aligned to the same boundaries, use a random
      global offset per epoch (step)
    - Use a small overlap (stride < seq_len)
Usage:
    python scripts/tokenize_fineweb_roberta.py --version 10B --output_dir data/fineweb_edu_10bt
    python scripts/tokenize_fineweb_roberta.py --version 100B --shard_size 100000000

Shard format (v3):
    Header (256 int32):
        [0] = 20240520    # magic number
        [1] = 3           # version (3 for RoBERTa-style flat stream)
        [2] = token_count # number of uint16 tokens
        [3] = vocab_size  # 50265 for roberta-base
        [4] = pad_id      # 1 for roberta-base (<pad>)
        [5] = cls_id      # 0 for roberta-base (<s>)
        [6] = sep_id      # 2 for roberta-base (</s>)
        [7] = mask_id     # 50264 for roberta-base (<mask>)

    Tokens (uint16[]):
        tok tok tok ... [SEP] tok tok tok ... [SEP] ...
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Suppress tokenizer warnings about sequence length
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# Shard format constants
HEADER_SIZE = 256
MAGIC_NUMBER = 20240520
ROBERTA_VERSION = 3

# Version to HuggingFace dataset mapping
VERSION_MAP = {
    "10B": ("sample-10BT", "fineweb_edu_10bt"),
    "100B": ("sample-100BT", "fineweb_edu_100bt"),
    "350B": ("sample-350BT", "fineweb_edu_350bt"),
}

# Global tokenizer (initialized in worker processes)
_tokenizer: AutoTokenizer | None = None


def init_worker(tokenizer_name: str) -> None:
    """Initialize tokenizer in worker process."""
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def tokenize_document(doc: dict) -> np.ndarray:
    """
    Tokenize a single document without [CLS], appending [SEP] as boundary.

    Args:
        doc: Document dict with "text" field

    Returns:
        numpy array of uint16 token IDs: [tok, tok, ..., SEP]
    """
    global _tokenizer
    assert _tokenizer is not None, "Tokenizer not initialized"

    # Tokenize WITHOUT special tokens
    tokens = _tokenizer.encode(
        doc["text"],
        add_special_tokens=False,
        truncation=False,
    )

    # Append [SEP] as document boundary marker
    tokens.append(_tokenizer.sep_token_id)

    return np.array(tokens, dtype=np.uint16)


def write_shard(
    filename: Path,
    tokens: np.ndarray,
    tokenizer: AutoTokenizer,
) -> None:
    """
    Write tokens to a binary shard file (v3 format).

    Args:
        filename: Output file path
        tokens: numpy array of uint16 tokens
        tokenizer: BERT tokenizer (for special token IDs)
    """
    assert len(tokens) < 2**31, f"Token count {len(tokens)} too large for header"

    header = np.zeros(HEADER_SIZE, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = ROBERTA_VERSION
    header[2] = len(tokens)
    header[3] = tokenizer.vocab_size
    header[4] = tokenizer.pad_token_id
    header[5] = tokenizer.cls_token_id
    header[6] = tokenizer.sep_token_id
    header[7] = tokenizer.mask_token_id

    if not isinstance(tokens, np.ndarray) or tokens.dtype != np.uint16:
        assert (tokens >= 0).all() and (tokens < 2**16).all(), (
            "Tokens must fit in uint16"
        )
        tokens = tokens.astype(np.uint16)

    print(f"Writing {len(tokens):,} tokens to {filename}")

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize FineWeb-Edu dataset (RoBERTa-style flat stream)"
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        default="100B",
        choices=["10B", "100B", "350B"],
        help="Dataset version: 10B, 100B, or 350B tokens",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: data/fineweb_edu_<version>)",
    )
    parser.add_argument(
        "-s",
        "--shard_size",
        type=int,
        default=100_000_000,
        help="Target number of tokens per shard (default: 100M)",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="roberta-base",
        help="RoBERTa tokenizer name or path",
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 2)",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing)",
    )
    parser.add_argument(
        "--val_shards",
        type=int,
        default=0,
        help="Number of leading shards to designate as validation (0 for none)",
    )
    args = parser.parse_args()

    if args.version not in VERSION_MAP:
        raise ValueError(f"Unknown version: {args.version}")

    remote_name, default_dir = VERSION_MAP[args.version]
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path("data") / default_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("FineWeb-Edu Pre-tokenization (RoBERTa-style)")
    print(f"  Version: {args.version} ({remote_name})")
    print(f"  Output: {output_dir}")
    print(f"  Shard size: {args.shard_size:,} tokens")
    print(f"  Tokenizer: {args.tokenizer}")
    print("  Format: v3 (flat stream with [SEP] separators)")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(
        f"  Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, "
        f"PAD={tokenizer.pad_token_id}, MASK={tokenizer.mask_token_id}"
    )

    print("\nLoading dataset from HuggingFace...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=remote_name,
        split="train",
        streaming=True,
    )

    if args.max_docs:
        dataset = dataset.take(args.max_docs)

    cpu_count: int = os.cpu_count() or 1
    num_workers = args.num_workers or max(1, cpu_count - 2)
    print(f"  Workers: {num_workers}")

    shard_index = 0
    token_buffer = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(num_workers, initializer=init_worker, initargs=(args.tokenizer,))

    try:
        for tokens in pool.imap(tokenize_document, dataset, chunksize=16):
            while token_count + len(tokens) >= args.shard_size:
                remainder = args.shard_size - token_count
                start, end = token_count, token_count + remainder
                token_buffer[start:end] = tokens[:remainder]

                split = "val" if shard_index < args.val_shards else "train"
                filename = output_dir / f"fineweb_{split}_{shard_index:06d}.bin"
                write_shard(filename, token_buffer, tokenizer)

                tokens = tokens[remainder:]
                shard_index += 1
                token_count = 0

                if progress_bar is not None:
                    progress_bar.close()
                progress_bar = tqdm(
                    total=args.shard_size,
                    unit="tokens",
                    desc=f"Shard {shard_index}",
                )

            if len(tokens) > 0:
                start, end = token_count, token_count + len(tokens)
                token_buffer[start:end] = tokens
                token_count += len(tokens)

                if progress_bar is None:
                    progress_bar = tqdm(
                        total=args.shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                progress_bar.update(len(tokens))

    finally:
        pool.close()
        pool.join()

    if token_count > 0:
        split = "val" if shard_index < args.val_shards else "train"
        filename = output_dir / f"fineweb_{split}_{shard_index:06d}.bin"
        write_shard(filename, token_buffer[:token_count], tokenizer)

    if progress_bar is not None:
        progress_bar.close()

    print(f"\nDone! Wrote {shard_index + 1} shards to {output_dir}")


if __name__ == "__main__":
    main()
