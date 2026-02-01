# Data Pipeline: FineWeb-Edu with BERT Pre-tokenization

## Overview

Pre-tokenized data pipeline for FineWeb-Edu using BERT tokenizer, with sharded binary storage for fast training.

**Key Decisions:**
- **Pre-tokenization** (not on-the-fly) - Zero tokenization overhead during training
- **BERT tokenizer** (WordPiece) - Standard [CLS], [SEP], [MASK] tokens for MLM/ELECTRA
- **Default subset**: `sample-100BT` (100 billion tokens, ~200GB tokenized)
- **Collators handle masking only** - No tokenization at training time
- **Document boundary awareness** - Optional `doc_aware` packing mode respects [SEP][CLS] boundaries

---

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  ONE-TIME: Pre-tokenization                                  │
│  python scripts/tokenize_fineweb.py --version 100B          │
│                                                              │
│  FineWeb-Edu (HuggingFace) ──► BERT tokenizer ──► .bin shards│
│  sample-100BT                  30522 vocab       data/       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINING: Load pre-tokenized shards                        │
│                                                              │
│  PreTokenizedDataset ──► MLMCollator ──► Model              │
│  (memory-mapped .bin)    (masking)       (BERT)             │
│                                                              │
│  - Zero tokenization overhead                                │
│  - Fast random access via mmap                               │
│  - Shard shuffling per epoch                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/embedding_trainer/data/
├── __init__.py
├── registry.py              # DATASET_REGISTRY, COLLATOR_REGISTRY
├── base.py                  # Protocols: DatasetProtocol, CollatorProtocol
├── loader.py                # DataLoader factory
├── datasets/
│   ├── __init__.py
│   └── pretokenized.py      # PreTokenizedDataset (loads .bin shards)
└── collators/
    ├── __init__.py
    ├── base.py              # BaseCollator
    ├── mlm.py               # MLMCollator (masking on pre-tokenized data)
    ├── electra.py           # ELECTRACollator
    └── span_corruption.py   # SpanCorruptionCollator

scripts/
└── tokenize_fineweb.py      # Pre-tokenization script (BERT tokenizer)

configs/
├── data/
│   ├── pretokenized.yaml    # Pre-tokenized shard loading
│   ├── fineweb_edu_10bt.yaml
│   ├── fineweb_edu_100bt.yaml
│   └── fineweb_edu_350bt.yaml
├── tokenizer/
│   ├── bert_base.yaml
│   └── bert_large.yaml
└── collator/
    ├── mlm.yaml
    ├── mlm_wwm.yaml
    ├── electra.yaml
    └── span_corruption.yaml

data/                        # Pre-tokenized shards (gitignored)
└── fineweb_edu_100bt/
    ├── fineweb_train_000000.bin
    ├── fineweb_train_000001.bin
    └── ...
```

---

## Implementation Steps

### Phase 1: Pre-tokenization Script

**File: `scripts/tokenize_fineweb.py`**

```python
"""
Pre-tokenize FineWeb-Edu with BERT tokenizer.

Usage:
    python scripts/tokenize_fineweb.py --version 100B --shard_size 100000000

Key differences from fineweb.example.py:
1. BERT tokenizer instead of tiktoken
2. Add [CLS] and [SEP] tokens per document
3. Store vocab size and special token IDs in header
4. Support sequence chunking for long documents
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm

def write_datafile(filename, toks, tokenizer):
    """
    Saves token data as a .bin file.
    - Header: 256 int32s (magic, version, token_count, vocab info)
    - Tokens: uint16 array
    """
    assert len(toks) < 2**31, "token count too large"

    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520      # magic
    header[1] = 2             # version (2 for BERT)
    header[2] = len(toks)     # token count
    header[3] = tokenizer.vocab_size
    header[4] = tokenizer.pad_token_id
    header[5] = tokenizer.cls_token_id
    header[6] = tokenizer.sep_token_id
    header[7] = tokenizer.mask_token_id

    toks_np = np.array(toks, dtype=np.uint16)

    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

def tokenize(doc, tokenizer):
    """Tokenize a single document with [CLS] and [SEP]."""
    tokens = tokenizer.encode(doc["text"], add_special_tokens=True)
    return np.array(tokens, dtype=np.uint16)
```

**Features:**
- Load FineWeb-Edu via `load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)`
- BERT tokenizer with [CLS]...[SEP] per document
- Parallel tokenization with `multiprocessing.Pool`
- Shards of ~100M tokens each
- CLI args: `--version 10B|100B|350B`, `--shard_size`

---

### Phase 2: Data Foundation

**File: `src/embedding_trainer/data/base.py`**

```python
from typing import Protocol, Iterator, Dict, Any, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import IterableDataset

@dataclass
class ShardInfo:
    """Metadata for a token shard."""
    path: str
    token_count: int
    split: str  # "train" or "val"

class DatasetProtocol(Protocol):
    """Protocol for datasets."""

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        ...

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling."""
        ...

class CollatorProtocol(Protocol):
    """Protocol for data collators."""

    def __call__(self, examples: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ...

@dataclass
class PreTokenizedBatch:
    """Typed batch from collator."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
```

**File: `src/embedding_trainer/data/registry.py`**

```python
from embedding_trainer.utils.registry import Registry

DATASET_REGISTRY = Registry("dataset")
COLLATOR_REGISTRY = Registry("collator")
```

---

### Phase 3: Pre-tokenized Dataset

**File: `src/embedding_trainer/data/datasets/pretokenized.py`**

```python
from dataclasses import dataclass
from typing import Iterator, Dict, Any, List, Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset
import glob
import random

from embedding_trainer.data.registry import DATASET_REGISTRY

@dataclass
class PreTokenizedConfig:
    """Configuration for pre-tokenized dataset."""
    data_dir: str
    max_seq_length: int = 512
    shuffle: bool = True
    seed: int = 42

    # Document boundary handling
    packing_mode: str = "flat"  # "flat" | "doc_aware"
    # flat: chunk tokens at fixed intervals (may split documents)
    # doc_aware: never split across [SEP][CLS] boundaries

    # Sequence sampling strategy (for flat mode)
    sampling_strategy: str = "epoch_offset"  # "fixed" | "epoch_offset" | "random" | "shuffle_indices"
    # fixed: always start at 0 with stride=seq_len (same sequences every epoch)
    # epoch_offset: offset = (epoch * seq_len // num_offsets) % seq_len
    # random: sample random start offsets within a buffer
    # shuffle_indices: shuffle sequence indices, then iterate

    num_offset_phases: int = 4  # for epoch_offset: cycle through N different offsets
    random_buffer_size: int = 1000  # for random: buffer size for sampling offsets

    # Distributed training
    world_size: int = 1
    rank: int = 0

@DATASET_REGISTRY.register("pretokenized")
class PreTokenizedDataset(IterableDataset):
    """
    Dataset that loads pre-tokenized .bin shards.

    Features:
    - Memory-maps shards for efficient reading
    - Chunks token stream into max_seq_length sequences
    - Shuffles shard order per epoch
    - Supports distributed training via shard assignment
    """

    HEADER_SIZE = 256 * 4  # 256 int32s = 1024 bytes

    def __init__(self, config: PreTokenizedConfig):
        self.config = config
        self._epoch = 0
        self._shards = self._discover_shards()

    def _discover_shards(self) -> List[str]:
        """Find all .bin shards in data_dir."""
        pattern = f"{self.config.data_dir}/*.bin"
        shards = sorted(glob.glob(pattern))
        if not shards:
            raise ValueError(f"No .bin shards found in {self.config.data_dir}")
        return shards

    def _get_shards_for_epoch(self) -> List[str]:
        """Get shuffled shards for this epoch and rank."""
        shards = self._shards.copy()

        if self.config.shuffle:
            rng = random.Random(self.config.seed + self._epoch)
            rng.shuffle(shards)

        # Distribute shards across ranks
        if self.config.world_size > 1:
            shards = shards[self.config.rank::self.config.world_size]

        return shards

    def _read_shard_header(self, path: str) -> Dict[str, int]:
        """Read shard header to get metadata."""
        with open(path, "rb") as f:
            header = np.frombuffer(f.read(self.HEADER_SIZE), dtype=np.int32)
        return {
            "magic": header[0],
            "version": header[1],
            "token_count": header[2],
            "vocab_size": header[3],
            "pad_id": header[4],
            "cls_id": header[5],
            "sep_id": header[6],
            "mask_id": header[7],
        }

    def _mmap_shard(self, path: str) -> np.ndarray:
        """Memory-map a shard's tokens."""
        header = self._read_shard_header(path)
        token_count = header["token_count"]

        # Memory-map just the tokens (skip header)
        return np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
            offset=self.HEADER_SIZE,
            shape=(token_count,),
        )

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for reproducible shuffling."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        shards = self._get_shards_for_epoch()
        seq_len = self.config.max_seq_length

        for shard_path in shards:
            tokens = self._mmap_shard(shard_path)
            header = self._read_shard_header(shard_path)

            if self.config.packing_mode == "flat":
                yield from self._iter_flat(tokens, seq_len)

            elif self.config.packing_mode == "doc_aware":
                # Never split across [SEP][CLS] boundaries
                yield from self._iter_doc_aware(tokens, header, seq_len)

    def _iter_flat(
        self, tokens: np.ndarray, seq_len: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over tokens in flat mode with configurable sampling strategy.
        """
        n_tokens = len(tokens)
        n_sequences = (n_tokens - seq_len) // seq_len + 1
        strategy = self.config.sampling_strategy

        if strategy == "fixed":
            # Always same sequences every epoch
            for start in range(0, n_tokens - seq_len + 1, seq_len):
                yield self._make_chunk(tokens, start, seq_len)

        elif strategy == "epoch_offset":
            # Rotate starting offset each epoch
            # Cycle through num_offset_phases different offsets
            phase = self._epoch % self.config.num_offset_phases
            offset = (phase * seq_len) // self.config.num_offset_phases
            for start in range(offset, n_tokens - seq_len + 1, seq_len):
                yield self._make_chunk(tokens, start, seq_len)

        elif strategy == "random":
            # Sample random offsets from a buffer
            rng = np.random.default_rng(self.config.seed + self._epoch)
            buffer_size = min(self.config.random_buffer_size, n_sequences)

            # Generate all valid start positions
            valid_starts = np.arange(0, n_tokens - seq_len + 1)

            # Yield in batches using reservoir-style sampling
            for batch_start in range(0, len(valid_starts), buffer_size):
                batch_indices = valid_starts[batch_start:batch_start + buffer_size]
                rng.shuffle(batch_indices)
                for start in batch_indices:
                    yield self._make_chunk(tokens, start, seq_len)

        elif strategy == "shuffle_indices":
            # Treat each sequence position as an index, shuffle them
            rng = np.random.default_rng(self.config.seed + self._epoch)
            # Non-overlapping sequence indices
            indices = np.arange(0, n_tokens - seq_len + 1, seq_len)
            rng.shuffle(indices)
            for start in indices:
                yield self._make_chunk(tokens, start, seq_len)

    def _make_chunk(
        self, tokens: np.ndarray, start: int, seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Extract a chunk and convert to tensor."""
        chunk = tokens[start:start + seq_len].copy()
        return {"input_ids": torch.from_numpy(chunk.astype(np.int64))}

    def _iter_doc_aware(
        self, tokens: np.ndarray, header: Dict[str, int], seq_len: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield sequences that respect document boundaries.

        Strategy:
        - Find all [CLS] positions (document starts)
        - Pack complete documents into sequences up to seq_len
        - If a single document exceeds seq_len, split it but keep [CLS] at start
        - Pad short sequences if needed
        """
        cls_id = header["cls_id"]
        sep_id = header["sep_id"]
        pad_id = header["pad_id"]

        # Find document boundaries (positions of [CLS] tokens)
        cls_positions = np.where(tokens == cls_id)[0].tolist()
        cls_positions.append(len(tokens))  # Add end sentinel

        buffer = []
        for i in range(len(cls_positions) - 1):
            doc_start = cls_positions[i]
            doc_end = cls_positions[i + 1]
            doc_tokens = tokens[doc_start:doc_end].tolist()

            # If single doc exceeds seq_len, split it
            if len(doc_tokens) > seq_len:
                # Yield current buffer first
                if buffer:
                    yield self._pad_and_yield(buffer, seq_len, pad_id)
                    buffer = []

                # Split long document into chunks
                for chunk_start in range(0, len(doc_tokens), seq_len):
                    chunk = doc_tokens[chunk_start:chunk_start + seq_len]
                    yield self._pad_and_yield(chunk, seq_len, pad_id)

            # If doc fits in remaining buffer space
            elif len(buffer) + len(doc_tokens) <= seq_len:
                buffer.extend(doc_tokens)

            # Buffer full, yield and start new buffer with this doc
            else:
                yield self._pad_and_yield(buffer, seq_len, pad_id)
                buffer = doc_tokens

        # Yield remaining buffer
        if buffer:
            yield self._pad_and_yield(buffer, seq_len, pad_id)

    def _pad_and_yield(
        self, tokens: list, seq_len: int, pad_id: int
    ) -> Dict[str, torch.Tensor]:
        """Pad tokens to seq_len and return as tensor dict."""
        if len(tokens) < seq_len:
            tokens = tokens + [pad_id] * (seq_len - len(tokens))
        return {"input_ids": torch.tensor(tokens, dtype=torch.long)}

    def state_dict(self) -> Dict[str, Any]:
        """Save state for checkpointing."""
        return {"epoch": self._epoch}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        self._epoch = state["epoch"]
```

---

### Phase 4: Collator Implementation

**File: `src/embedding_trainer/data/collators/mlm.py`**

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import torch

from embedding_trainer.data.registry import COLLATOR_REGISTRY

@dataclass
class MLMCollatorConfig:
    """Configuration for MLM collator."""
    mlm_probability: float = 0.15
    mask_replace_prob: float = 0.8    # Replace with [MASK]
    random_replace_prob: float = 0.1  # Replace with random token
    # Remaining 0.1 kept as original

    # BERT special token IDs
    vocab_size: int = 30522
    pad_token_id: int = 0
    cls_token_id: int = 101
    sep_token_id: int = 102
    mask_token_id: int = 103

    pad_to_multiple_of: Optional[int] = 8

@COLLATOR_REGISTRY.register("mlm")
class MLMCollator:
    """
    Data collator for Masked Language Modeling.

    Input: List of {"input_ids": Tensor} from PreTokenizedDataset
    Output: {"input_ids": masked, "labels": original, "attention_mask": mask}
    """

    def __init__(self, config: MLMCollatorConfig):
        self.config = config

        # Special tokens to never mask
        self.special_token_ids = {
            config.pad_token_id,
            config.cls_token_id,
            config.sep_token_id,
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input_ids
        input_ids = torch.stack([ex["input_ids"] for ex in examples])

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.config.pad_token_id).long()

        # Apply masking
        masked_ids, labels = self._mask_tokens(input_ids)

        return {
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _mask_tokens(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking."""
        labels = input_ids.clone()

        # Create mask of positions eligible for masking
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_mask |= (input_ids == token_id)

        # Sample positions to mask
        probability_matrix = torch.full(
            input_ids.shape, self.config.mlm_probability
        )
        probability_matrix.masked_fill_(special_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Labels: -100 for non-masked (ignored in loss)
        labels[~masked_indices] = -100

        # 80% -> [MASK]
        indices_replaced = (
            torch.bernoulli(
                torch.full(input_ids.shape, self.config.mask_replace_prob)
            ).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.config.mask_token_id

        # 10% -> random token
        remaining = masked_indices & ~indices_replaced
        indices_random = (
            torch.bernoulli(
                torch.full(input_ids.shape, 0.5)  # 50% of remaining 20%
            ).bool() & remaining
        )
        random_words = torch.randint(
            self.config.vocab_size, input_ids.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # 10% -> keep original (no change needed)

        return input_ids, labels
```

**File: `src/embedding_trainer/data/collators/electra.py`**

```python
@COLLATOR_REGISTRY.register("electra")
class ELECTRACollator:
    """
    Data collator for ELECTRA pretraining.

    Output:
        - input_ids: Original tokens (for discriminator labels)
        - masked_input_ids: With [MASK] tokens (generator input)
        - attention_mask: Attention mask
        - mlm_labels: Generator labels (-100 for non-masked)
        - is_masked: Boolean mask of masked positions
    """
    # Similar to MLMCollator but outputs both original and masked
```

**File: `src/embedding_trainer/data/collators/span_corruption.py`**

```python
@COLLATOR_REGISTRY.register("span_corruption")
class SpanCorruptionCollator:
    """
    Data collator for T5-style span corruption.

    Replaces random spans with sentinel tokens.
    Creates encoder input (with sentinels) and decoder target (original spans).
    """
    # Implements random span selection and sentinel token replacement
```

---

### Phase 5: DataLoader Factory

**File: `src/embedding_trainer/data/loader.py`**

```python
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader, IterableDataset

@dataclass
class DataLoaderConfig:
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    drop_last: bool = True

def create_dataloader(
    dataset: IterableDataset,
    collate_fn,
    config: DataLoaderConfig,
) -> DataLoader:
    """Create DataLoader for streaming datasets."""
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
```

---

### Phase 6: Hydra Configs

**File: `configs/data/pretokenized.yaml`**

```yaml
_target_: embedding_trainer.data.datasets.pretokenized.PreTokenizedDataset

data_dir: "${hydra:runtime.cwd}/data/fineweb_edu_100bt"
max_seq_length: 512
shuffle: true  # shuffle shard order
seed: ${seed}

# Document boundary handling
# - flat: chunk at fixed intervals (faster, may split documents)
# - doc_aware: never split across [SEP][CLS] (respects document boundaries)
packing_mode: flat

# Sequence sampling strategy (for flat mode only)
# - fixed: always start at 0, same sequences every epoch
# - epoch_offset: rotate offset each epoch (recommended)
# - random: sample random start offsets from buffer
# - shuffle_indices: shuffle sequence indices then iterate
sampling_strategy: epoch_offset
num_offset_phases: 4      # for epoch_offset
random_buffer_size: 1000  # for random

world_size: ${distributed.world_size:1}
rank: ${distributed.rank:0}
```

**File: `configs/data/pretokenized_doc_aware.yaml`**

```yaml
defaults:
  - pretokenized

packing_mode: doc_aware
```

**File: `configs/collator/mlm.yaml`**

```yaml
_target_: embedding_trainer.data.collators.mlm.MLMCollator

mlm_probability: 0.15
mask_replace_prob: 0.8
random_replace_prob: 0.1
vocab_size: 30522
pad_token_id: 0
cls_token_id: 101
sep_token_id: 102
mask_token_id: 103
pad_to_multiple_of: 8
```

---

## FineWeb-Edu Dataset Reference

**Available subsets:**
| Subset | Tokens | Tokenized Size (approx) |
|--------|--------|-------------------------|
| `sample-10BT` | 10B | ~20 GB |
| `sample-100BT` | 100B | ~200 GB |
| `sample-350BT` | 350B | ~700 GB |
| `default` | 1.3T | ~2.6 TB |

**Shard format:**
```
Header (256 int32, 1024 bytes):
  [0] = 20240520    # magic number
  [1] = 2           # version (2 for BERT)
  [2] = token_count # number of uint16 tokens
  [3] = vocab_size  # 30522 for bert-base-uncased
  [4] = pad_id      # 0
  [5] = cls_id      # 101
  [6] = sep_id      # 102
  [7] = mask_id     # 103

Tokens (uint16[token_count]):
  [CLS] tok tok tok ... [SEP] [CLS] tok tok ... [SEP] ...
```

---

## Verification

```bash
# Step 1: Pre-tokenize (start with 10B for testing)
python scripts/tokenize_fineweb.py --version 10B --shard_size 100000000

# Step 2: Verify shard contents
python -c "
import numpy as np
with open('data/fineweb_edu_10bt/fineweb_train_000000.bin', 'rb') as f:
    header = np.frombuffer(f.read(256*4), dtype=np.int32)
    print(f'Magic: {header[0]}, Version: {header[1]}, Tokens: {header[2]:,}')
    print(f'Vocab: {header[3]}, PAD: {header[4]}, CLS: {header[5]}, SEP: {header[6]}, MASK: {header[7]}')
    tokens = np.frombuffer(f.read(1000*2), dtype=np.uint16)
    print(f'First 20 tokens: {tokens[:20]}')
"

# Step 3: Test dataset loading
python -c "
from embedding_trainer.data.datasets.pretokenized import PreTokenizedDataset, PreTokenizedConfig
config = PreTokenizedConfig(data_dir='data/fineweb_edu_10bt', max_seq_length=512)
dataset = PreTokenizedDataset(config)
for i, batch in enumerate(dataset):
    print(f'Sample {i}: input_ids shape = {batch[\"input_ids\"].shape}')
    if i >= 2: break
"

# Step 4: Test MLM collator
python -c "
from embedding_trainer.data.collators.mlm import MLMCollator, MLMCollatorConfig
import torch
config = MLMCollatorConfig()
collator = MLMCollator(config)
samples = [{'input_ids': torch.randint(0, 30000, (512,))} for _ in range(4)]
batch = collator(samples)
print(f'Batch keys: {list(batch.keys())}')
print(f'Input shape: {batch[\"input_ids\"].shape}')
print(f'Masked tokens: {(batch[\"labels\"] != -100).sum().item()}')
"

# Step 5: Full training test
python scripts/train.py \
    experiment=bert_mlm_bf16 \
    data=pretokenized \
    data.data_dir=data/fineweb_edu_10bt \
    training.max_steps=10
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pre-tokenization | Zero tokenization overhead during training |
| BERT tokenizer | Standard WordPiece with [CLS], [SEP], [MASK] for MLM/ELECTRA |
| Binary shards | Memory-mapped for efficient random access |
| ~100M tokens/shard | Balance between file count and per-file overhead |
| uint16 storage | BERT vocab (30522) fits; 50% storage savings vs int32 |
| Shard-level shuffling | Shuffle shard order per epoch; deterministic |
| Header with token IDs | Self-describing format; collator reads from header |
| Epoch offset sampling | Different sequences each epoch without full shuffle overhead |

---

## Sampling Strategy Comparison (flat mode)

| Strategy | Behavior | Pros | Cons |
|----------|----------|------|------|
| `fixed` | Always start at 0, stride=seq_len | Deterministic, simple | Same sequences every epoch |
| `epoch_offset` | Rotate offset: `(epoch * seq_len / N) % seq_len` | Different sequences each epoch, deterministic | Limited to N variations |
| `random` | Sample random offsets from buffer | Good variety, overlapping sequences | Slightly more compute |
| `shuffle_indices` | Shuffle non-overlapping sequence indices | Full shuffle, no overlap | Memory for index array |

**Default: `epoch_offset`** with `num_offset_phases=4`
- Epoch 0: offset=0, Epoch 1: offset=128, Epoch 2: offset=256, Epoch 3: offset=384 (for seq_len=512)
- Cycles back after 4 epochs, giving 4x data variety with minimal overhead

```
# Example: seq_len=8, num_offset_phases=4, tokens=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

Epoch 0 (offset=0): [0-7], [8-15]
Epoch 1 (offset=2): [2-9], [10-17]  <- different token boundaries
Epoch 2 (offset=4): [4-11], [12-19]
Epoch 3 (offset=6): [6-13], [14-21]
Epoch 4 (offset=0): cycles back
```

---

## Packing Mode Comparison

| Mode | Behavior | Pros | Cons |
|------|----------|------|------|
| `flat` | Fixed-length chunks at regular intervals | Faster, no boundary detection | May split documents mid-sentence |
| `doc_aware` | Pack complete documents, respect [SEP][CLS] | Clean document boundaries | Slightly slower, may have padding waste |

**When to use each:**
- **`flat`**: Default for most experiments; slight document splitting has minimal impact on MLM
- **`doc_aware`**: When training with Next Sentence Prediction (NSP), or when document coherence matters

**`doc_aware` strategy:**
1. Find all `[CLS]` positions (document starts)
2. Pack complete documents until `max_seq_length` reached
3. If single document > `max_seq_length`, split it (keep `[CLS]` at chunk start)
4. Pad short sequences to `max_seq_length`

```
# Example: max_seq_length=10, 3 documents

Tokens: [CLS] A B [SEP] [CLS] C D E [SEP] [CLS] F [SEP]

flat mode:
  Chunk 1: [CLS] A B [SEP] [CLS] C D E [SEP] [CLS]  <- splits doc 3
  Chunk 2: F [SEP] ...

doc_aware mode:
  Chunk 1: [CLS] A B [SEP] [CLS] C D E [SEP] [PAD]  <- complete docs 1+2
  Chunk 2: [CLS] F [SEP] [PAD] [PAD] [PAD] ...      <- complete doc 3
```
