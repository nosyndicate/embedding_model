# Experimental Embedding Model Training Framework

## Overview

Create an extensible Python framework for experimenting with various training methods, algorithms, and low-precision techniques for language embedding models (BERT, ModernBERT).

**Priority Choices:**
- First model: **BERT** (classic baseline, well-documented)
- Precision modes: **BF16 + FP32** (BF16 for fast training, FP32 as baseline)
- Experiment tracking: **Weights & Biases** (best visualization, easy comparison)
- Pretraining tasks: **MLM, ELECTRA, Span Corruption**
- Dataset: **FineWeb-Edu sample-100BT** (pre-tokenized with BERT tokenizer)

## Key Design Principles

1. **Plugin/Registry Architecture** - Add new components (models, optimizers, tasks) without modifying core code
2. **Configuration-Driven** - Use Hydra/YAML for reproducible experiments
3. **Precision Abstraction** - Swap between FP32/FP16/BF16/INT8 via config
4. **Callback System** - Hook into training at any point for custom logic

---

## Directory Structure

```
embedding_model/
├── pyproject.toml                    # Dependencies (uv/poetry)
├── configs/                          # Hydra YAML configs
│   ├── config.yaml                   # Main entry point
│   ├── experiment/                   # Experiment-specific configs
│   │   ├── bert_mlm_fp32.yaml
│   │   ├── bert_mlm_bf16.yaml
│   │   ├── bert_electra_bf16.yaml
│   │   └── bert_span_corruption_bf16.yaml
│   ├── model/                        # Model architecture configs
│   │   ├── bert_base.yaml
│   │   └── bert_large.yaml
│   ├── optimizer/                    # Optimizer configs
│   │   ├── adamw.yaml
│   │   ├── lion.yaml
│   │   └── sophia.yaml
│   ├── scheduler/                    # LR scheduler configs
│   │   ├── cosine.yaml
│   │   └── warmup_cosine_decay.yaml
│   ├── precision/                    # FP32/BF16 configs
│   │   ├── fp32.yaml
│   │   └── bf16.yaml
│   ├── task/                         # Pretraining task configs
│   │   ├── mlm.yaml
│   │   ├── electra.yaml
│   │   └── span_corruption.yaml
│   ├── logging/                      # Experiment tracking configs
│   │   └── wandb.yaml
│   ├── data/                         # Data configs
│   │   ├── pretokenized.yaml
│   │   ├── fineweb_edu_10bt.yaml
│   │   ├── fineweb_edu_100bt.yaml
│   │   └── fineweb_edu_350bt.yaml
│   ├── tokenizer/                    # Tokenizer configs
│   │   ├── bert_base.yaml
│   │   └── bert_large.yaml
│   ├── collator/                     # Collator configs
│   │   ├── mlm.yaml
│   │   ├── mlm_wwm.yaml
│   │   ├── electra.yaml
│   │   └── span_corruption.yaml
│   └── distributed/                  # Distributed strategy configs
│       ├── single_gpu.yaml
│       ├── ddp.yaml
│       └── tensor_parallel.yaml
│
├── src/embedding_trainer/            # Main package
│   ├── __init__.py
│   ├── __main__.py                   # CLI entry point
│   │
│   ├── core/                         # Base abstractions & protocols
│   │   ├── __init__.py
│   │   ├── types.py                  # Protocol definitions
│   │   ├── base_model.py             # Abstract embedding model
│   │   ├── base_trainer.py           # Abstract trainer
│   │   ├── base_task.py              # Abstract pretraining task
│   │   ├── base_callback.py          # Callback protocol
│   │   └── precision.py              # Precision context abstraction
│   │
│   ├── models/                       # Model implementations
│   │   ├── __init__.py
│   │   ├── registry.py               # Model registry
│   │   └── bert/                     # BERT implementation
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── config.py
│   │
│   ├── training/                     # Training loop & strategies
│   │   ├── __init__.py
│   │   ├── trainer.py                # Main trainer
│   │   ├── electra_trainer.py        # ELECTRA-specific trainer (gen+disc)
│   │   └── checkpointing.py          # Save/resume
│   │
│   ├── optimizers/                   # Optimizer implementations
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── adamw.py
│   │
│   ├── schedulers/                   # LR schedulers
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── cosine.py
│   │
│   ├── precision/                    # Precision modes
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── fp32.py
│   │   └── bf16.py
│   │
│   ├── tasks/                        # Pretraining tasks
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── mlm.py                    # Masked Language Modeling
│   │   ├── electra.py                # ELECTRA (replaced token detection)
│   │   └── span_corruption.py        # T5-style span corruption
│   │
│   ├── callbacks/                    # Training callbacks
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── checkpoint.py
│   │   └── gradient_monitor.py
│   │
│   ├── logging/                      # Experiment tracking
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── base.py
│   │   └── wandb_logger.py
│   │
│   ├── data/                         # Data pipeline (see DATA_PIPELINE.md)
│   │   ├── __init__.py
│   │   ├── registry.py               # DATASET_REGISTRY, COLLATOR_REGISTRY
│   │   ├── base.py                   # Protocols: DatasetProtocol, CollatorProtocol
│   │   ├── loader.py                 # DataLoader factory
│   │   ├── datasets/
│   │   │   ├── __init__.py
│   │   │   └── pretokenized.py       # PreTokenizedDataset (loads .bin shards)
│   │   └── collators/
│   │       ├── __init__.py
│   │       ├── base.py               # BaseCollator
│   │       ├── mlm.py                # MLMCollator
│   │       ├── electra.py            # ELECTRACollator
│   │       └── span_corruption.py    # SpanCorruptionCollator
│   │
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── schema.py                 # Dataclass schemas
│   │   └── resolvers.py              # Custom Hydra resolvers
│   │
│   ├── distributed/                  # Multi-GPU strategies
│   │   ├── __init__.py
│   │   ├── registry.py               # DISTRIBUTED_REGISTRY
│   │   ├── base.py                   # DistributedStrategy protocol
│   │   ├── single_gpu.py             # Default no-op strategy
│   │   ├── ddp.py                    # Basic DDP wrapper
│   │   ├── tensor_parallel.py        # Parameter banks + comm overlap
│   │   └── communication.py          # Async reduce-scatter/all-reduce helpers
│   │
│   └── utils/                        # Utilities
│       ├── __init__.py
│       ├── registry.py               # Generic registry class
│       ├── seed.py                   # Reproducibility
│       └── device.py                 # Device management
│
├── scripts/
│   ├── tokenize_fineweb.py           # Pre-tokenization script (BERT tokenizer)
│   ├── train.py                      # Main training script
│   ├── train_hf_baseline.py          # HuggingFace reference trainer (correctness oracle)
│   └── evaluate.py
│
├── data/                             # Pre-tokenized data shards (gitignored)
│   └── fineweb_edu_100bt/            # Default: 100B token sample
│       ├── fineweb_train_000000.bin
│       └── ...
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_precision.py
│   │   └── test_tasks.py
│   └── integration/
│       ├── __init__.py
│       └── test_training_loop.py
│
└── notebooks/                        # Exploration notebooks
    └── .gitkeep
```

---

## Implementation Order

**Priority Choices:**
- First model: **BERT** (classic baseline, well-documented)
- Precision modes: **BF16 + FP32** (BF16 for fast training, FP32 as baseline)
- Experiment tracking: **Weights & Biases** (best visualization)
- Tasks: **MLM, ELECTRA, Span Corruption**

### Phase 1: Foundation
1. Set up `pyproject.toml` with dependencies
2. Create `src/embedding_trainer/utils/registry.py` - Generic registry class
3. Create `src/embedding_trainer/core/types.py` - Protocol definitions
4. Create `src/embedding_trainer/utils/seed.py` - Reproducibility utilities

### Phase 2: Core Abstractions
5. Create `src/embedding_trainer/core/base_model.py`
6. Create `src/embedding_trainer/core/precision.py` - Precision context
7. Create `src/embedding_trainer/core/base_task.py`
8. Create `src/embedding_trainer/core/base_trainer.py`

### Phase 2.5: Data Pipeline (see DATA_PIPELINE.md)
- Create `scripts/tokenize_fineweb.py` - Pre-tokenization script
- Create `src/embedding_trainer/data/` - Data loading and collators
- Pre-tokenize FineWeb-Edu with BERT tokenizer to .bin shards
- Dataset: **FineWeb-Edu sample-100BT** (100 billion tokens)
- Tokenizer: **BERT** (bert-base-uncased, WordPiece)

### Phase 2.6: Reference Baseline (Correctness Oracle)
- Create `scripts/train_hf_baseline.py` - HuggingFace Trainer baseline
- Uses same pre-tokenized data as custom trainer
- Validates: masking logic, loss computation, metrics, data loading
- Serves as correctness oracle for custom implementation
- **Purpose**: Prevents spending weeks debugging bugs that manifest as "my new low-precision method is unstable"

### Phase 3: Initial Implementations
9. Create `src/embedding_trainer/models/bert/` - BERT model
10. Create `src/embedding_trainer/precision/fp32.py` - FP32 baseline
11. Create `src/embedding_trainer/precision/bf16.py` - BF16 for fast training
12. Create `src/embedding_trainer/tasks/mlm.py` - MLM task
13. Create `src/embedding_trainer/optimizers/adamw.py`

### Phase 4: Training Loop
14. Create `src/embedding_trainer/training/trainer.py`
15. Create `src/embedding_trainer/training/checkpointing.py`
16. Create `src/embedding_trainer/callbacks/` - Basic callbacks

### Phase 5: Experiment Tracking
17. Create `src/embedding_trainer/logging/wandb_logger.py` - W&B integration
18. Create `src/embedding_trainer/logging/base.py` - Logger protocol

### Phase 6: Configuration
19. Set up Hydra configs in `configs/`
20. Create `src/embedding_trainer/config/schema.py`
21. Create `scripts/train.py` - Main entry point

### Phase 7: Extensions (Optimizers + Distributed)
22. Add more optimizers (Lion, Sophia, Schedule-Free)
23. Add ModernBERT model
24. Add FP16 with gradient scaling (if needed)
25. Create `src/embedding_trainer/distributed/` - Distributed strategies
26. Implement Parameter Banks model variant (`bert_banked`)
27. Implement Communication Overlap (async reduce-scatter/all-reduce)
28. Add `configs/distributed/` - single_gpu, ddp, tensor_parallel configs

### Phase 8: Additional Tasks
29. Create `src/embedding_trainer/tasks/electra.py` - ELECTRA task
30. Create `src/embedding_trainer/training/electra_trainer.py` - ELECTRA trainer
31. Create `src/embedding_trainer/tasks/span_corruption.py` - Span corruption task

### Phase 9: Testing & Documentation
32. Create unit tests in `tests/unit/`
33. Create integration tests in `tests/integration/`
34. Update README with usage examples

---

## Verification Plan

### Unit Tests
```bash
pytest tests/unit/ -v
```

### Baseline Parity Tests (Correctness Validation)
```bash
# Run HF baseline for N steps, log loss/metrics
python scripts/train_hf_baseline.py \
    --data_dir ./data/fineweb_edu_100bt \
    --max_steps 100 \
    --output_dir ./baseline_results

# Run custom trainer with identical config
python scripts/train.py \
    experiment=bert_mlm_bf16 \
    training.max_steps=100

# Compare:
# - Loss curves should match within tolerance
# - MLM accuracy should match
# - Gradient norms should be similar
```

**What to Validate:**
1. **Masking logic parity** - Same tokens masked, same mask ratio
2. **Loss parity** - Loss values match within numerical tolerance (< 1%)
3. **Metric parity** - MLM accuracy, perplexity match
4. **Data parity** - Same batches in same order (seeded)

### Integration Test (End-to-End Training)
```bash
# MLM training
python scripts/train.py \
    experiment=bert_mlm_bf16 \
    model.hidden_size=64 \
    model.num_hidden_layers=2 \
    training.max_steps=10

# ELECTRA training
python scripts/train.py \
    experiment=bert_electra_bf16 \
    training.max_steps=10

# Span corruption training
python scripts/train.py \
    experiment=bert_span_corruption_bf16 \
    training.max_steps=10
```

### Test Different Precision Modes
```bash
python scripts/train.py precision=fp32 training.max_steps=5
python scripts/train.py precision=bf16 training.max_steps=5
```

### Test Checkpoint Resume
```bash
python scripts/train.py training.max_steps=10 training.save_steps=5
python scripts/train.py resume_from=./experiments/checkpoint_step_5.pt
```

### Test Multi-GPU (Distributed)
```bash
# Test DDP (basic multi-GPU)
torchrun --nproc_per_node=2 scripts/train.py \
    experiment=bert_mlm_bf16 \
    distributed=ddp \
    training.max_steps=10

# Test tensor parallel with parameter banks
torchrun --nproc_per_node=4 scripts/train.py \
    experiment=bert_mlm_bf16 \
    distributed=tensor_parallel \
    model=bert_banked \
    training.max_steps=10
```

---

## Example Usage (After Implementation)

```bash
# Basic MLM training
python scripts/train.py experiment=bert_mlm_bf16

# ELECTRA training
python scripts/train.py experiment=bert_electra_bf16

# Span corruption (T5-style)
python scripts/train.py experiment=bert_span_corruption_bf16

# Override task settings
python scripts/train.py \
    experiment=bert_mlm_bf16 \
    task.mlm_probability=0.20

# ELECTRA with custom generator size
python scripts/train.py \
    experiment=bert_electra_bf16 \
    task.generator_size_factor=0.33 \
    task.disc_weight=100.0
```

### Example Hydra Config

```yaml
# configs/experiment/bert_electra_bf16.yaml
defaults:
  - /model: bert_base
  - /optimizer: adamw
  - /precision: bf16
  - /task: electra
  - /data: pretokenized
  - /collator: electra
  - /logging: wandb

experiment_name: bert_electra_bf16
training:
  max_epochs: 3
  gradient_accumulation_steps: 4
data:
  data_dir: "${hydra:runtime.cwd}/data/fineweb_edu_100bt"
task:
  generator_size_factor: 0.25
  disc_weight: 50.0
```

---

## How to Add New Components

### Adding a New Optimizer
1. Create `src/embedding_trainer/optimizers/new_opt.py`
2. Use `@OPTIMIZER_REGISTRY.register("name")` decorator
3. Create `configs/optimizer/new_opt.yaml`

### Adding a New Model
1. Create `src/embedding_trainer/models/new_model/`
2. Extend `BaseEmbeddingModel`
3. Register with `@MODEL_REGISTRY.register("name")`

### Adding a New Task
1. Create `src/embedding_trainer/tasks/new_task.py`
2. Extend `BaseTask`
3. Register with `@TASK_REGISTRY.register("name")`
4. Create `configs/task/new_task.yaml`
