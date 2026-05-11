# Changelog

## 0.5.0 (2026-05-11)

This release reworks `quanda` around ready-to-use benchmark assets and
extends it beyond image classification to text classification and causal
language modeling.

### Features

- **Explainers**
  - New wrappers: Kronfluence, `dattri` explainers
  - Support for text classification and causal language modeling tasks
- **Benchmarks**
  - Fully refactored interface
  - Full MNIST benchmark suite
  - Full CIFAR-10 / ResNet-9 benchmark suite
  - BERT/QNLI benchmarks (mislabeling, mixed datasets, LDS, SGD variants)
  - AwA2 / ResNet-50 benchmark configurations
  - Fact-tracing benchmarks for GPT-2 smal/ T-REx
- **Metrics**
  - Recall-at-s Class/Subclass Detection metrics
- **Tooling**
  - Metadata class for handling transformed dataset metadata
  - YAML-based benchmark configuration

### Fixes & refactors

- Numerous fixes across benchmark training, dataset handling
  (splits, HF cache reuse, shuffling), checkpoint loading, and device
  placement
- Consistent benchmark flags (e.g., `use_prediction`) aligned with originating papers
- Reworked explainer dataset/device handling

### Docs

- Updated README, docs, and tutorials
- Quickstart and integration tests are now used as doc snippets
