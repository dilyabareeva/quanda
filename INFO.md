# Fact Tracing Benchmark with TREx and GPT-2

## 📊 Dataset

We reconstruct the dataset used in [pretraining-tda](https://github.com/PAIR-code/pretraining-tda). The dataset consists of:

- **Prompts:** e.g., `"The capital of France is: "`
- **Answers:** e.g., `"Paris"`
- **Evidence Sentences:** Factual sentences containing the answer.

The original 5.4k sample dataset from the repository is not accessible ([broken link](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts_sample.jsonl)), so we recreated it using the following files:

- [`trex_retrievals_trak.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_retrievals_trak.jsonl)
- [`trex_facts.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts.jsonl)
- 20 TREx sentence shards: https://storage.googleapis.com/tda-resources/2410.17413/public/trex_sentences.jsonl-000[XY]-of-00020

We removed prompts without evidence sentences, resulting in **4,927 samples**.

### 📁 Constructed Datasets

| Dataset | Description |
|--------|-------------|
| [`quanda-bench-test/trex-subset`](https://huggingface.co/datasets/quanda-bench-test/trex-subset) | Full 4,927-sample dataset. |
| [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split) | Train/Val split used for fine-tuning and evaluation. |
| [`quanda-bench-test/trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) | Subset of correctly predicted validation prompts (used for benchmarking). |

Each sample contains:
- `prompt`: The factual prompt.
- `answer`: The primary correct answer.
- `alt_answers`: List of alternative correct answers.
- `evidence_sentences`: Sentences containing the correct answer.

---

## 🧠 Models

We pretrain and fine-tune GPT-2 variants using a forked [NanoGPT](https://github.com/aski02/nanoGPT).

### Pretraining

We trained GPT-2 small (124M) and medium (355M) models under three different data regimes:

1. **TREx-only:** 20M TREx sentences, 99/1 train/val split.
2. **TREx + WikiText-103-v1:** full TREx + train split of WikiText-103-v1; evaluated on a subset of WikiText-103-v1.
3. **TREx + OpenWebText (10%):** full TREx + 10% of OpenWebText; evaluated on a subset of OpenWebText.

Each variant was trained using dedicated `.py` training scripts wrapped in `.sh` shell scripts for execution on a cluster.

**Example script pairs:**
- `train_trex_small.py` / `train_trex_small.sh`
- `train_trex_openwebtext_med.py` / `train_trex_openwebtext_med.sh`
- `train_trex_wikitext_small.py` / `train_trex_wikitext_small.sh`

> **Important:** Before training, download all 20 TREx sentence shards and place them in:
>
> ```
> nanoGPT/data/trex/
> ```
> Files can be found at:
> ```
> https://storage.googleapis.com/tda-resources/2410.17413/public/trex_sentences.jsonl-000[00-19]-of-00020
> ```

---

### Fine-tuning

After pretraining, we fine-tuned each model to generate concise answers in the correct format.

Fine-tuning was performed on the `train` split from [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split). Inference was then run on the corresponding `val` split.

You can fine-tune a pretrained model using the following command inside the [NanoGPT fork](https://github.com/aski02/nanoGPT):

```bash
python finetuning.py \
  --checkpoint_path gpt2-small-trex.pt \
  --output_dir finetuned \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-5 \
  --max_length 64
```

This command trains the model to generate brief answers like `"Paris"` for prompts like `"The capital of France is: "`.

The best-performing model was the one trained on **TREx + OpenWebText**, which achieved **~35% accuracy** on the validation split. Correctly answered samples were saved in the [`trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) set. These can be used in the benchmark.

---

### Inference

**TODO: explain how the `inference.py` script builds the benchmark dataset.**

---

### 📦 Available Model Checkpoints

All models are uploaded under the `quanda-bench-test` namespace:

#### 🔹 GPT-2 Small (124M)
- [`gpt2-small-trex`](https://huggingface.co/quanda-bench-test/gpt2-small-trex)
- [`gpt2-small-trex-wikitext`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-wikitext)
- [`gpt2-small-trex-openwebtext`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext)

**Fine-tuned versions:**
- [`gpt2-small-trex-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-ft)
- [`gpt2-small-trex-wikitext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-wikitext-ft)
- [`gpt2-small-trex-openwebtext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext-ft)

#### 🔹 GPT-2 Medium (355M)
- [`gpt2-med-trex`](https://huggingface.co/quanda-bench-test/gpt2-med-trex)
- [`gpt2-med-trex-wikitext`](https://huggingface.co/quanda-bench-test/gpt2-med-trex-wikitext)
- [`gpt2-med-trex-openwebtext`](https://huggingface.co/quanda-bench-test/gpt2-med-trex-openwebtext)

---

## 🛠️ Benchmark

In order to now create a benchmark, all we need is to select one of the finetuned models as well as the [`trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) dataset.

In `quanda/tests/conftest.py` there is a method `load_fact_tracing_dataset` which generates a fitting dataset from the provided huggingface dataset. It selects a specified number of prompts and combines all their evidence sentences into a large corpus. The TDA method should then rank the entire set of evidence sentences for each prompt. If the evidence sentences which belong to that prompt are among the top proponents, then we will get a high score.
`quanda/tests/benchmarks/downstream_eval/test_mrr.py` already implements such a benchmark but on a very small scale. 

**TODO: Write about larger scale benchmark incl. script.**
