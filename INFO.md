# Fact Tracing Benchmark with T-REx and GPT-2

## 📊 Dataset

We reconstruct the dataset used in [pretraining-tda](https://github.com/PAIR-code/pretraining-tda). The dataset consists of:

- **Prompts:** e.g., `"The capital of France is: "`
- **Answers:** e.g., `"Paris"`
- **Evidence Sentences:** e.g., `"Paris is the capital and largest city of France."`

The original **5.4k** sample dataset from the repository is not accessible ([broken link](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts_sample.jsonl)), so we recreated it using the following files:

- [`trex_retrievals_trak.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_retrievals_trak.jsonl)
- [`trex_facts.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts.jsonl)
- **20** TREx sentence shards: https://storage.googleapis.com/tda-resources/2410.17413/public/trex_sentences.jsonl-000[XY]-of-00020

[`trex_retrievals_trak.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_retrievals_trak.jsonl) contains the prompts and their answers under the keys `inputs_plaintext` and `targets_plaintext` for the **5.4k** samples. However, the evidence sentences are not included here.

This is where we need [`trex_facts.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts.jsonl), which contains all **1.2M** samples in TREx including prompts, answers and evidence sentences. We can now merge [`trex_retrievals_trak.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_retrievals_trak.jsonl) and [`trex_facts.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts.jsonl) on the prompt field. 

The last remaining issue is that in [`trex_facts.jsonl`](https://storage.googleapis.com/tda-resources/2410.17413/public/trex_facts.jsonl) the evidence sentences are not included as text but just as an ID. Therefore, we require one last step: merging our **5.4k** sample dataset with the full **20** T-REx shards which map the IDs to the actual sentences.

Now we recreated the **5.4k** sample dataset from the paper. As a last step, we removed prompts without evidence sentences, resulting in **4,927 samples**. The result is this dataset, which we uploaded to huggingface: [`quanda-bench-test/trex-subset`](https://huggingface.co/datasets/quanda-bench-test/trex-subset).

### 📁 Constructed Datasets

| Dataset | Description |
|--------|-------------|
| [`quanda-bench-test/trex-subset`](https://huggingface.co/datasets/quanda-bench-test/trex-subset) | Full **4,927**-sample dataset. |
| [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split) | Train/Val split used for fine-tuning and evaluation. |
| [`quanda-bench-test/trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) | Subset containing the prompts for which [`gpt2-small-trex-openwebtext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext-ft) predicted the answer correctly. |

Each sample contains:
- `prompt`: The factual prompt.
- `answer`: The primary correct answer.
- `alt_answers`: List of alternative correct answers.
- `evidence_sentences`: Sentences containing the correct answer.
- `prediction`: The model's prediction for the prompt (only in `trex-subset-benchmark`).

---

## 🧠 Models

We pretrain and fine-tune GPT-2 variants using a forked [NanoGPT](https://github.com/aski02/nanoGPT).

### Pretraining

We trained GPT-2 small (**124M**) and medium (**355M**) models under three different data regimes:

1. **T-REx only:** **20M** T-REx sentences, **99/1** train/val split.
2. **T-REx + WikiText-103-v1:** full T-REx + train split of WikiText-103-v1; evaluated on a subset of WikiText-103-v1.
3. **T-REx + OpenWebText (10%):** full T-REx + **10%** of OpenWebText; evaluated on a subset of OpenWebText.

Each variant was trained using dedicated `.py` training scripts wrapped in `.sh` shell scripts for execution on a cluster. There is also an [`apptainer.def`](https://github.com/aski02/nanoGPT/blob/master/apptainer.def) file available.

**Example script pairs:**
- [`train_trex_small.py`](https://github.com/aski02/nanoGPT/blob/master/train_trex_small.py) / [`train_trex_small.sh`](https://github.com/aski02/nanoGPT/blob/master/train_trex_small.sh)
- [`train_trex_openwebtext_med.py`](https://github.com/aski02/nanoGPT/blob/master/train_trex_openwebtext_med.py) / [`train_trex_openwebtext_med.sh`](https://github.com/aski02/nanoGPT/blob/master/train_trex_openwebtext_med.sh)
- [`train_trex_wikitext_small.py`](https://github.com/aski02/nanoGPT/blob/master/train_trex_wikitext_small.py) / [`train_trex_wikitext_small.sh`](https://github.com/aski02/nanoGPT/blob/master/train_trex_wikitext_small.sh)

> **Important:** Before training, download all **20** T-REx sentence shards and place them in:
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

After pretraining, we fine-tuned each model to generate concise answers in the correct format. During fine-tuning, we masked out the tokens relating to the prompt (setting their label to `-100`) such that the loss is only calculated on the answer tokens. This ensures the model focuses on learning to generate precise answers given a prompt, rather than memorizing the prompts themselves.

Here is an example with a tokenized text sequence of length **12** which was padded to length **16** (we use `50256` as the padding token). All label-tokens are set to `-100` except for the answer-tokens:
```python
plain_text = "The Fugitive originated in the following country: United States of America"

inputs_ids = [464, 47832, 1800, 20973, 287, 262, 1708, 1499, 25, 1578, 1829, 286, 2253, 50256, 50256, 50256, 50256]

labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, 1578, 1829, 286, 2253, -100, -100, -100, -100]
```

Fine-tuning was performed on the `train` split from [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split). This partition was further split into a train and val set for monitoring training progress.

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

This command trains the model to generate brief answers like `"Paris"` for prompts like `"The capital of France is: "` using the [`finetuning.py`](https://github.com/aski02/nanoGPT/blob/master/finetuning.py) script. The fine-tuned model will be stored inside the specified `output_dir`.


---

### Inference

Once a model was fine-tuned, we can use it to predict answers for the T-REx prompts. The [`inference.py`](https://github.com/aski02/nanoGPT/blob/master/inference.py) script inside the [NanoGPT fork](https://github.com/aski02/nanoGPT) will run through the validation set in [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split) and save all the correctly predicted prompts to the specified output file:

```bash
python inference.py \
  --model_dir gpt2-small-trex \
  --output_path trex-subset-benchmark.jsonl \
  --num_prompts 50
```

For fact tracing benchmarks we oftentimes want to evaluate wether the evidence sentence (i.e. the sentence containing the answer to the prompt) is inside the top proponents for a given prompt. Therefore it makes sense to only evaluate prompts for which the model knows the correct answer.

The best-performing model was the one trained on **T-REx + OpenWebText**, which achieved **~33%** accuracy on the validation split from [`quanda-bench-test/trex-subset-split`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-split). Correctly answered samples were saved in the [`trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) set. These can be used in the benchmark.

---

### 📦 Available Model Checkpoints

All models are uploaded under the [`quanda-bench-test`](https://huggingface.co/quanda-bench-test) namespace:

#### 🔹 GPT-2 Small (**124M**)
- [`gpt2-small-trex`](https://huggingface.co/quanda-bench-test/gpt2-small-trex)
- [`gpt2-small-trex-wikitext`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-wikitext)
- [`gpt2-small-trex-openwebtext`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext)

**Fine-tuned versions:**
- [`gpt2-small-trex-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-ft)
- [`gpt2-small-trex-wikitext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-wikitext-ft)
- [`gpt2-small-trex-openwebtext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext-ft)

There are also trained models for **GPT-2 Medium (**355M**)**, however they are quite large (**4.5 GB** each) and I got rate-limited by huggingface.

---

## 🛠️ Benchmark

In order to now create a benchmark, all we need is to select one of the fine-tuned models as well as the [`trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) dataset.

In `quanda/tests/conftest.py` there is a method `load_fact_tracing_dataset` which generates a fitting dataset from the provided huggingface dataset. It selects a specified number of prompts and combines all their evidence sentences into a large corpus. The TDA method should then rank the entire set of evidence sentences for each prompt. If the evidence sentences which belong to that prompt are among the top proponents, then we will get a high score. 
The final test inside `quanda/tests/benchmarks/downstream_eval/test_mrr.py` already implements such a benchmark using the fine-tuned [`gpt2-small-trex-openwebtext-ft`](https://huggingface.co/quanda-bench-test/gpt2-small-trex-openwebtext-ft) as well as the [`trex-subset-benchmark`](https://huggingface.co/datasets/quanda-bench-test/trex-subset-benchmark) dataset for the MRR metric. The scores however are quite low when increasing the number of prompts.

> **Note:** Kronfluence expects a model's forward pass to return full sequence logits of shape `[B, T, V]`.  
> The original NanoGPT implementation by Karpathy only returns a single-token prediction during inference (`[B, 1, V]`) when `targets=None` for efficiency reasons.
> For compatibility, we modified the forward pass of the GPT model to always return full logits:
>
> ```python
> logits = self.lm_head(x)  # instead of logits = self.lm_head(x[:, [-1], :])
> ```
>
> We did this change only in the `quanda/tests/models.py` file. In the [training repository](https://github.com/aski02/nanoGPT) we still use the optimized code.

