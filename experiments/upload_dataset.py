import pandas as pd
from datasets import Dataset

HF_DATASET_REPO = "quanda-bench-test/trex-subset"
JSONL_PATH = "trex_subset.jsonl"

df = pd.read_json(JSONL_PATH, lines=True)

dataset = Dataset.from_pandas(df)

dataset.push_to_hub(HF_DATASET_REPO)
