import re
from typing import Callable, List, Union

import torch
from transformers import (  # type: ignore
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


def split_text_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    return [s.strip() for s in sentences if s.strip()]


def sliding_window_entailment(
    premise: str,
    hypothesis: str,
    entailment_model: Union[Callable, torch.nn.Module],
    window_size: int = 3,
    stride: int = 1,
) -> float:
    sentences = split_text_into_sentences(premise)

    if len(sentences) <= window_size:
        return entailment_model(premise, hypothesis)

    max_score = 0.0
    for i in range(0, len(sentences) - window_size + 1, stride):
        window_text = " ".join(sentences[i : i + window_size])
        score = entailment_model(window_text, hypothesis)
        max_score = max(max_score, score)

    return max_score


class T5EntailmentScorer:
    def __init__(
        self,
        model_name: str = "google-t5/t5-small",
        device: str = "mps",
        max_length: int = 256,
        window_size: int = 3,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            device
        )
        self.device = device
        self.max_length = max_length
        self.window_size = window_size

    def __call__(self, premise: str, hypothesis: str) -> float:
        input_text = f"mnli premise: {premise} hypothesis: {hypothesis}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=20,
                return_dict_in_generate=False,
                output_scores=False,
            )

        decoded = (
            self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            .strip()
            .lower()
        )
        if decoded == "entailment":
            return 1.0
        elif decoded == "contradiction":
            return 0.0
        else:
            return 0.5

    def score_sliding_window(self, premise: str, hypothesis: str) -> float:
        return sliding_window_entailment(
            premise=premise,
            hypothesis=hypothesis,
            entailment_model=self,
            window_size=self.window_size,
        )


def main():
    premise = (
        "The Eiffel Tower is one of the most famous landmarks in the world. "
        "It was built in Germany in 1889 as the entrance arch to the World's Fair. "
        "Tourists from around the globe visit it every year. "
        "Paris is also known for its art, food, and history. "
        "The monument stands at over 300 meters tall and offers a great view of the city."
    )

    hypothesis = "The Eiffel Tower is located in France."

    scorer = T5EntailmentScorer()

    score_direct = scorer(premise, hypothesis)
    print(f"Entailment score: {score_direct}")

    score_windowed = scorer.score_sliding_window(premise, hypothesis)
    print(f"Sliding window entailment score: {score_windowed}")


if __name__ == "__main__":
    main()
