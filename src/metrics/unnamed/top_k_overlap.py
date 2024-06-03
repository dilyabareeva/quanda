import torch

from src.metrics.base import Metric


class TopKOverlap(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        top_k: int = 1,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(model, train_dataset, *args, **kwargs)
        self.top_k = top_k
        self.all_top_k_examples = []

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        top_k_indices = torch.topk(explanations, self.top_k).indices
        self.all_top_k_examples.append(top_k_indices)

    def compute(self, *args, **kwargs):
        return len(set(self.all_top_k_examples))

    def reset(self, *args, **kwargs):
        self.all_top_k_examples = []

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.all_top_k_examples = state_dict["all_top_k_examples"]

    def state_dict(self, *args, **kwargs):
        return {"all_top_k_examples": self.all_top_k_examples}
