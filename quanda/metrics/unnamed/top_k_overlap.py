import torch

from quanda.metrics.base import Metric


class TopKOverlapMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        top_k: int = 1,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset, device=device)
        self.top_k = top_k
        self.all_top_k_examples = torch.empty(0, top_k).to(device)

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        explanations = explanations.to(self.device)

        top_k_indices = torch.topk(explanations, self.top_k).indices
        self.all_top_k_examples = torch.concat((self.all_top_k_examples, top_k_indices), dim=0)

    def compute(self, *args, **kwargs):
        return len(torch.unique(self.all_top_k_examples))

    def reset(self, *args, **kwargs):
        self.all_top_k_examples = torch.empty(0, self.top_k)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.all_top_k_examples = state_dict["all_top_k_examples"]

    def state_dict(self, *args, **kwargs):
        return {"all_top_k_examples": self.all_top_k_examples}
