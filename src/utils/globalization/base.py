import torch


class Globalization:
    def __init__(self, training_dataset: torch.utils.data.Dataset, *args, **kwargs):
        self.dataset = training_dataset
        self.scores = torch.zeros((len(training_dataset)))
        raise NotImplementedError

    def get_global_ranking(self):
        return self.scores.argmax()
