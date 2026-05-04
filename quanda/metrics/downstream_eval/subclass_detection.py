"""Subclass Detection Metric."""

from typing import List, Optional, Union

import torch

from quanda.metrics.downstream_eval import ClassDetectionMetric
from quanda.utils.common import CheckpointLoadFunc, chunked_logits, ds_len


class SubclassDetectionMetric(ClassDetectionMetric):
    """Subclass Detection Metric.

    A model is trained on a dataset where labels are grouped into superclasses.
    For each test sample, the fraction of training points that share the test
    sample's (ungrouped) subclass among the ``s`` most influential training
    points (top-``s`` highest attribution scores). The metric returns the
    average of this fraction across test samples. With ``s=1`` this reduces to
    the original subclass detection accuracy of Hanawa et al. (2021).

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        train_subclass_labels: torch.Tensor,
        s: int = 5,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[CheckpointLoadFunc] = None,
        filter_by_prediction: bool = False,
        inference_batch_size: Optional[int] = None,
    ):
        """Initialize the Subclass Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        train_subclass_labels : torch.Tensor
            The subclass labels of the training dataset.
        s : int, optional
            Number of top-attributed training points to consider when
            computing the same-subclass fraction, by default 5.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[CheckpointLoadFunc], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the correct superclass is predicted, by
            default False.
        inference_batch_size : Optional[int], optional
            If set, split the model forward used to filter by prediction into
            sub-batches of this size. ``None`` disables chunking.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            s=s,
            checkpoints_load_func=checkpoints_load_func,
            inference_batch_size=inference_batch_size,
        )

        if len(train_subclass_labels) != ds_len(self.train_dataset):
            raise ValueError(
                f"Number of subclass labels ({len(train_subclass_labels)}) "
                f"does not match the number of train dataset samples "
                f"({ds_len(self.train_dataset)})."
            )
        self.subclass_labels = train_subclass_labels
        self.filter_by_prediction = filter_by_prediction

    def update(
        self,
        explanations: torch.Tensor,
        test_targets: Union[List[int], torch.Tensor],
        test_data: Optional[torch.Tensor] = None,
        test_superclass_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        test_targets : Union[List[int], torch.Tensor]
            Original **not-grouped, subclass** labels of the test samples.
        test_data: Optional[torch.Tensor]
            Test samples to used to generate the explanations.
            Only required if `filter_by_prediction` is True during
            initalization.
        test_superclass_targets: Optional[torch.Tensor]
            The true superclasses of the test samples. Only required
            if `filter_by_prediction` is True during initalization.
        **kwargs
             Additional keyword arguments.

        Raises
        ------
        ValueError
            If `test_data` and `test_superclass_targets` are not
            provided when `filter_by_prediction` is True.

        """
        explanations = explanations.to(self.device)

        if isinstance(test_targets, list):
            test_targets = torch.tensor(test_targets)
        test_targets = test_targets.to(self.device)

        select_idx = torch.tensor([True] * len(explanations)).to(self.device)
        if self.filter_by_prediction:
            if test_data is None or test_superclass_targets is None:
                raise ValueError(
                    "test_data and test_superclass_targets must be "
                    "provided if filter_by_prediction is True"
                )
            if isinstance(test_superclass_targets, list):
                test_superclass_targets = torch.tensor(test_superclass_targets)
            test_superclass_targets = test_superclass_targets.to(self.device)
            model_device = next(self.model.parameters()).device
            test_data = test_data.to(model_device)
            logits = chunked_logits(
                self.model, test_data, self.inference_batch_size
            )
            pred_cls = logits.argmax(dim=1).to(self.device)
            select_idx *= pred_cls == test_superclass_targets

        explanations = explanations[select_idx]
        test_targets = test_targets[select_idx].to(self.device)

        k = min(self.s, explanations.shape[1])
        _, top_xpl_indices = explanations.topk(k=k, dim=1)
        top_xpl_targets = self.subclass_labels.to(self.device)[top_xpl_indices]
        scores = (
            (top_xpl_targets == test_targets.unsqueeze(1)).float().mean(dim=1)
        )
        self.scores.append(scores)
