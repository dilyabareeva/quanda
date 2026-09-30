"""Random explainer module."""

from typing import Any, List, Optional, Union

import torch

from quanda.explainers import Explainer
from quanda.utils.common import CheckpointLoadFunc, cache_result, ds_len
from quanda.utils.tasks import TaskLiterals


class RandomExplainer(Explainer):
    """The most basic version of a random explainer.

    The explanations are generated with independent values sampled from a
    uniform distribution in [0,1].
    """

    accepted_tasks: List[TaskLiterals] = ["image_classification"]

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[CheckpointLoadFunc] = None,
        seed: int = 27,
        device: Optional[str] = None,
    ):
        """Initialize RandomExplainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model for attributions. It is called as ``model(inputs)`` for
            the ``image_classification`` task, where ``inputs`` is a batched
            input tensor, and as ``model(**inputs)`` for the
            ``text_classification`` and ``causal_lm`` tasks, where ``inputs``
            is a dictionary of tokenized inputs following the Hugging Face
            convention: ``input_ids``, ``attention_mask`` and, optionally,
            ``token_type_ids``, each of shape ``(batch, seq_len)``. Targets
            are under the ``labels`` key, and are not passed
            to the model. The forward pass is expected to return raw,
            unnormalized logits, either as a tensor or as an object exposing
            a ``.logits`` attribute, as returned by Hugging Face models, of
            shape ``(batch, n_classes)`` for the classification tasks and
            ``(batch, seq_len, vocab_size)`` for ``causal_lm``.
        train_dataset : torch.utils.data.Dataset
            Training dataset that was used to train the model.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[CheckpointLoadFunc], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        seed : int, optional
            Seed for random number generator, by default 27.
        device : Optional[str], optional
            Device to run the computation on. If None, inferred from the
            model's parameters. Defaults to None.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )

        if device is not None:
            self.device = device

        self.seed = seed
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.seed)

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Optional[Union[List[int], torch.Tensor]] = None,
    ):
        """Return random explanations.

        Random explainer does not explain anything, just returns random values.

        Parameters
        ----------
        test_data : torch.Tensor
            Test points for the model decisions to be explained. Is not used
            or the `RandomExplainer`.
        targets : Optional[Union[List[int], torch.Tensor]] = None
            The model outputs to be explained.
            Some methods do not need this. Defaults to None. Is not used in
            `RandomExplainer`.

        Returns
        -------
        torch.Tensor
            Random tensor of shape `(test.shape[0],train_dataset_length)`

        """
        n = (
            next(iter(test_data.values())).size(0)
            if isinstance(test_data, dict)
            else test_data.size(0)
        )
        return torch.rand(
            n,
            ds_len(self.train_dataset),
            generator=self.generator,
            device=self.device,
        )

    @cache_result
    def self_influence(
        self, batch_size: int = 32, **kwargs: Any
    ) -> torch.Tensor:
        """Random self-influence.

        Random self-influence is just a vector of random values of the length
        of the training dataset.

        Parameters
        ----------
        batch_size : int
            `RandomExplainer` does not use this.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Random tensor of shape `(train dataset length,)`

        """
        return torch.rand(
            ds_len(self.train_dataset),
            generator=self.generator,
            device=self.device,
        )
