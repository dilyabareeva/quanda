"""Wrappers for the Captum influence computation methods."""

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import lightning as L
import torch
from captum._utils.av import AV  # type: ignore
from captum.influence import (  # type: ignore
    SimilarityInfluence,
    TracInCP,
    TracInCPFast,
    TracInCPFastRandProj,
)

# TODO Should be imported directly from captum.influence once available
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)
from captum.influence._utils.nearest_neighbors import (  # type: ignore
    NearestNeighbors,
)

from quanda.explainers.base import Explainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from quanda.utils.common import (
    default_tensor_type,
    ds_len,
    map_location_context,
    process_targets,
)
from quanda.utils.datasets import OnDeviceDataset
from quanda.utils.functions import cosine_similarity
from quanda.utils.tasks import TaskLiterals

logger = logging.getLogger(__name__)


class CaptumInfluence(Explainer, ABC):
    """Base class for the Captum explainers."""

    accepted_tasks: List[TaskLiterals] = ["image_classification"]

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        explain_kwargs: Any,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the base `CaptumInfluence` wrapper.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        explainer_cls : type
            The class of the explainer from Captum.
        explain_kwargs : Any
            Additional keyword arguments for the explainer.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification", "text_classification",
            "causal_lm".
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.explainer_cls = explainer_cls
        self.explain_kwargs = explain_kwargs

    def init_explainer(self, **explain_kwargs: Any):
        """Initialize the Captum explainer.

        Parameters
        ----------
        **explain_kwargs : Any
            Additional keyword arguments to be passed to the explainer.

        """
        self.captum_explainer = self.explainer_cls(**explain_kwargs)

    @abstractmethod
    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """Abstract method for computing influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        raise NotImplementedError


class CaptumSimilarity(CaptumInfluence):
    # TODO: incorporate SimilarityInfluence kwargs into init_kwargs
    """Class for Similarity Influence wrapper.

    This explainer uses a similarity function on its inputs to rank the
    training data.

    Notes
    -----
    The user is referred to captum's codebase [1] for details on the specifics
    of the parameters.

    References
    ----------
    1) https://captum.ai/api/influence.html#similarityinfluence

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        layers: Union[str, List[str]],
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        cache_dir: str = "./cache",
        similarity_metric: Callable = cosine_similarity,
        similarity_direction: str = "max",
        batch_size: int = 1,
        replace_nan: int = 0,
        load_from_disk: bool = True,
        **explainer_kwargs: Any,
    ):
        """Initialize the `CaptumSimilarity` explainer.

        The Captum implementation includes a bug in the batch processing of the
        dataset, which leads to an error if the dataset size is not divisible
        by the batch size. To circumvent this issue, we divide the dataset into
        two subsets and process them separately.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Checkpoints for the model.
        model_id : str
            Identifier for the model.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        layers : Union[str, List[str]]
            Layers of the model for which the activations are computed.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification".
        checkpoints_load_func : Optional[Callable], optional
            Function to load checkpoints. If None, a default function is used.
            Defaults to None.
        cache_dir : str, optional
            Directory for caching activations. Defaults to "./cache".
        similarity_metric : Callable, optional
            Metric for computing similarity. Defaults to `cosine_similarity`.
        similarity_direction : str, optional
            Direction for similarity computation. Can be either "min" or "max".
            Defaults to "max".
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        replace_nan : int, optional
            The value to replace NaN values in similarity scores with. Defaults
            to 0.
        load_from_disk : bool, optional
            If True, activations will be loaded from disk if available, instead
            of being recomputed. Defaults to True.
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.

        """
        logger.info("Initializing Captum SimilarityInfluence explainer...")

        # extract and validate layer from kwargs
        self._layer: Optional[Union[List[str], str]] = None
        self.layer = layers

        model_id += "_main"

        super().__init__(
            model=model,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            train_dataset=train_dataset,
            task=task,
            explainer_cls=SimilarityInfluence,
            explain_kwargs=explainer_kwargs,
        )

        self.model_id = model_id
        self.cache_dir = cache_dir

        self.modulo_batch_size = ds_len(train_dataset) % batch_size
        # divide train_dataset into two subsets to make up for a batching bug
        self.train_set_1 = torch.utils.data.Subset(
            self.train_dataset,
            [i for i in range(ds_len(train_dataset) - self.modulo_batch_size)],
        )

        explainer_kwargs.update(
            {
                "module": model,
                "influence_src_dataset": self.train_set_1,
                "activation_dir": cache_dir,
                "model_id": model_id,
                "layers": self.layer,
                "similarity_direction": similarity_direction,
                "similarity_metric": similarity_metric,
                "batch_size": batch_size,
                "replace_nan": replace_nan,
                **explainer_kwargs,
            }
        )

        # As opposed to the original implementation, we move the activation
        # generation to the init method.
        AV.generate_dataset_activations(
            self.cache_dir,
            self.model,
            self.model_id,
            self.layer,
            torch.utils.data.DataLoader(
                self.train_set_1, batch_size, shuffle=False
            ),
            identifier="src",
            load_from_disk=load_from_disk,
            return_activations=True,
        )

        self.captum_explainer_1 = self.explainer_cls(**explainer_kwargs)

        self.train_set_2: Optional[torch.utils.data.Subset] = None

        if self.modulo_batch_size > 0:
            self.train_set_2 = torch.utils.data.Subset(
                self.train_dataset,
                [
                    i
                    for i in range(
                        ds_len(train_dataset) - self.modulo_batch_size,
                        ds_len(train_dataset),
                    )
                ],
            )
            explainer_kwargs_2 = explainer_kwargs.copy()
            explainer_kwargs_2["influence_src_dataset"] = self.train_set_2
            explainer_kwargs_2["batch_size"] = ds_len(self.train_set_2)
            explainer_kwargs_2["model_id"] = model_id + "_suppl_act"

            AV.generate_dataset_activations(
                self.cache_dir,
                self.model,
                self.model_id + "_suppl_act",
                self.layer,
                torch.utils.data.DataLoader(
                    self.train_set_2, ds_len(self.train_set_2), shuffle=False
                ),
                identifier="src",
                load_from_disk=load_from_disk,
                return_activations=True,
            )

            self.captum_explainer_2 = self.explainer_cls(**explainer_kwargs_2)

        # explicitly specifying explain method kwargs as instance attributes

        if "top_k" in explainer_kwargs:
            warnings.warn(
                "top_k is not supported by CaptumSimilarity explainer. "
                "Ignoring the argument."
            )

    @property
    def layer(self):
        """Return the layer for which the activations are computed."""
        return self._layer

    @layer.setter
    def layer(self, layers: Any):
        """Set layer value.

        Our wrapper only allows a single layer to be passed, while the
        Captum implementation allows multiple layers. Here, we validate if
        only a single layer was passed.
        """
        if isinstance(layers, str):
            self._layer = layers
            return
        if len(layers) != 1:
            raise ValueError(
                "A single layer shall be passed to the CaptumSimilarity "
                "explainer."
            )
        self._layer = layers[0]

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor] = torch.tensor(0),
    ):
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor], optional
            Labels for the test samples. This argument is ignored.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        test_data = test_data.to(self.device)

        with (
            map_location_context(self.device),
            default_tensor_type(self.device),
        ):
            topk_idx_1, topk_val_1 = self.captum_explainer_1.influence(
                inputs=test_data,
                top_k=ds_len(self.train_dataset) - self.modulo_batch_size,
            )[self.layer]
            if self.modulo_batch_size > 0:
                topk_idx_2, topk_val_2 = self.captum_explainer_2.influence(
                    inputs=test_data, top_k=self.modulo_batch_size
                )[self.layer]
                _, inverted_idx_1 = topk_idx_1.sort()
                _, inverted_idx_2 = topk_idx_2.sort()
                return torch.cat(
                    [
                        torch.gather(topk_val_1, 1, inverted_idx_1),
                        torch.gather(topk_val_2, 1, inverted_idx_2),
                    ],
                    dim=1,
                )
            else:
                _, inverted_idx = topk_idx_1.sort()
                return torch.gather(topk_val_1, 1, inverted_idx)


def captum_similarity_explain(
    model: Union[torch.nn.Module, L.LightningModule],
    model_id: str,
    test_data: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `CaptumSimilarity` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    model_id : str
        Identifier for the model.
    test_data : torch.Tensor
        Test samples for which influence scores are computed.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    cache_dir : str, optional
        Directory for caching activations. Defaults to "./cache".
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        checkpoints=checkpoints,
        model_id=model_id,
        cache_dir=cache_dir,
        test_data=test_data,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def captum_similarity_self_influence(
    model: Union[torch.nn.Module, L.LightningModule],
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    batch_size: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional CaptumSimilarity explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    model_id : str
        Identifier for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    cache_dir : str, optional
        Directory for caching activations. Defaults to "./cache".
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        batch_size=batch_size,
        **kwargs,
    )


class CaptumArnoldi(CaptumInfluence):
    """Class for Arnoldi Influence Function wrapper.

    This implements the ArnoldiInfluence method of Schioppa et al. (2022) to
    compute influence function explanations as described by Koh et al. (2017).

    Notes
    -----
    The user is referred to captum's codebase [3] for details on the specifics
    of the parameters.

    References
    ----------
    (1) Schioppa, Andrea, et al. (2022). Scaling up influence functions.
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36.
    No. 8.

    (2) Koh, Pang Wei, and Percy Liang. (2017). "Understanding black-box
    predictions via influence functions." International conference on machine
    learning. PMLR

    (3) https://github.com/pytorch/captum/blob/master/captum/influence/_core/
    arnoldi_influence_function.py

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str]],
        task: TaskLiterals = "image_classification",
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(
            reduction="none"
        ),
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        batch_size: int = 1,
        hessian_dataset: Optional[torch.utils.data.Dataset] = None,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: int = 50,
        seed: int = 0,
        arnoldi_dim: int = 200,
        arnoldi_tol: float = 1e-1,
        hessian_reg: float = 1e-3,
        hessian_inverse_tol: float = 1e-4,
        projection_on_cpu: bool = True,
        show_progress: bool = False,
        device: str = "cpu",
        **explainer_kwargs: Any,
    ):
        """Initialize CaptumArnoldi explainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Checkpoints for the model.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification".
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        loss_fn : Union[torch.nn.Module, Callable], optional
            Loss function which is applied to the model. Required to be a
            reduction='none' loss.
            Defaults to CrossEntropyLoss with reduction='none'.
        checkpoints_load_func : Optional[Callable], optional
            Function to load checkpoints. If None, a default function is used.
            Defaults to None.
        layers : Optional[List[str]], optional
            Layers used to compute the gradients. If None, all layers are used.
            Defaults to None.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        hessian_dataset : Optional[torch.utils.data.Dataset], optional
            Dataset for calculating the Hessian. It should be smaller than
            train_dataset.
            If None, the entire train_dataset is used. Defaults to None.
        test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function which is used for the test samples. If None, loss_fn
            is used. Defaults to None.
        sample_wise_grads_per_batch : bool, optional
            Whether to compute sample-wise gradients per batch. Defaults to
            False.
            Note: This feature is currently not supported.
        projection_dim : int, optional
            Captum's ArnoldiInfluenceFunction produces a low-rank approximation
            of the (inverse) Hessian.
            projection_dim is the rank of that approximation. Defaults to 50.
        seed : int, optional
            Random seed for reproducibility. Defaults to 0.
        arnoldi_dim : int, optional
            Calculating the low-rank approximation of the (inverse) Hessian
            requires approximating the Hessian's top eigenvectors /
            eigenvalues. This is done by first computing a Krylov subspace via
            the Arnoldi iteration, and then finding the top eigenvectors /
            eigenvalues of the restriction of the Hessian to the Krylov
            subspace. Because only the top eigenvectors / eigenvalues computed
            in the restriction will be similar to those in the full space,
            `arnoldi_dim` should be chosen to be larger than `projection_dim`.
            Defaults to 200.
        arnoldi_tol : float, optional
            After many iterations, the already-obtained basis vectors may
            already approximately span the Krylov subspace, in which case the
            addition of additional basis vectors involves normalizing a vector
            with a small norm. These vectors are not necessary to include in
            the basis and furthermore, their small norm leads to numerical
            issues. Therefore we stop the Arnoldi iteration when the addition
            of additional vectors involves normalizing a vector with norm
            below a certain threshold. This argument specifies that threshold.
            Defaults to 1e-1.
        hessian_reg : float, optional
            After computing the basis for the Krylov subspace, the restriction
            of the Hessian to the subspace may not be positive definite, which
            is required, as we compute a low-rank approximation of its square
            root via eigen-decomposition. `hessian_reg` adds an entry to the
            diagonals of the restriction of the Hessian to encourage it to be
            positive definite. This argument specifies that entry. Note that
            the regularized Hessian (i.e. with `hessian_reg` added to its
            diagonals) does not actually need to be positive definite - it just
            needs to have at least 1 positive eigenvalue. Defaults to 1e-3.
        hessian_inverse_tol : float, optional
            The tolerance to use when computing the pseudo-inverse of the
            (square root of) hessian, restricted to the Krylov subspace.
            Defaults to 1e-4.
        projection_on_cpu : bool, optional
            Whether to move the projection, i.e. low-rank approximation of the
            inverse Hessian, to cpu, to save gpu memory. Defaults to True.
        show_progress : bool, optional
            Whether to display a progress bar. Defaults to False.
        device : str, optional
            Device to run the computation on. Defaults to "cpu".
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.

        """
        logger.info("Initializing Captum ArnoldiInfluence explainer...")

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(
                    f"{arg} is not supported by CaptumArnoldi explainer. "
                    f"Ignoring the argument."
                )

        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            task=task,
            explainer_cls=ArnoldiInfluenceFunction,
            explain_kwargs=explainer_kwargs,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.hessian_dataset = (
            OnDeviceDataset(hessian_dataset, self.device)
            if hessian_dataset is not None
            else None
        )
        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": self.train_dataset,
                "checkpoint": self.checkpoints[-1],
                "checkpoints_load_func": self.checkpoints_load_func,
                "layers": layers,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "hessian_dataset": self.hessian_dataset,
                "test_loss_fn": test_loss_fn,
                "sample_wise_grads_per_batch": sample_wise_grads_per_batch,
                "projection_dim": projection_dim,
                "seed": seed,
                "arnoldi_dim": arnoldi_dim,
                "arnoldi_tol": arnoldi_tol,
                "hessian_reg": hessian_reg,
                "hessian_inverse_tol": hessian_inverse_tol,
                "projection_on_cpu": projection_on_cpu,
                "show_progress": show_progress,
                **explainer_kwargs,
            }
        )
        self.init_explainer(**explainer_kwargs)

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ):
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(
            inputs=(test_data, targets)
        )
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is
            ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.

        """
        influence_scores = self.captum_explainer.self_influence(
            inputs_dataset=None
        )
        return influence_scores


def captum_arnoldi_explain(
    model: Union[torch.nn.Module, L.LightningModule],
    checkpoints: Union[str, List[str]],
    test_data: torch.Tensor,
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `CaptumArnoldi` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    test_data : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumArnoldi,
        model=model,
        checkpoints=checkpoints,
        test_data=test_data,
        targets=explanation_targets,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def captum_arnoldi_self_influence(
    model: torch.nn.Module,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional `CaptumArnoldi` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumArnoldi,
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


class CaptumTracInCP(CaptumInfluence):
    """Wrapper for the captum TracInCP explainer.

    Notes
    -----
    The user is referred to captum's codebase [2] for details on the specifics
    of the parameters.

    References
    ----------
    (1) Pruthi, Garima, et al. (2020). Estimating training data influence by
    tracing gradient descent. Advances in Neural Information Processing Systems
    33. (19920-19930).

    (2) https://github.com/pytorch/captum/blob/master/captum/influence/_core
    /tracincp.py

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str]],
        task: TaskLiterals = "image_classification",
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[
            Union[torch.nn.Module, Callable]
        ] = torch.nn.CrossEntropyLoss(reduction="none"),
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        device: str = "cpu",
        **explainer_kwargs: Any,
    ):
        """Initialize the `CaptumTracInCP` explainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Checkpoints for the model.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification", "text_classification",
            "causal_lm".
        checkpoints_load_func : Optional[Callable], optional
            Function to load checkpoints. If None, a default function is used.
            Defaults to None.
        layers : Optional[List[str]], optional
            Layers used to compute the gradients. Defaults to None.
        loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function used for influence computation.
            If reduction='none', then sample_wise_grads_per_batch must be set
            to False. Otherwise, sample_wise_grads_per_batch must be True.
            Defaults to CrossEntropyLoss with reduction='none'.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function which is used for the test samples. If None, loss_fn
            is used. Defaults to None.
        sample_wise_grads_per_batch : bool, optional
            Whether to compute sample-wise gradients per batch.
            If set to True, the loss function must use a reduction method (f.e.
            reduction='sum').
            Defaults to False.
        device : str, optional
            Device to run the computation on. Defaults to "cpu".
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.

        """
        logger.info("Initializing Captum TracInCP explainer...")

        unsupported_args = ["k", "proponents", "aggregate"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(
                    f"{arg} is not supported by CaptumTraceInCP explainer. "
                    f"Ignoring the argument."
                )

        self.outer_loop_by_checkpoints = explainer_kwargs.pop(
            "outer_loop_by_checkpoints", False
        )
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            task=task,
            explainer_cls=TracInCP,
            explain_kwargs=explainer_kwargs,
            checkpoints_load_func=checkpoints_load_func,
        )

        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": self.checkpoints_load_func,
                "layers": layers,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "test_loss_fn": test_loss_fn,
                "sample_wise_grads_per_batch": sample_wise_grads_per_batch,
                **explainer_kwargs,
            }
        )

        self.init_explainer(**explainer_kwargs)
        self.device = device

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ):
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(
            inputs=(test_data, targets)
        )
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is
            ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.

        """
        influence_scores = self.captum_explainer.self_influence(
            inputs=None,
            outer_loop_by_checkpoints=self.outer_loop_by_checkpoints,
        )
        return influence_scores


def captum_tracincp_explain(
    model: Union[torch.nn.Module, L.LightningModule],
    checkpoints: Union[str, List[str]],
    test_data: torch.Tensor,
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `CaptumTracInCP` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    test_data : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCP,
        model=model,
        checkpoints=checkpoints,
        test_data=test_data,
        targets=explanation_targets,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def captum_tracincp_self_influence(
    model: Union[torch.nn.Module, L.LightningModule],
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional `CaptumTracInCP` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCP,
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


class CaptumTracInCPFast(CaptumInfluence):
    """Wrapper for the captum TracInCPFast explainer.

    This implements the TracIn method by Pruthi et al. (2020) using only the
    final layer parameters.

    Notes
    -----
    The user is referred to captum's codebase [2] for details on the specifics
    of the parameters.

    References
    ----------
    (1) Pruthi, Garima, et al. (2020). "Estimating training data influence by
    tracing gradient descent."
        Advances in Neural Information Processing Systems 33. (19920-19930).

    (2) https://github.com/pytorch/captum/blob/master/captum/influence/_core/
    tracincp_fast_rand_proj.py

    """

    def __init__(
        self,
        model: torch.nn.Module,
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str]],
        task: TaskLiterals = "image_classification",
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        loss_fn: Optional[
            Union[torch.nn.Module, Callable]
        ] = torch.nn.CrossEntropyLoss(reduction="sum"),
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        vectorize: bool = False,
        device: str = "cpu",
        **explainer_kwargs: Any,
    ):
        """Initialize the `CaptumTracInCPFast` explainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be used for the influence computation.
        final_fc_layer : torch.nn.Module
            Final fully connected layer of the model.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Checkpoints for the model.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification".
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load checkpoints. If None, a default function is used.
        loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function used for influence computation. Defaults to
            `CrossEntropyLoss` with `reduction='sum'`.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function which is used for the test samples. If None, loss_fn
            is used. Defaults to None.
        vectorize : bool, optional
            Whether to use experimental vectorize functionality for
            `torch.autograd.functional.jacobian`. Defaults to False.
        device : str, optional
            Device to run the computation on. Defaults to "cpu".
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.

        """
        logger.info("Initializing Captum TracInCPFast explainer...")

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(
                    f"{arg} is not supported by CaptumTraceInCPFast "
                    f"explainer. Ignoring the argument."
                )

        self.outer_loop_by_checkpoints = explainer_kwargs.pop(
            "outer_loop_by_checkpoints", False
        )

        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            task=task,
            explainer_cls=TracInCPFast,
            explain_kwargs=explainer_kwargs,
            checkpoints_load_func=checkpoints_load_func,
        )

        explainer_kwargs.update(
            {
                "model": model,
                "final_fc_layer": final_fc_layer,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": self.checkpoints_load_func,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "test_loss_fn": test_loss_fn,
                "vectorize": vectorize,
                **explainer_kwargs,
            }
        )
        self.init_explainer(**explainer_kwargs)

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ):
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(
            inputs=(test_data, targets), k=None
        )
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is
            ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.

        """
        influence_scores = self.captum_explainer.self_influence(
            inputs=None,
            outer_loop_by_checkpoints=self.outer_loop_by_checkpoints,
        )
        return influence_scores


def captum_tracincp_fast_explain(
    model: torch.nn.Module,
    test_data: torch.Tensor,
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: torch.utils.data.Dataset,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `CaptumTracInCPFast` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    test_data : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCPFast,
        model=model,
        test_data=test_data,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def captum_tracincp_fast_self_influence(
    model: torch.nn.Module,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    outer_loop_by_checkpoints: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional CaptumTracInCPFast` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load checkpoints. If None, a default function is used.
    outer_loop_by_checkpoints : bool, optional
        Whether to perform an outer loop over the checkpoints. Defaults to
        False.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCPFast,
        checkpoints=checkpoints,
        model=model,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        outer_loop_by_checkpoints=outer_loop_by_checkpoints,
        **kwargs,
    )


class CaptumTracInCPFastRandProj(CaptumInfluence):
    """Wrapper for the captum TracInCPFastRandProj explainer.

    This implements the TracIn method by Pruthi et al. (2020) using only the
    final layer parameters
    and random projections to speed up computation.

    Notes
    -----
    The user is referred to captum's codebase [2] for details on the specifics
    of the parameters.

    References
    ----------
    (1) Pruthi, Garima, et al. (2020). "Estimating training data influence by
    tracing gradient descent." Advances in Neural Information Processing
    Systems 33. (19920-19930).

    (2) https://github.com/pytorch/captum/blob/master/captum/influence/_core/
    tracincp_fast_rand_proj.py

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str]],
        task: TaskLiterals = "image_classification",
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(
            reduction="sum"
        ),
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        vectorize: bool = False,
        nearest_neighbors: Optional[NearestNeighbors] = None,
        projection_dim: Optional[int] = None,
        seed: int = 0,
        device: str = "cpu",
        **explainer_kwargs: Any,
    ):
        """Initialize the `CaptumTracInCPFastRandProj` explainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        final_fc_layer : torch.nn.Module
            Final fully connected layer of the model.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Checkpoints for the model.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification".
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load checkpoints. If None, a default function is used.
        loss_fn : Union[torch.nn.Module, Callable], optional
            Loss function used for influence computation. Defaults to `
            CrossEntropyLoss` with `reduction='sum'`.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
            Loss function which is used for the test samples. If None, loss_fn
            is used. Defaults to None.
        vectorize : bool, optional
            Whether to use experimental vectorize functionality for
            `torch.autograd.functional.jacobian`. Defaults to False.
        nearest_neighbors : Optional[NearestNeighbors], optional
            Nearest neighbors model for finding nearest neighbors. If None,
            defaults to AnnoyNearestNeighbors is used.
        projection_dim : Optional[int], optional
            Each example will be represented in
            the nearest neighbors data structure with a vector. This vector
            is the concatenation of several "checkpoint vectors", each of which
            is computed using a different checkpoint in the `checkpoints`
            argument. If `projection_dim` is an int, it represents the
            dimension we will project each "checkpoint vector" to, so that the
            vector for each example will be of dimension at most
            `projection_dim` * C, where C is the number of checkpoints.
            Regarding the dimension of each vector, D: Let I be the dimension
            of the output of the last fully-connected layer times the dimension
            of the input of the last fully-connected layer. If `projection_dim`
            is not `None`, then D = min(I * C, `projection_dim` * C).
            Otherwise, D = I * C. In summary, if `projection_dim` is None, the
            dimension of this vector will be determined by the size of the
            input and output of the last fully-connected layer of `model`, and
            the number of checkpoints. Otherwise, `projection_dim` must be an
            int, and random projection will be performed to ensure that the
            vector is of dimension no more than `projection_dim` * C.
            `projection_dim` corresponds to the variable d in the top of page
            5 of the TracIn paper (Reference 1).
        seed : int, optional
            Random seed for reproducibility. Defaults to 0.
        device : str, optional
            Device to run the computation on. Defaults to "cpu".
        **explainer_kwargs : Any
            Additional keyword arguments passed to the explainer.

        """
        logger.info("Initializing Captum TracInCPFastRandProj explainer...")

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(
                    f"{arg} is not supported by CaptumTraceInCPFastRandProj "
                    f"explainer. Ignoring the argument."
                )

        self.outer_loop_by_checkpoints = explainer_kwargs.pop(
            "outer_loop_by_checkpoints", False
        )
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            task=task,
            explainer_cls=TracInCPFastRandProj,
            explain_kwargs=explainer_kwargs,
            checkpoints_load_func=checkpoints_load_func,
        )

        explainer_kwargs.update(
            {
                "model": model,
                "final_fc_layer": final_fc_layer,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": self.checkpoints_load_func,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "test_loss_fn": test_loss_fn,
                "vectorize": vectorize,
                "nearest_neighbors": nearest_neighbors,
                "projection_dim": projection_dim,
                "seed": seed,
                **explainer_kwargs,
            }
        )

        self.init_explainer(**explainer_kwargs)

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ):
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(
            inputs=(test_data, targets), k=None
        )
        return influence_scores


def captum_tracincp_fast_rand_proj_explain(
    model: Union[torch.nn.Module, L.LightningModule],
    checkpoints: Union[str, List[str]],
    test_data: torch.Tensor,
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `CaptumTracInCPFastRandProj` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    test_data : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCPFastRandProj,
        model=model,
        checkpoints=checkpoints,
        test_data=test_data,
        targets=explanation_targets,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def captum_tracincp_fast_rand_proj_self_influence(
    model: torch.nn.Module,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    outer_loop_by_checkpoints: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional `CaptumTracInCPFastRandProj` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        Checkpoints for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used.
        Defaults to None.
    outer_loop_by_checkpoints : bool, optional
        Whether to perform an outer loop over the checkpoints. Defaults to
        False.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCPFastRandProj,
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        outer_loop_by_checkpoints=outer_loop_by_checkpoints,
        **kwargs,
    )
