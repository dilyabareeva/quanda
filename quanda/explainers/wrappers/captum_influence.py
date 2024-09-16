import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Union

import pytorch_lightning as pl
import torch
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
from quanda.utils.common import default_tensor_type, get_load_state_dict_func
from quanda.utils.datasets import OnDeviceDataset
from quanda.utils.functions import cosine_similarity
from quanda.utils.validation import validate_checkpoints_load_func


class CaptumInfluence(Explainer, ABC):
    """
    Base class for the Captum explainers.

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
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        explain_kwargs: Any,
        model_id: Optional[str] = None,
        cache_dir: str = "./cache",
    ):
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
        )
        self.explainer_cls = explainer_cls
        self.explain_kwargs = explain_kwargs

    def init_explainer(self, **explain_kwargs: Any):
        """Initialize the Captum explainer.

        Parameters
        ----------
        **explain_kwargs : Any
            Additional keyword arguments to be passed to the explainer."""
        self.captum_explainer = self.explainer_cls(**explain_kwargs)

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """
        Abstract method for computing influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]], optional
            Labels for the test samples. Defaults to None.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        raise NotImplementedError


class CaptumSimilarity(CaptumInfluence):
    # TODO: incorporate SimilarityInfluence kwargs into init_kwargs
    # TODO: Check usage of 'replace_nan' in SimilarityInfluence
    """
    Class for Similarity Influence wrapper. This explainer uses a similarity function on its inputs to rank the training data.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    model_id : str
        Identifier for the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    layers : Union[str, List[str]]
        Layers of the model for which the activations are computed.
    cache_dir : str, optional
        Directory for caching activations. Defaults to "./cache".
    similarity_metric : Callable, optional
        Metric for computing similarity. Defaults to `cosine_similarity`.
    similarity_direction : str, optional
        Direction for similarity computation. Can be either "min" or "max". Defaults to "max".
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    replace_nan : bool, optional
        Whether to replace NaN values in similarity scores. Defaults to False.
    **explainer_kwargs : Any
        Additional keyword arguments passed to the explainer.

    References
    ----------
    1) https://captum.ai/api/influence.html#similarityinfluence
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        model_id: str,
        train_dataset: torch.utils.data.Dataset,
        layers: Union[str, List[str]],
        cache_dir: str = "./cache",
        similarity_metric: Callable = cosine_similarity,
        similarity_direction: str = "max",
        batch_size: int = 1,
        replace_nan: bool = False,
        **explainer_kwargs: Any,
    ):
        # extract and validate layer from kwargs
        self._layer: Optional[Union[List[str], str]] = None
        self.layer = layers

        # TODO: validate SimilarityInfluence kwargs

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            explainer_cls=SimilarityInfluence,
            explain_kwargs=explainer_kwargs,
        )

        explainer_kwargs.update(
            {
                "module": model,
                "influence_src_dataset": self.train_dataset,
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

        self.init_explainer(**explainer_kwargs)
        # explicitly specifying explain method kwargs as instance attributes
        self.top_k = self.dataset_length

        if "top_k" in explainer_kwargs:
            warnings.warn("top_k is not supported by CaptumSimilarity explainer. Ignoring the argument.")

    @property
    def layer(self):
        """Return the layer for which the activations are computed."""
        return self._layer

    @layer.setter
    def layer(self, layers: Any):
        """
        Our wrapper only allows a single layer to be passed, while the Captum implementation allows multiple layers.
        Here, we validate if there is only a single layer passed.
        """
        if isinstance(layers, str):
            self._layer = layers
            return
        if len(layers) != 1:
            raise ValueError("A single layer shall be passed to the CaptumSimilarity explainer.")
        self._layer = layers[0]

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]], optional
            Labels for the test samples. This argument is ignored.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test = test.to(self.device)

        if targets is not None:
            self._process_targets(targets=targets)
            warnings.warn("CaptumSimilarity explainer does not support target indices. Ignoring the argument.")

        with default_tensor_type(self.device):
            topk_idx, topk_val = self.captum_explainer.influence(inputs=test, top_k=self.top_k)[self.layer]

        _, inverted_idx = topk_idx.sort()
        return torch.gather(topk_val, 1, inverted_idx)


def captum_similarity_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    model_id: str,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `CaptumSimilarity` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    model_id : str
        Identifier for the model.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    cache_dir : str, optional
        Directory for caching activations. Defaults to "./cache".
    explanation_targets : Optional[Union[List[int], torch.Tensor]], optional
        Labels for the test samples. Defaults to None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def captum_similarity_self_influence(
    model: Union[torch.nn.Module, pl.LightningModule],
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    batch_size: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the self-influence scores using the CaptumSimilarity explainer.

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
    """
    Class for Arnoldi Influence Function wrapper.
    This implements the ArnoldiInfluence method of (1) to compute influence function explanations (2).
    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoint : str
        Checkpoint file for the model.
    loss_fn : Union[torch.nn.Module, Callable], optional
        Loss function which is applied to the model. Required to be a reduction='none' loss.
        Defaults to CrossEntropyLoss with reduction='none'.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used. Defaults to None.
    layers : Optional[List[str]], optional
        Layers used to compute the gradients. If None, all layers are used. Defaults to None.
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    hessian_dataset : Optional[torch.utils.data.Dataset], optional
        Dataset for calculating the Hessian. It should be smaller than train_dataset.
        If None, the entire train_dataset is used. Defaults to None.
    test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function which is used for the test samples. If None, loss_fn is used. Defaults to None.
    projection_dim : int, optional
        Captum's ArnoldiInfluenceFunction produces a low-rank approximation of the (inverse) Hessian.
        projection_dim is the rank of that approximation. Defaults to 50.
    seed : int, optional
        Random seed for reproducibility. Defaults to 0.
    arnoldi_dim : int, optional
        Calculating the low-rank approximation of the (inverse) Hessian requires approximating
        the Hessian's top eigenvectors / eigenvalues.
        This is done by first computing a Krylov subspace via the Arnoldi iteration,
        and then finding the top eigenvectors / eigenvalues of the restriction of the Hessian to the Krylov subspace.
        Because only the top eigenvectors / eigenvalues computed in the restriction will be similar to those in the full space,
        `arnoldi_dim` should be chosen to be larger than `projection_dim`.
        Defaults to 200.
    arnoldi_tol : float, optional
        After many iterations, the already-obtained basis vectors may already approximately span the Krylov subspace,
        in which case the addition of additional basis vectors involves normalizing a vector with a small norm.
        These vectors are not necessary to include in the basis and furthermore, their small norm leads to numerical issues.
        Therefore we stop the Arnoldi iteration when the addition of additional vectors involves normalizing a vector with norm
        below a certain threshold. This argument specifies that threshold. Defaults to 1e-1.
    hessian_reg : float, optional
        After computing the basis for the Krylov subspace, the restriction of the Hessian to the
        subspace may not be positive definite, which is required, as we compute a low-rank approximation
        of its square root via eigen-decomposition. `hessian_reg` adds an entry to the diagonals of the
        restriction of the Hessian to encourage it to be positive definite. This argument specifies that entry.
        Note that the regularized Hessian (i.e. with `hessian_reg` added to its diagonals) does not actually need
        to be positive definite - it just needs to have at least 1 positive eigenvalue.
        Defaults to 1e-3.
    hessian_inverse_tol : float, optional
        The tolerance to use when computing the pseudo-inverse of the (square root of) hessian,
        restricted to the Krylov subspace. Defaults to 1e-4.
    projection_on_cpu : bool, optional
        Whether to move the projection, i.e. low-rank approximation of the inverse Hessian, to cpu, to save gpu memory.
        Defaults to True.
    show_progress : bool, optional
        Whether to display a progress bar. Defaults to False.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    device : Union[str, torch.device], optional
        Device to run the computation on. Defaults to "cpu".
    **explainer_kwargs : Any
        Additional keyword arguments passed to the explainer.

    Notes
    ------
    The user is referred to captum's codebase for details on the specifics of the parameters.

    References
    ----------
    (1) Schioppa, Andrea, et al. "Scaling up influence functions."
        Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 8. 2022.
    (2) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions."
        International conference on machine learning. PMLR, 2017.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoint: str,
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(reduction="none"),
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
        model_id: Optional[str] = None,
        cache_dir: str = "./cache",
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        if checkpoints_load_func is None:
            checkpoints_load_func = get_load_state_dict_func(device)
        else:
            validate_checkpoints_load_func(checkpoints_load_func)

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(f"{arg} is not supported by CaptumArnoldi explainer. Ignoring the argument.")

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            explainer_cls=ArnoldiInfluenceFunction,
            explain_kwargs=explainer_kwargs,
        )

        self.hessian_dataset = OnDeviceDataset(hessian_dataset, self.device) if hessian_dataset is not None else None
        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": self.train_dataset,
                "checkpoint": checkpoint,
                "checkpoints_load_func": checkpoints_load_func,
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

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test = test.to(self.device)

        if targets is None:
            raise ValueError("Targets are required for CaptumArnoldi explainer.")

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """
        Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.
        """
        influence_scores = self.captum_explainer.self_influence(inputs_dataset=None)
        return influence_scores


def captum_arnoldi_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    test_tensor: torch.Tensor,
    explanation_targets: Optional[Union[List[int], torch.Tensor]],
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `CaptumArnoldi` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumArnoldi,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def captum_arnoldi_self_influence(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the self-influence scores using the `CaptumArnoldi` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
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
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        **kwargs,
    )


class CaptumTracInCP(CaptumInfluence):
    """
    Wrapper for the captum TracInCP explainer. This implements the TracIn method  (1).

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints : Union[str, List[str], Iterator]
        Checkpoints for the model.
    checkpoints_load_func : Optional[Callable], optional
        Function to load checkpoints. If None, a default function is used. Defaults to None.
    layers : Optional[List[str]], optional
        Layers used to compute the gradients. Defaults to None.
    loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function used for influence computation. Defaults to CrossEntropyLoss with reduction='sum'.
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function which is used for the test samples. If None, loss_fn is used. Defaults to None.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    device : Union[str, torch.device], optional
        Device to run the computation on. Defaults to "cpu".
    **explainer_kwargs : Any
        Additional keyword arguments passed to the explainer.


    References
    ----------
    (1) Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent."
        Advances in Neural Information Processing Systems 33 (2020): 19920-19930.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = torch.nn.CrossEntropyLoss(reduction="sum"),
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        model_id: Optional[str] = None,
        cache_dir: str = "./cache",
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        if checkpoints_load_func is None:
            checkpoints_load_func = get_load_state_dict_func(device)
        else:
            validate_checkpoints_load_func(checkpoints_load_func)

        unsupported_args = ["k", "proponents", "aggregate"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(f"{arg} is not supported by CaptumTraceInCP explainer. Ignoring the argument.")

        self.outer_loop_by_checkpoints = explainer_kwargs.pop("outer_loop_by_checkpoints", False)
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            explainer_cls=TracInCP,
            explain_kwargs=explainer_kwargs,
        )

        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": checkpoints_load_func,
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

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test = test.to(self.device)

        if targets is None:
            raise ValueError("Targets are required for CaptumTracInCP explainer.")

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """
        Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.
        """
        influence_scores = self.captum_explainer.self_influence(
            inputs=None, outer_loop_by_checkpoints=self.outer_loop_by_checkpoints
        )
        return influence_scores


def captum_tracincp_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    test_tensor: torch.Tensor,
    explanation_targets: Optional[Union[List[int], torch.Tensor]],
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `CaptumTracInCP` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCP,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def captum_tracincp_self_influence(
    model: Union[torch.nn.Module, pl.LightningModule],
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the self-influence scores using the `CaptumTracInCP` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
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
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        **kwargs,
    )


class CaptumTracInCPFast(CaptumInfluence):
    """
    Wrapper for the captum TracInCPFast explainer. This implements the TracIn method (1) using only the final layer parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    final_fc_layer : torch.nn.Module
        Final fully connected layer of the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints : Union[str, List[str], Iterator]
        Checkpoints for the model.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load checkpoints. If None, a default function is used.
    loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function used for influence computation. Defaults to `CrossEntropyLoss` with `reduction='sum'`.
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function which is used for the test samples. If None, loss_fn is used. Defaults to None.
    vectorize : bool, optional
        Whether to use experimental vectorize functionality for `torch.autograd.functional.jacobian`. Defaults to False.
    device : Union[str, torch.device], optional
        Device to run the computation on. Defaults to "cpu".
    **explainer_kwargs : Any
        Additional keyword arguments passed to the explainer.

    References
    ----------
    (1) Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent."
        Advances in Neural Information Processing Systems 33 (2020): 19920-19930.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
        model_id: Optional[str] = None,
        cache_dir: str = "./cache",
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        vectorize: bool = False,
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        if checkpoints_load_func is None:
            checkpoints_load_func = get_load_state_dict_func(device)
        else:
            validate_checkpoints_load_func(checkpoints_load_func)

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(f"{arg} is not supported by CaptumTraceInCPFast explainer. Ignoring the argument.")

        self.outer_loop_by_checkpoints = explainer_kwargs.pop("outer_loop_by_checkpoints", False)

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            explainer_cls=TracInCPFast,
            explain_kwargs=explainer_kwargs,
        )
        explainer_kwargs.update(
            {
                "model": model,
                "final_fc_layer": final_fc_layer,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": checkpoints_load_func,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "test_loss_fn": test_loss_fn,
                "vectorize": vectorize,
                **explainer_kwargs,
            }
        )
        self.init_explainer(**explainer_kwargs)

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test = test.to(self.device)

        if targets is None:
            raise ValueError("Targets are required for CaptumTracInCPFast explainer.")

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets), k=None)
        return influence_scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """
        Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.
        """
        influence_scores = self.captum_explainer.self_influence(
            inputs=None, outer_loop_by_checkpoints=self.outer_loop_by_checkpoints
        )
        return influence_scores


def captum_tracincp_fast_explain(
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    explanation_targets: Optional[Union[List[int], torch.Tensor]],
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `CaptumTracInCPFast` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCPFast,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def captum_tracincp_fast_self_influence(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    outer_loop_by_checkpoints: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the self-influence scores using the `CaptumTracInCPFast` explainer.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    outer_loop_by_checkpoints : bool, optional
        Whether to use checkpoints for the outer loop. Defaults to False.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.
    """
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCPFast,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        outer_loop_by_checkpoints=outer_loop_by_checkpoints,
        **kwargs,
    )


class CaptumTracInCPFastRandProj(CaptumInfluence):
    """
    Wrapper for the captum TracInCPFastRandProj explainer.
    This implements the TracIn method (1) using only the final layer parameters and random projections to speed up computation.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    final_fc_layer : torch.nn.Module
        Final fully connected layer of the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints : Union[str, List[str], Iterator]
        Checkpoints for the model.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load checkpoints. If None, a default function is used.
    loss_fn : Union[torch.nn.Module, Callable], optional
        Loss function used for influence computation. Defaults to `CrossEntropyLoss` with `reduction='sum'`.
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    test_loss_fn : Optional[Union[torch.nn.Module, Callable]], optional
        Loss function which is used for the test samples. If None, loss_fn is used. Defaults to None.
    vectorize : bool, optional
        Whether to use experimental vectorize functionality for `torch.autograd.functional.jacobian`. Defaults to False.
    nearest_neighbors : Optional[NearestNeighbors], optional
        Nearest neighbors model for finding nearest neighbors. If None, defaults to AnnoyNearestNeighbors is used.
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
        5 of the TracIn paper (1).
    seed : int, optional
        Random seed for reproducibility. Defaults to 0.
    device : Union[str, torch.device], optional
        Device to run the computation on. Defaults to "cpu".
    **explainer_kwargs : Any
        Additional keyword arguments passed to the explainer.

    References
    ----------
    (1) Pruthi, Garima, et al. "Estimating training data influence by tracing gradient descent."
        Advances in Neural Information Processing Systems 33 (2020): 19920-19930.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
        model_id: Optional[str] = None,
        cache_dir: str = "./cache",
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(reduction="sum"),
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        vectorize: bool = False,
        nearest_neighbors: Optional[NearestNeighbors] = None,
        projection_dim: Optional[int] = None,
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        if checkpoints_load_func is None:
            checkpoints_load_func = get_load_state_dict_func(device)
        else:
            validate_checkpoints_load_func(checkpoints_load_func)

        unsupported_args = ["k", "proponents"]
        for arg in unsupported_args:
            if arg in explainer_kwargs:
                explainer_kwargs.pop(arg)
                warnings.warn(f"{arg} is not supported by CaptumTraceInCPFastRandProj explainer. Ignoring the argument.")

        self.outer_loop_by_checkpoints = explainer_kwargs.pop("outer_loop_by_checkpoints", False)
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            explainer_cls=TracInCPFastRandProj,
            explain_kwargs=explainer_kwargs,
        )

        explainer_kwargs.update(
            {
                "model": model,
                "final_fc_layer": final_fc_layer,
                "train_dataset": self.train_dataset,
                "checkpoints": checkpoints,
                "checkpoints_load_func": checkpoints_load_func,
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

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        test = test.to(self.device)

        if targets is None:
            raise ValueError("Targets are required for CaptumTracInCPFastRandProj explainer.")

        if isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets), k=None)
        return influence_scores


def captum_tracincp_fast_rand_proj_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `CaptumTracInCPFastRandProj` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=CaptumTracInCPFastRandProj,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )
