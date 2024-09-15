import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Union
from captum._utils.av import AV
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
        self.captum_explainer = self.explainer_cls(**explain_kwargs)

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError


class CaptumSimilarity(CaptumInfluence):
    # TODO: incorporate SimilarityInfluence kwargs into init_kwargs
    """
    init_kwargs = signature(SimilarityInfluence.__init__).parameters.items()
    init_kwargs.append("replace_nan")
    explain_kwargs = signature(SimilarityInfluence.influence)
    si_kwargs = signature(SimilarityInfluence.selinfluence)
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
        load_from_disk: bool = True,
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

        # As opposed to the original implementation, we move the activation generation to the init method.
        AV.generate_dataset_activations(
            self.cache_dir,
            self.model,
            self.model_id,
            self.layer,
            torch.utils.data.DataLoader(self.train_dataset, batch_size, shuffle=False),
            identifier="src",
            load_from_disk=load_from_disk,
            return_activations=True,
        )

    @property
    def layer(self):
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
    batch_size: int = 32,
    **kwargs: Any,
) -> torch.Tensor:
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
    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoint: str,
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(),
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
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        influence_scores = self.captum_explainer.self_influence(inputs_dataset=None)
        return influence_scores


def captum_arnoldi_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
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
    batch_size: int = 32,
    **kwargs: Any,
) -> torch.Tensor:
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumArnoldi,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        batch_size=batch_size,
        **kwargs,
    )


class CaptumTracInCP(CaptumInfluence):
    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
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
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        influence_scores = self.captum_explainer.self_influence(
            inputs=None, outer_loop_by_checkpoints=self.outer_loop_by_checkpoints
        )
        return influence_scores


def captum_tracincp_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    model_id: Optional[str] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
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
    batch_size: int = 32,
    **kwargs: Any,
) -> torch.Tensor:
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCP,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        batch_size=batch_size,
        **kwargs,
    )


class CaptumTracInCPFast(CaptumInfluence):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
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
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets), k=None)
        return influence_scores

    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        influence_scores = self.captum_explainer.self_influence(
            inputs=None, outer_loop_by_checkpoints=self.outer_loop_by_checkpoints
        )
        return influence_scores


def captum_tracincp_fast_explain(
    model: torch.nn.Module,
    model_id: str,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
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
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    outer_loop_by_checkpoints: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
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
    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        model_id: str,
        final_fc_layer: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
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
        """
        # Initialize TracInCPFast to use its self_influence method
        self.tracin_fast_explainer = TracInCPFast(
            model=model,
            final_fc_layer=final_fc_layer,
            train_dataset=train_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            loss_fn=loss_fn,
            batch_size=batch_size,
            test_loss_fn=test_loss_fn,
            vectorize=vectorize,
        )
        """

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets), k=None)
        return influence_scores

    """
    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        influence_scores = self.tracin_fast_explainer.self_influence(
            inputs=None, outer_loop_by_checkpoints=self.outer_loop_by_checkpoints
        )
        return influence_scores
    """


def captum_tracincp_fast_rand_proj_explain(
    model: Union[torch.nn.Module, pl.LightningModule],
    model_id: str,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
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


def captum_tracincp_fast_rand_proj_self_influence(
    model: torch.nn.Module,
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    outer_loop_by_checkpoints: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCPFastRandProj,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        outer_loop_by_checkpoints=outer_loop_by_checkpoints,
        **kwargs,
    )
