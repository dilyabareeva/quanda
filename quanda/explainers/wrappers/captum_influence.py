import copy
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import torch
from captum.influence import SimilarityInfluence, TracInCP  # type: ignore

# TODO Should be imported directly from captum.influence once available
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)

from quanda.explainers.base import BaseExplainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.functions import cosine_similarity
from quanda.utils.validation import validate_checkpoints_load_func


class CaptumInfluence(BaseExplainer, ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explainer_cls: type,
        explain_kwargs: Any,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
        )
        self.explainer_cls = explainer_cls
        self.explain_kwargs = explain_kwargs
        self._init_explainer(**explain_kwargs)

    def _init_explainer(self, **explain_kwargs: Any):
        self.captum_explainer = self.explainer_cls(**explain_kwargs)

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """Comment for Galip and Niklas: We are now expecting explicit declaration of
        explain method keyword arguments in specific explainer class __init__ methods.
        Right now the only such keyword argument is `top_k` for CaptumSimilarity, which we
        anyway overwrite with the dataset length."""
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
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        layers: Union[str, List[str]],
        similarity_metric: Callable = cosine_similarity,
        similarity_direction: str = "max",
        batch_size: int = 1,
        replace_nan: bool = False,
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        # extract and validate layer from kwargs
        self._layer: Optional[Union[List[str], str]] = None
        self.layer = layers

        if device != "cpu":
            warnings.warn("CaptumSimilarity explainer only supports CPU devices. Setting device to 'cpu'.")
            device = "cpu"

        model_passed = copy.deepcopy(model)  # CaptumSimilarity only does cpu,
        # we still want to keep the model on cuda for the metrics
        # TODO: validate SimilarityInfluence kwargs
        explainer_kwargs.update(
            {
                "module": model_passed,
                "influence_src_dataset": train_dataset,
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

        super().__init__(
            model=model_passed,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            explain_kwargs=explainer_kwargs,
        )

        # explicitly specifying explain method kwargs as instance attributes
        self.top_k = self.dataset_length

        if "top_k" in explainer_kwargs:
            warnings.warn("top_k is not supported by CaptumSimilarity explainer. Ignoring the argument.")

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

        topk_idx, topk_val = self.captum_explainer.influence(inputs=test, top_k=self.top_k)[self.layer]
        _, inverted_idx = topk_idx.sort()
        return torch.gather(topk_val, 1, inverted_idx)


def captum_similarity_explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
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
        device=device,
        **kwargs,
    )


def captum_similarity_self_influence(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    batch_size: Optional[int] = 32,
    **kwargs: Any,
) -> torch.Tensor:
    self_influence_kwargs = {
        "batch_size": batch_size,
    }
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        self_influence_kwargs=self_influence_kwargs,
        **kwargs,
    )


class CaptumArnoldi(CaptumInfluence):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoint: str,
        loss_fn: Union[torch.nn.Module, Callable] = torch.nn.CrossEntropyLoss(),
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        batch_size: int = 1,
        hessian_dataset: Optional[Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]] = None,
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
        device: Union[str, torch.device] = "cpu",  # TODO Check if gpu works
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
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

        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": train_dataset,
                "checkpoint": checkpoint,
                "checkpoints_load_func": checkpoints_load_func,
                "layers": layers,
                "loss_fn": loss_fn,
                "batch_size": batch_size,
                "hessian_dataset": hessian_dataset,
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

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=ArnoldiInfluenceFunction,
            explain_kwargs=explainer_kwargs,
        )

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, **kwargs: Any) -> torch.Tensor:
        inputs_dataset = kwargs.get("inputs_dataset", None)
        influence_scores = self.captum_explainer.self_influence(inputs_dataset=inputs_dataset)
        return influence_scores


def captum_arnoldi_explain(
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
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
        device=device,
        **kwargs,
    )


def captum_arnoldi_self_influence(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    inputs_dataset: Optional[Union[Tuple[Any, ...], torch.utils.data.DataLoader]] = None,
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    self_influence_kwargs = {
        "inputs_dataset": inputs_dataset,
    }
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumArnoldi,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        self_influence_kwargs=self_influence_kwargs,
        **kwargs,
    )


class CaptumTracInCP(CaptumInfluence):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str], Iterator],
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layers: Optional[List[str]] = None,
        loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        batch_size: int = 1,
        test_loss_fn: Optional[Union[torch.nn.Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        device: Union[str, torch.device] = "cpu",
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
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

        explainer_kwargs.update(
            {
                "model": model,
                "train_dataset": train_dataset,
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

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=TracInCP,
            explain_kwargs=explainer_kwargs,
        )

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        test = test.to(self.device)

        if targets is not None:
            if isinstance(targets, list):
                targets = torch.tensor(targets).to(self.device)
            else:
                targets = targets.to(self.device)

        influence_scores = self.captum_explainer.influence(inputs=(test, targets))
        return influence_scores

    def self_influence(self, **kwargs: Any) -> torch.Tensor:
        inputs = kwargs.get("inputs", None)
        outer_loop_by_checkpoints = kwargs.get("outer_loop_by_checkpoints", False)
        influence_scores = self.captum_explainer.self_influence(
            inputs=inputs, outer_loop_by_checkpoints=outer_loop_by_checkpoints
        )
        return influence_scores


def captum_tracincp_explain(
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
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
        device=device,
        **kwargs,
    )


def captum_tracincp_self_influence(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    inputs: Optional[Union[Tuple[Any, ...], torch.utils.data.DataLoader]] = None,
    outer_loop_by_checkpoints: bool = False,
    model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    self_influence_kwargs = {"inputs": inputs, "outer_loop_by_checkpoints": outer_loop_by_checkpoints}
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumTracInCP,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        self_influence_kwargs=self_influence_kwargs,
        **kwargs,
    )
