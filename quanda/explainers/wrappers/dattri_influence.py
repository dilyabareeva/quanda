"""Wrappers for the dattri influence computation methods."""

import logging
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import datasets as hf_datasets  # type: ignore
import lightning as L
import torch
from dattri.algorithm.influence_function import (  # type: ignore
    IFAttributorArnoldi,
    IFAttributorCG,
    IFAttributorDataInf,
    IFAttributorEKFAC,
    IFAttributorExplicit,
    IFAttributorLiSSA,
)
from dattri.algorithm.tracin import TracInAttributor  # type: ignore
from dattri.algorithm.trak import TRAKAttributor  # type: ignore
from dattri.task import AttributionTask  # type: ignore

from quanda.explainers.base import Explainer
from quanda.utils.common import get_load_state_dict_func, process_targets
from quanda.utils.datasets.dataset_handlers import (
    HuggingFaceDatasetHandler,
    HuggingFaceSequenceDatasetHandler,
    get_dataset_handler,
)
from quanda.utils.tasks import TaskLiterals

logger = logging.getLogger(__name__)


def _resolve_model_fn(fn: Optional[Callable], model: Any) -> Any:
    """Resolve a loss/probability callable that may be a builder."""
    if fn is None:
        return None
    return fn(model)


def _wrap_checkpoints_load_func(
    checkpoints_load_func: Callable,
) -> Optional[Callable[..., torch.nn.Module]]:
    """Adapt quanda's ``checkpoints_load_func`` to dattri's contract."""

    def _load(model: torch.nn.Module, checkpoint: Any) -> torch.nn.Module:
        checkpoints_load_func(model, checkpoint)
        return model

    return _load


def _resolve_device(model: torch.nn.Module, device: Optional[str]) -> str:
    """Fall back to the model's parameter device when ``device`` is unset."""
    if device is not None:
        return device
    param = next(model.parameters(), None)
    return str(param.device) if param is not None else "cpu"


class DattriInfluence(Explainer, ABC):
    """Base class for dattri explainer wrappers."""

    accepted_tasks: List[TaskLiterals] = [
        "image_classification",
        "text_classification",
        "causal_lm",
    ]

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        attributor_cls: type,
        attributor_kwargs: Dict[str, Any],
        task: TaskLiterals = "image_classification",
        target_func: Optional[Callable] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        use_cache: bool = True,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the base `DattriInfluence` wrapper.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        loss_func : Callable
            Builder for dattri's `AttributionTask` loss, with signature:
            ```
            def loss_func(
                model: torch.nn.Module,
            ) -> Callable[[Dict[str, torch.Tensor], Tuple], torch.Tensor]:
                ...
            ```
            The returned callable takes `(params, batch)` and returns a
            per-sample loss tensor (compatible with
            `torch.func.functional_call`).
        attributor_cls : type
            The dattri attributor class.
        attributor_kwargs : Dict[str, Any]
            Keyword arguments passed to the dattri attributor constructor.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
        target_func : Optional[Callable], optional
            Builder for the `AttributionTask` target function, with the
            same signature as `loss_func`. If None, the loss function is
            used. Defaults to None.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s). If None, the current
            `state_dict` of `model` is used. Defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file.
        layer_name : Optional[Union[str, List[str]]], optional
            Layer name(s) to restrict the gradient computation to. If None,
            all parameters are used. Defaults to None.
        batch_size : int, optional
            Batch size for the dataloaders. Defaults to 1.
        collate_fn : Optional[Callable], optional
            Collate function for the dataloaders. Defaults to None.
        use_cache : bool, optional
            Whether to cache the full training data on the attributor at
            initialization. Defaults to True.
        device : str, optional
            Device to run the computation on. Defaults to the model's
            parameter device.
        hf_input_keys : Optional[Sequence[str]], optional
            Input keys of the HF ``train_dataset`` for the model inference; the
            ``"labels"`` field is always appended last. Defaults to
            ``("input_ids", "token_type_ids", "attention_mask")``.

        """
        device = _resolve_device(model, device)

        if checkpoints_load_func is None:
            checkpoints_load_func = get_load_state_dict_func(device)

        checkpoints_load_func = _wrap_checkpoints_load_func(
            checkpoints_load_func
        )
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.device = device
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.layer_name = layer_name
        self.hf_input_keys: Optional[Sequence[str]] = (
            tuple(hf_input_keys) if hf_input_keys is not None else None
        )

        loss_func = _resolve_model_fn(loss_func, model)
        target_func = _resolve_model_fn(target_func, model)

        task_kwargs: Dict[str, Any] = {}

        if target_func is not None:
            task_kwargs["target_func"] = target_func

        if self.checkpoints:
            task_checkpoints: Any = self.checkpoints
            task_load_func: Optional[Callable] = self.checkpoints_load_func
        else:
            task_checkpoints = [
                {k: v.detach().clone() for k, v in model.state_dict().items()}
            ]
            task_load_func = None

        self.attribution_task = AttributionTask(
            loss_func=loss_func,
            model=model,
            checkpoints=task_checkpoints,
            checkpoints_load_func=task_load_func,
            **task_kwargs,
        )

        init_kwargs = dict(attributor_kwargs)
        if layer_name is not None:
            init_kwargs.setdefault("layer_name", layer_name)
        init_kwargs.setdefault("device", device)

        self.attributor = attributor_cls(
            task=self.attribution_task, **init_kwargs
        )

        if use_cache and hasattr(self.attributor, "cache"):
            with torch.no_grad():
                self.attributor.cache(self._make_loader(self.train_dataset))

    def _make_loader(
        self,
        dataset: Any,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> torch.utils.data.DataLoader:
        """Build a DataLoader using the matching quanda dataset handler."""
        handler = get_dataset_handler(dataset)
        if isinstance(handler, HuggingFaceDatasetHandler) and not isinstance(
            handler, HuggingFaceSequenceDatasetHandler
        ):
            handler_kwargs: Dict[str, Any] = {}
            if self.hf_input_keys is not None:
                handler_kwargs["input_keys"] = self.hf_input_keys
            handler = HuggingFaceSequenceDatasetHandler(**handler_kwargs)
        return handler.create_dataloader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            collate_fn=collate_fn or self.collate_fn,
        )

    def _create_test_dataset(
        self,
        test_data: Union[torch.Tensor, Dict[str, Any]],
        targets: Optional[Union[List[int], torch.Tensor]],
    ) -> Any:
        """Turn `(test_data, targets)` into a dataset consumable by dattri."""
        if isinstance(test_data, torch.Tensor):
            if targets is None:
                raise ValueError("Targets required for tensor test_data.")
            targets = process_targets(targets, self.device)
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            return torch.utils.data.TensorDataset(
                test_data.to(self.device), targets.to(self.device)
            )
        if isinstance(test_data, dict):
            data = {
                k: v.tolist() if isinstance(v, torch.Tensor) else list(v)
                for k, v in test_data.items()
            }
            if targets is not None:
                data["labels"] = (
                    targets if isinstance(targets, list) else targets.tolist()
                )
            return hf_datasets.Dataset.from_dict(data)
        raise ValueError(
            f"Unsupported test_data type: {type(test_data)}. "
            "Expected torch.Tensor or Dict[str, Tensor]."
        )

    def _call_attribute(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.attributor.attribute(
                train_dataloader=train_loader, test_dataloader=test_loader
            )

    def explain(
        self,
        test_data: Union[torch.Tensor, Dict[str, Any]],
        targets: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Test samples for which influence scores are computed.
        targets : Optional[Union[List[int], torch.Tensor]], optional
            Labels for the test samples. Required when ``test_data`` is a
            bare tensor; merged into the dataset under the ``"labels"`` key
            when ``test_data`` is a dict.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size).

        """
        test_dataset = self._create_test_dataset(test_data, targets)
        test_loader = self._make_loader(test_dataset)
        train_loader = self._make_loader(self.train_dataset)
        scores = self._call_attribute(train_loader, test_loader)
        return scores.T.detach().cpu()

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute self-influence scores via dattri's `self_attribute`.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for iterating over the training dataset. Defaults
            to 1.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.

        """
        logger.info("Computing self-influence...")
        train_loader = self._make_loader(
            self.train_dataset, batch_size=batch_size
        )
        scores = self.attributor.self_attribute(train_loader)
        return scores.detach().cpu()


class DattriTRAK(DattriInfluence):
    """Wrapper for the `TRAKAttributor` from dattri.

    Notes
    -----
    For the `loss_func` and `correct_probability_func` conventions expected
    by dattri's `TRAKAttributor`, see dattri's documentation [2].

    References
    ----------
    (1) Park, Sung Min, et al. (2023). "TRAK: Attributing Model Behavior at
    Scale." Proceedings of the 40th International Conference on Machine
    Learning. PMLR. Vol. 202. (27074-27113).

    (2) https://github.com/TRAIS-Lab/dattri

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        correct_probability_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        regularization: float = 0.0,
        device: Optional[str] = None,
        use_cache: bool = True,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriTRAK` explainer."""
        logger.info("Initializing dattri TRAK explainer...")

        device = _resolve_device(model, device)
        correct_probability_func = _resolve_model_fn(
            correct_probability_func, model
        )
        attributor_kwargs: Dict[str, Any] = {
            "correct_probability_func": correct_probability_func,
            "regularization": regularization,
        }
        proj_kwargs = dict(projector_kwargs or {})
        proj_kwargs.setdefault("device", device)
        attributor_kwargs["projector_kwargs"] = proj_kwargs

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=TRAKAttributor,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=use_cache,
            device=device,
            hf_input_keys=hf_input_keys,
        )

    def _call_attribute(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        # dattri TRAK: attribute(test_loader, train_loader) → (n_train, n_test)
        with torch.no_grad():
            if self.attributor.full_train_dataloader is not None:
                return self.attributor.attribute(test_loader)
            return self.attributor.attribute(test_loader, train_loader)

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute TRAK self-influence scores."""
        logger.info("Computing self-influence...")
        if self.attributor.full_train_dataloader is not None:
            scores = self.attributor.self_attribute()
        else:
            train_loader = self._make_loader(
                self.train_dataset, batch_size=batch_size
            )
            scores = self.attributor.self_attribute(train_loader)
        return scores.detach().cpu()


class DattriTracInCP(DattriInfluence):
    """Wrapper for the `TracInAttributor` from dattri with TracInCP weights.

    This implements the TracIn method of Pruthi et al. (2020), where the
    `weight_list` corresponds to the checkpoint learning rates.

    References
    ----------
    (1) Pruthi, Garima, et al. (2020). "Estimating training data influence
    by tracing gradient descent." Advances in Neural Information Processing
    Systems 33. (19920-19930).

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        learning_rate: float = 0.001,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        normalized_grad: bool = False,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriTracInCP` explainer."""
        logger.info("Initializing dattri TracInCP explainer...")

        device = _resolve_device(model, device)
        n_ckpts = len(checkpoints) if isinstance(checkpoints, list) else 1
        weight_list = torch.tensor([learning_rate] * n_ckpts, device=device)

        attributor_kwargs: Dict[str, Any] = {
            "weight_list": weight_list,
            "normalized_grad": normalized_grad,
        }
        proj_kwargs = dict(projector_kwargs or {})
        proj_kwargs.setdefault("device", device)
        attributor_kwargs["projector_kwargs"] = proj_kwargs

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=TracInAttributor,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=False,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriGradDot(DattriTracInCP):
    """Grad-Dot wrapper using dattri's `TracInAttributor`.

    Grad-Dot is TracIn with a single unit weight and no gradient
    normalization: it computes the dot product of train/test gradients at a
    single checkpoint.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriGradDot` explainer."""
        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            learning_rate=1.0,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            projector_kwargs=projector_kwargs,
            normalized_grad=False,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriGradCos(DattriTracInCP):
    """Grad-Cos wrapper using dattri's `TracInAttributor`.

    Grad-Cos is TracIn with a single unit weight and cosine-normalized
    gradients: it computes the cosine similarity of train/test gradients at a
    single checkpoint.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriGradCos` explainer."""
        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            learning_rate=1.0,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            projector_kwargs=projector_kwargs,
            normalized_grad=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriArnoldi(DattriInfluence):
    """Wrapper for `IFAttributorArnoldi` from dattri.

    This implements the Arnoldi-iteration based influence function approach
    of Schioppa et al. (2022).

    References
    ----------
    (1) Schioppa, Andrea, et al. (2022). "Scaling up influence functions."
    Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36.
    No. 8.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        proj_dim: int = 100,
        max_iter: int = 100,
        norm_constant: float = 1.0,
        tol: float = 1e-7,
        regularization: float = 0.0,
        seed: int = 0,
        device: Optional[str] = None,
        precompute_data_ratio: float = 1.0,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriArnoldi` explainer."""
        logger.info("Initializing dattri Arnoldi explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "proj_dim": proj_dim,
            "max_iter": max_iter,
            "norm_constant": norm_constant,
            "tol": tol,
            "regularization": regularization,
            "seed": seed,
            "precompute_data_ratio": precompute_data_ratio,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorArnoldi,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriEKFAC(DattriInfluence):
    """Wrapper for `IFAttributorEKFAC` from dattri.

    References
    ----------
    (1) Grosse, Roger, et al. (2023). "Studying Large Language Model
    Generalization with Influence Functions." arXiv preprint
    arXiv:2308.03296.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        module_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        damping: float = 0.0,
        max_iter: Optional[int] = None,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriEKFAC` explainer."""
        logger.info("Initializing dattri EK-FAC explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "module_name": module_name,
            "damping": damping,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        # EK-FAC takes `module_name`, not `layer_name`.
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorEKFAC,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=None,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=False,
            device=device,
            hf_input_keys=hf_input_keys,
        )
        self.attributor.cache(
            self._make_loader(self.train_dataset),
            max_iter=max_iter,
        )


class DattriIFExplicit(DattriInfluence):
    """Wrapper for `IFAttributorExplicit` from dattri.

    Computes the influence function with the explicit Hessian inverse.
    Practical only on small models / small parameter subsets selected via
    ``layer_name``. Used in dattri's ``experiments/gpt2_wikitext``.

    References
    ----------
    (1) Koh, Pang Wei, and Percy Liang. (2017). "Understanding black-box
    predictions via influence functions." International Conference on
    Machine Learning. PMLR.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        regularization: float = 0.0,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriIFExplicit` explainer."""
        logger.info("Initializing dattri IF-Explicit explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "regularization": regularization,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorExplicit,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriIFCG(DattriInfluence):
    """Wrapper for `IFAttributorCG` from dattri.

    References
    ----------
    (1) Martens, James. (2010). "Deep learning via Hessian-free
    optimization." Proceedings of the 27th International Conference on
    Machine Learning. (735-742).

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        max_iter: int = 10,
        tol: float = 1e-7,
        mode: str = "rev-rev",
        regularization: float = 0.0,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriIFCG` explainer."""
        logger.info("Initializing dattri IF-CG explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "max_iter": max_iter,
            "tol": tol,
            "mode": mode,
            "regularization": regularization,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorCG,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriIFLiSSA(DattriInfluence):
    """Wrapper for `IFAttributorLiSSA` from dattri.

    Stochastic Hessian-inverse approximation via the LiSSA algorithm of
    Agarwal et al. (2017). Used in dattri's ``experiments/gpt2_wikitext``.

    References
    ----------
    (1) Agarwal, Naman, Brian Bullins, and Elad Hazan. (2017). "Second-order
    stochastic optimization for machine learning in linear time." Journal
    of Machine Learning Research 18. (1-40).

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        lissa_batch_size: int = 1,
        num_repeat: int = 1,
        recursion_depth: int = 5000,
        damping: float = 0.0,
        scaling: float = 50.0,
        mode: str = "rev-rev",
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriIFLiSSA` explainer."""
        logger.info("Initializing dattri IF-LiSSA explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "batch_size": lissa_batch_size,
            "num_repeat": num_repeat,
            "recursion_depth": recursion_depth,
            "damping": damping,
            "scaling": scaling,
            "mode": mode,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorLiSSA,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


class DattriIFDataInf(DattriInfluence):
    """Wrapper for `IFAttributorDataInf` from dattri.

    Layer-wise empirical-Fisher influence approximation of Kwon et al.
    (2024).

    References
    ----------
    (1) Kwon, Yongchan, et al. (2024). "DataInf: Efficiently Estimating Data
    Influence in LoRA-tuned LLMs and Diffusion Models." International
    Conference on Learning Representations.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        regularization: float = 0.0,
        fim_estimate_data_ratio: float = 1.0,
        device: Optional[str] = None,
        hf_input_keys: Optional[Sequence[str]] = None,
    ):
        """Initialize the `DattriIFDataInf` explainer."""
        logger.info("Initializing dattri IF-DataInf explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "regularization": regularization,
            "fim_estimate_data_ratio": fim_estimate_data_ratio,
        }

        if isinstance(checkpoints, list) and len(checkpoints) > 1:
            checkpoints = [checkpoints[-1]]

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            attributor_cls=IFAttributorDataInf,
            attributor_kwargs=attributor_kwargs,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            use_cache=True,
            device=device,
            hf_input_keys=hf_input_keys,
        )


__all__ = [
    "DattriInfluence",
    "DattriTRAK",
    "DattriTracInCP",
    "DattriArnoldi",
    "DattriEKFAC",
    "DattriGradDot",
    "DattriGradCos",
    "DattriIFExplicit",
    "DattriIFCG",
    "DattriIFLiSSA",
    "DattriIFDataInf",
]
