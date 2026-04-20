"""Wrappers for the dattri influence computation methods."""

import logging
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import datasets as hf_datasets  # type: ignore
import lightning as L
import torch
from dattri.algorithm.influence_function import (  # type: ignore
    IFAttributorArnoldi,
    IFAttributorEKFAC,
)
from dattri.algorithm.tracin import TracInAttributor  # type: ignore
from dattri.algorithm.trak import TRAKAttributor  # type: ignore
from dattri.task import AttributionTask  # type: ignore

from quanda.explainers.base import Explainer
from quanda.utils.common import process_targets
from quanda.utils.datasets.dataset_handlers import (
    DatasetHandler,
    TorchDatasetHandler,
    get_dataset_handler,
)
from quanda.utils.tasks import TaskLiterals

logger = logging.getLogger(__name__)


_NUMERIC_DTYPES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "bool",
)


def _is_numeric_feature(feat: Any) -> bool:
    """Return True if an HF dataset feature is stored as numeric tensors."""
    if isinstance(feat, hf_datasets.Value):
        return feat.dtype in _NUMERIC_DTYPES
    if isinstance(feat, hf_datasets.Sequence):
        return _is_numeric_feature(feat.feature)
    if isinstance(feat, list) and feat:
        return _is_numeric_feature(feat[0])
    return False


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
        device: str = "cpu",
    ):
        """Initialize the base `DattriInfluence` wrapper.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        loss_func : Callable
            The loss function for the dattri `AttributionTask`. Signature
            must follow dattri's expectations for the specific attributor.
        attributor_cls : type
            The dattri attributor class.
        attributor_kwargs : Dict[str, Any]
            Keyword arguments passed to the dattri attributor constructor.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
        target_func : Optional[Callable], optional
            Target function for the dattri `AttributionTask`. If None, loss
            function is used. Defaults to None.
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
            Device to run the computation on. Defaults to "cpu".

        """
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

        dattri_checkpoints = (
            self.checkpoints if self.checkpoints else model.state_dict()
        )

        task_kwargs: Dict[str, Any] = {}
        if checkpoints_load_func is not None:
            task_kwargs["checkpoints_load_func"] = checkpoints_load_func
        if target_func is not None:
            task_kwargs["target_func"] = target_func

        self.attribution_task = AttributionTask(
            loss_func=loss_func,
            model=model,
            checkpoints=dattri_checkpoints,
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
            self.attributor.cache(self._make_loader(self.train_dataset))

    def _make_loader(
        self,
        dataset: Any,
        batch_size: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """Build a DataLoader using the matching quanda dataset handler."""
        bs = batch_size if batch_size is not None else self.batch_size
        if self.collate_fn is not None:
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
        handler = self._get_handler(dataset)
        return handler.create_dataloader(
            dataset=dataset, batch_size=bs, shuffle=False
        )

    @staticmethod
    def _get_handler(dataset: Any) -> DatasetHandler:
        """Return a quanda dataset handler for the given dataset.

        Falls back to ``TorchDatasetHandler`` for custom ``Dataset`` types
        (e.g., ``TensorDataset`` subsets) that ``get_dataset_handler`` may
        not match directly.
        """
        try:
            return get_dataset_handler(dataset)
        except ValueError:
            if isinstance(dataset, torch.utils.data.Dataset):
                return TorchDatasetHandler()
            raise

    def _create_test_dataset(
        self,
        test_data: Any,
        targets: Optional[Union[List[int], torch.Tensor]],
    ) -> Any:
        """Turn `(test_data, targets)` into a dataset consumable by dattri."""
        if isinstance(test_data, torch.utils.data.Dataset):
            return test_data
        if isinstance(test_data, torch.Tensor):
            if targets is None:
                raise ValueError("Targets required for tensor test_data.")
            targets = process_targets(targets, self.device)
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            return torch.utils.data.TensorDataset(
                test_data.to(self.device), targets.to(self.device)
            )
        if isinstance(test_data, list):
            first = test_data[0]
            if isinstance(first, dict):
                data = {key: [d[key] for d in test_data] for key in first}
                if targets is not None:
                    data["labels"] = (
                        targets
                        if isinstance(targets, list)
                        else targets.tolist()
                    )
                ds = hf_datasets.Dataset.from_dict(data)
                return self._set_torch_format(ds)
            if isinstance(first, (tuple, list)):
                stacked = [
                    torch.stack([torch.as_tensor(d[i]) for d in test_data])
                    for i in range(len(first))
                ]
                return torch.utils.data.TensorDataset(*stacked)
        if isinstance(test_data, hf_datasets.Dataset):
            return self._set_torch_format(test_data)
        raise ValueError(
            f"Unsupported test_data type: {type(test_data)}. "
            "Expected torch.Tensor, List[dict], List[tuple], "
            "torch.utils.data.Dataset, or datasets.Dataset."
        )

    @staticmethod
    def _set_torch_format(ds: hf_datasets.Dataset) -> hf_datasets.Dataset:
        """Restrict HF dataset format to its numeric (tensor-safe) columns.

        HF datasets often carry raw string/object columns (e.g., original
        text fields) alongside the tokenized tensor columns. Passing those
        into dattri's vmap-based pipeline fails, so we restrict the dataset
        format to columns with numeric feature types.
        """
        numeric_cols = [
            name
            for name, feat in ds.features.items()
            if _is_numeric_feature(feat)
        ]
        ds.set_format(type="torch", columns=numeric_cols)
        return ds

    def _prepare_train_dataset(self) -> Any:
        """Return the training dataset in a dattri-friendly format."""
        train_dataset = self.train_dataset
        if isinstance(train_dataset, hf_datasets.Dataset):
            return self._set_torch_format(train_dataset)
        return train_dataset

    def _call_attribute(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """Invoke the attributor's attribute method."""
        return self.attributor.attribute(train_loader, test_loader)

    def explain(
        self,
        test_data: Any,
        targets: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : Any
            Test samples for which influence scores are computed. Supported
            types: ``torch.Tensor``, list of dicts, list of tuples,
            ``torch.utils.data.Dataset``, or ``datasets.Dataset``.
        targets : Optional[Union[List[int], torch.Tensor]], optional
            Labels for the test samples. Required when ``test_data`` is a
            bare tensor; ignored when ``test_data`` already carries labels.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size).

        """
        test_dataset = self._create_test_dataset(test_data, targets)
        test_loader = self._make_loader(test_dataset)
        train_loader = self._make_loader(self._prepare_train_dataset())
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
        train_loader = self._make_loader(
            self._prepare_train_dataset(), batch_size=batch_size
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
    (1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
        and Aleksander Madry. (2023). "TRAK: attributing model behavior at
        scale". ICML.

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
        device: str = "cpu",
        use_cache: bool = True,
    ):
        """Initialize the `DattriTRAK` explainer."""
        logger.info("Initializing dattri TRAK explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "correct_probability_func": correct_probability_func,
            "regularization": regularization,
        }
        if projector_kwargs is not None:
            attributor_kwargs["projector_kwargs"] = projector_kwargs

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
        )

    def _call_attribute(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        # dattri TRAK: attribute(test_loader, train_loader) → (n_train, n_test)
        if self.attributor.full_train_dataloader is not None:
            return self.attributor.attribute(test_loader)
        return self.attributor.attribute(test_loader, train_loader)

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """Compute TRAK self-influence scores."""
        if self.attributor.full_train_dataloader is not None:
            scores = self.attributor.self_attribute()
        else:
            train_loader = self._make_loader(
                self._prepare_train_dataset(), batch_size=batch_size
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
        by tracing gradient descent." NeurIPS.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        loss_func: Callable,
        weight_list: torch.Tensor,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        layer_name: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1,
        collate_fn: Optional[Callable] = None,
        projector_kwargs: Optional[Dict[str, Any]] = None,
        normalized_grad: bool = False,
        device: str = "cpu",
    ):
        """Initialize the `DattriTracInCP` explainer."""
        logger.info("Initializing dattri TracInCP explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "weight_list": weight_list,
            "normalized_grad": normalized_grad,
        }
        if projector_kwargs is not None:
            attributor_kwargs["projector_kwargs"] = projector_kwargs

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
        device: str = "cpu",
    ):
        """Initialize the `DattriGradDot` explainer."""
        n_ckpts = len(checkpoints) if isinstance(checkpoints, list) else 1
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            weight_list=torch.ones(n_ckpts),
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            projector_kwargs=projector_kwargs,
            normalized_grad=False,
            device=device,
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
        device: str = "cpu",
    ):
        """Initialize the `DattriGradCos` explainer."""
        n_ckpts = len(checkpoints) if isinstance(checkpoints, list) else 1
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            loss_func=loss_func,
            weight_list=torch.ones(n_ckpts),
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            layer_name=layer_name,
            batch_size=batch_size,
            collate_fn=collate_fn,
            projector_kwargs=projector_kwargs,
            normalized_grad=True,
            device=device,
        )


class DattriArnoldi(DattriInfluence):
    """Wrapper for `IFAttributorArnoldi` from dattri.

    This implements the Arnoldi-iteration based influence function approach
    of Schioppa et al. (2022).

    References
    ----------
    (1) Schioppa, Andrea, et al. (2022). "Scaling up influence functions."
        AAAI.

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
        device: str = "cpu",
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
        }

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
        )


class DattriEKFAC(DattriInfluence):
    """Wrapper for `IFAttributorEKFAC` from dattri.

    This implements the EK-FAC inverse-FIM approximation of George et al.
    (2018) for influence function estimation.

    References
    ----------
    (1) George, Thomas, et al. (2018). "Fast approximate natural gradient
        descent in a Kronecker-factored eigenbasis." NeurIPS.

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
        device: str = "cpu",
    ):
        """Initialize the `DattriEKFAC` explainer."""
        logger.info("Initializing dattri EK-FAC explainer...")

        attributor_kwargs: Dict[str, Any] = {
            "module_name": module_name,
            "damping": damping,
        }

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
        )
        self.attributor.cache(
            self._make_loader(self._prepare_train_dataset()),
            max_iter=max_iter,
        )


__all__ = [
    "DattriInfluence",
    "DattriTRAK",
    "DattriTracInCP",
    "DattriArnoldi",
    "DattriEKFAC",
    "DattriGradDot",
    "DattriGradCos",
]
