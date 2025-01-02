"""Benchmark for noisy label detection."""

import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Union, Any

import lightning as L
import torch
import torch.utils
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.common import load_last_checkpoint
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MislabelingDetection(Benchmark):
    # TODO: remove ALL PAPERS USE SELF-INFLUENCE? OTHERWISE WE CAN USE
    #  PREDICTIONS
    """Benchmark for noisy label detection.

    This benchmark generates a dataset with mislabeled samples, and trains a
    model on it. Afterward, it evaluates the effectiveness of a given data
    attributor for detecting the mislabeled examples using
    ´quanda.metrics.downstream_eval.MislabelingDetectionMetric´.

    This is done by computing a cumulative detection curve (as described in the
    below references) and calculating the AUC following Kwon et al. (2024).

    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
    influence functions. In International Conference on Machine Learning
    (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018).
    Representer point selection for explaining deep neural networks. In
    Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating
    training data influence by tracing gradient descent. In Advances in Neural
    Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes,
    J.-M., & Pauwels, E. (2022). Leveraging influence functions for dataset
    exploration and cleaning. In 11th European Congress on Embedded Real-Time
    Systems (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Y., Wu, E., Wu, K., & Zou, J. (2024). DataInf: Efficiently
    estimating data influence in LoRA-tuned LLMs and diffusion models. In The
    Twelfth International Conference on Learning Representations (pp. 1-8).

    """

    name: str = "Mislabeling Detection"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Mislabeling Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: Optional[torch.utils.data.Dataset]
        self.mislabeling_dataset: LabelFlippingDataset
        self.dataset_transform: Optional[Callable]
        self.mislabeling_indices: List[int]
        self.mislabeling_labels: Dict[int, int]
        self.mislabeling_train_dl: torch.utils.data.DataLoader
        self.mislabeling_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.p: float
        self.global_method: Union[str, type] = "self-influence"
        self.n_classes: int
        self.use_predictions: bool

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        use_predictions: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        global_method: Union[str, type] = "self-influence",
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generate the benchmark by specifying parameters.

        This module handles the dataset creation and model training on the
        label-poisoned dataset. The evaluation can then be run using the
        `evaluate` method.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
            Note that a new model will be trained on the label-poisoned
            dataset.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Vanilla training dataset to be used for the benchmark. If a string
            is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning
            Trainer or a `BaseTrainer`.
        cache_dir : str
            Directory to store the generated benchmark components.
        eval_dataset : Optional[torch.utils.data.Dataset]
            Dataset to be used for the evaluation.
            This is only used if `global_method` is not "self-influence", by
            default None.
            Original papers use the self-influence method to reach a global
            ranking of the data,
            instead of using aggregations of generated local explanations.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation.
            This is only used if `global_method` is not "self-influence", by
            default True.
            Original papers use the self-influence method to reach a global
            ranking of the data,
            instead of using aggregations of generated local explanations.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of
            `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        p : float, optional
            The probability of mislabeling per sample, by default 0.3.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by
            default None.
        seed : int, optional
            Seed for reproducibility, by default 27.
        batch_size : int, optional
            Batch size that is used for training, by default 8.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MislabelingDetection
            The benchmark instance.

        """
        logger.info(
            f"Generating {MislabelingDetection.name} benchmark components "
            f"based on passed arguments..."
        )
        if global_method != "self-influence":
            assert eval_dataset is not None, (
                "MislabelingDetection should have "
                "global_method='self-influence' or eval_dataset should be "
                "given."
            )

        obj = cls()
        obj._set_devices(model)
        # this sets the function to the default value
        obj.checkpoints_load_func = None
        obj.base_dataset = obj._process_dataset(
            base_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj._generate(
            model=model,
            cache_dir=cache_dir,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            trainer=trainer,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        cache_dir: str,
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_transform: Optional[Callable],
        mislabeling_labels: Optional[Dict[int, int]] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, type] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
    ):
        """Generate the benchmark from components.

        This function is internally used for generating the benchmark instance.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
            Note that a new model will be trained on the label-poisoned
            dataset.
        cache_dir : str
            Directory to store the generated benchmark components.
        n_classes : int
            Number of classes in the dataset.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning
            Trainer or a `BaseTrainer`.
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        mislabeling_labels : Optional[Dict[int, int]], optional
            Optional dictionary containing indices as keys and new labels as
            values, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of
            `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        p : float, optional
            The probability of mislabeling per sample, by default 0.3.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by
            default None.
        seed : int, optional
            Seed for reproducibility, by default 27.
        batch_size : int, optional
            Batch size that is used for training, by default 8.

        Raises
        ------
        ValueError
            If the model is not a LightningModule and the trainer is a
            Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module and the trainer is a
            BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.

        """
        self.p = p
        self.global_method = global_method
        self.n_classes = n_classes
        self.dataset_transform = dataset_transform
        mislabeling_indices = (
            list(mislabeling_labels.keys())
            if mislabeling_labels is not None
            else None
        )

        self.mislabeling_dataset = LabelFlippingDataset(
            dataset=self.base_dataset,
            p=p,
            transform_indices=mislabeling_indices,
            dataset_transform=dataset_transform,
            mislabeling_labels=mislabeling_labels,
            n_classes=n_classes,
            seed=seed,
        )

        self.mislabeling_indices = self.mislabeling_dataset.transform_indices
        self.mislabeling_labels = self.mislabeling_dataset.mislabeling_labels
        self.mislabeling_train_dl = torch.utils.data.DataLoader(
            self.mislabeling_dataset, batch_size=batch_size
        )
        self.original_train_dl = torch.utils.data.DataLoader(
            self.base_dataset, batch_size=batch_size
        )
        if val_dataset:
            mislabeling_val_dataset = LabelFlippingDataset(
                dataset=val_dataset,
                dataset_transform=self.dataset_transform,
                p=self.p,
                n_classes=self.n_classes,
            )
            self.mislabeling_val_dl = torch.utils.data.DataLoader(
                mislabeling_val_dataset, batch_size=batch_size
            )
        else:
            self.mislabeling_val_dl = None

        self.model = copy.deepcopy(model).train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=self.model,
                train_dataloaders=self.mislabeling_train_dl,
                val_dataloaders=self.mislabeling_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )

            trainer.fit(
                model=self.model,
                train_dataloaders=self.mislabeling_train_dl,
                val_dataloaders=self.mislabeling_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError(
                "Trainer should be a Lightning Trainer or a BaseTrainer"
            )

        # save check point to cache_dir
        # TODO: add model id
        torch.save(
            self.model.state_dict(),
            os.path.join(cache_dir, "model_mislabeling_detection.pth"),
        )
        self.checkpoints = [
            os.path.join(cache_dir, "model_mislabeling_detection.pth")
        ]  # TODO: save checkpoints
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """Download a precomputed benchmark.

        Load precomputed benchmark components from a file and creates an
        instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        checkpoint_paths = []
        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        eval_dataset = obj._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split=bench_state["test_split_name"],
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            model=module,
            checkpoints=bench_state["checkpoints_binary"],
            checkpoints_load_func=bench_load_state_dict,
            base_dataset=bench_state["dataset_str"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            n_classes=bench_state["n_classes"],
            mislabeling_labels=bench_state["mislabeling_labels"],
            dataset_transform=dataset_transform,
            global_method=bench_state.get("global_method", "self-influence"),
            checkpoint_paths=checkpoint_paths,
        )

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        mislabeling_labels: Dict[int, int],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        use_predictions: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        global_method: Union[str, type] = "self-influence",
        batch_size: int = 8,
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark. This model should be trained on
            the mislabeled dataset.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is
            passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        mislabeling_labels : Dict[int, int]
            Dictionary containing indices as keys and new labels as values.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        eval_dataset : Optional[torch.utils.data.Dataset]
            Dataset to be used for the evaluation by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation.
            This is only used if `global_method` is not "self-influence",
            by default True. Original papers use the self-influence method to
            reach a global ranking of the data, instead of using aggregations
            of generated local explanations.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of
            `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        batch_size : int, optional
            Batch size that is used for training, by default 8.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MislabelingDetection
            The benchmark instance.

        """
        if global_method != "self-influence":
            assert eval_dataset is not None, (
                "MislabelingDetection should have "
                "global_method='self-influence' or eval_dataset should be "
                "given."
            )

        obj = cls()
        obj.model = model
        obj.checkpoints = checkpoints
        obj.base_dataset = obj._process_dataset(
            base_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        obj.dataset_transform = dataset_transform
        obj.global_method = global_method
        obj.n_classes = n_classes
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        mislabeling_indices = (
            list(mislabeling_labels.keys())
            if mislabeling_labels is not None
            else None
        )

        obj.mislabeling_dataset = LabelFlippingDataset(
            dataset=obj._process_dataset(
                base_dataset, transform=None, dataset_split=dataset_split
            ),
            dataset_transform=dataset_transform,
            transform_indices=mislabeling_indices,
            n_classes=n_classes,
            mislabeling_labels=mislabeling_labels,
        )

        obj.mislabeling_indices = obj.mislabeling_dataset.transform_indices
        obj.mislabeling_labels = obj.mislabeling_dataset.mislabeling_labels

        obj.mislabeling_train_dl = torch.utils.data.DataLoader(
            obj.mislabeling_dataset, batch_size=batch_size
        )
        obj.original_train_dl = torch.utils.data.DataLoader(
            obj.base_dataset, batch_size=batch_size
        )
        obj._checkpoint_paths = checkpoint_paths
        obj._set_devices(model)
        obj.checkpoints_load_func = checkpoints_load_func

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Evaluate the given data attributor.

        Parameters
        ----------
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None.
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 8.

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.mislabeling_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )

        if self.eval_dataset is not None:
            mislabeling_expl_ds = LabelFlippingDataset(
                dataset=self.eval_dataset,
                dataset_transform=self.dataset_transform,
                n_classes=self.n_classes,
                p=0.0,
            )
            expl_dl = torch.utils.data.DataLoader(
                mislabeling_expl_ds, batch_size=batch_size
            )
        if self.global_method != "self-influence":
            metric = MislabelingDetectionMetric.aggr_based(
                model=self.model,
                train_dataset=self.mislabeling_dataset,
                mislabeling_indices=self.mislabeling_indices,
                aggregator_cls=self.global_method,
            )

            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (inputs, labels) in enumerate(pbar):
                pbar.set_description(
                    "Metric evaluation, batch %d/%d" % (i + 1, n_batches)
                )

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if self.use_predictions:
                    with torch.no_grad():
                        targets = self.model(inputs).argmax(dim=-1)
                else:
                    targets = labels
                explanations = explainer.explain(
                    test_tensor=inputs, targets=targets
                )
                metric.update(
                    test_data=inputs,
                    test_labels=labels,
                    explanations=explanations,
                )
        else:
            metric = MislabelingDetectionMetric.self_influence_based(
                model=self.model,
                checkpoints=self.checkpoints,
                checkpoints_load_func=self.checkpoints_load_func,
                train_dataset=self.mislabeling_dataset,
                mislabeling_indices=self.mislabeling_indices,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )

        return metric.compute()
