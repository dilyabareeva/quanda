"""Mixed Datasets benchmark module."""

import copy
import logging
import os
import zipfile
from typing import Callable, List, Optional, Union, Any

import lightning as L
import requests
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.common import ds_len, load_last_checkpoint
from quanda.utils.datasets import SingleClassImageDataset
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MixedDatasets(Benchmark):
    # TODO: remove FILTER BY "CORRECT" PREDICTION FOR BACKDOOR implied
    #  https://arxiv.org/pdf/2201.10055
    """Mixed Datasets Benchmark.

    Evaluates the performance of a given data attribution estimation method in
    identifying adversarial examples in a classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial"
    subsets, whereby the number of samples in the clean dataset is
    significantly larger than the number of samples in the adversarial dataset.
    All adversarial samples are labeled with one label from the clean dataset.
    The evaluation is based on the area under the precision-recall curve
    (AUPRC), which quantifies the ranking of the influence of adversarial
    relative to clean samples. AUPRC is chosen because it provides better
    insight into performance in highly-skewed classification tasks where
    false positives are common.

    Unlike the original implementation, we only employ a single trained model,
    but we aggregate the AUPRC scores across
    multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's
    target using renormalized influence estimation. In Proceedings of the 2022
    ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).

    """

    name: str = "Mixed Datasets"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Mixed Datasets benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.mixed_dataset: torch.utils.data.Dataset
        self.adversarial_indices: List[int]
        self.use_predictions: bool
        self.adversarial_label: int
        self.filter_by_prediction: bool
        self.cache_dir: str

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        adversarial_dir: str,
        adversarial_label: int,
        adv_train_indices: List[int],
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        data_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        adversarial_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generate the benchmark with passed components.

        This module handles the dataset creation and model training on the
        mixed dataset. The evaluation can then be run using the `evaluate`
        method.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        base_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed,
            it should be a HuggingFace dataset. If a torch Dataset is passed,
            every item of the dataset is a tuple of the form (input, label).
        eval_dataset: torch.utils.data.Dataset
            The dataset containing the adversarial examples used for
            evaluation. They should belong to the same dataset and the same
            class as the samples in the adversarial dataset.
        adversarial_dir: str
            Path to directory containing the adversarial dataset. Typically
            consists of the same class of objects (e.g. images of the same
            class).
        adversarial_label: int
            The label to be used for the adversarial dataset.
        adv_train_indices: List[int]
            List of indices of the adversarial dataset used for training.
        trainer: Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning
            Trainer or a `BaseTrainer`.
        cache_dir: str
            Directory to store the generated benchmark.
        data_transform: Optional[Callable], optional
            Transform to be applied to the clean dataset, by default None.
        use_predictions: bool, optional
            Whether to use the model's predictions for generating attributions.
            Defaults to True.
        filter_by_prediction: bool, optional
            Whether to filter the adversarial examples to only use correctly
            predicted test samples. Defaults to True.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        adversarial_transform : Optional[Callable], optional
             Transform to be applied to the adversarial dataset, by default
             None.
        val_dataset: Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None.
        trainer_fit_kwargs: Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by
            default None.
        batch_size: int, optional
            Batch size that is used for training, by default 8
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MixedDatasets
            The benchmark instance.

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
        logger.info(
            f"Generating {MixedDatasets.name} benchmark components based on "
            f"passed arguments..."
        )

        obj = cls()
        obj.cache_dir = cache_dir
        obj._set_devices(model)
        # this sets the function to the default value
        obj.checkpoints_load_func = None

        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.adversarial_label = adversarial_label
        obj.filter_by_prediction = filter_by_prediction

        pr_base_dataset = obj._process_dataset(
            base_dataset, transform=data_transform, dataset_split=dataset_split
        )

        adversarial_dataset = SingleClassImageDataset(
            root=adversarial_dir,
            label=adversarial_label,
            transform=adversarial_transform,
            indices=adv_train_indices,
        )

        obj.mixed_dataset = torch.utils.data.ConcatDataset(
            [adversarial_dataset, pr_base_dataset]
        )
        obj.adversarial_indices = [1] * ds_len(adversarial_dataset) + [
            0
        ] * ds_len(pr_base_dataset)

        mixed_train_dl = torch.utils.data.DataLoader(
            obj.mixed_dataset, batch_size=batch_size
        )

        if val_dataset is not None:
            val_dl = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            val_dl = None

        obj.model = copy.deepcopy(model).train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(obj.model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=obj.model,
                train_dataloaders=mixed_train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(obj.model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )

            trainer.fit(
                model=obj.model,
                train_dataloaders=mixed_train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError(
                "Trainer should be a Lightning Trainer or a BaseTrainer"
            )

        torch.save(
            obj.model.state_dict(),
            os.path.join(cache_dir, "model_mixed_datasets.pth"),
        )
        obj.checkpoints = [
            os.path.join(cache_dir, "model_mixed_datasets.pth")
        ]  # TODO: save checkpoints
        obj.model.to(obj.device)
        obj.model.eval()
        return obj

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
        args : Any
            Additional positional arguments.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        MixedDatasets
            The benchmark instance.

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

        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        adversarial_dir_url = bench_state["adversarial_dir_url"]
        adversarial_dir = obj._download_adversarial_dataset(
            name=name,
            adversarial_dir_url=adversarial_dir_url,
            cache_dir=cache_dir,
        )

        adversarial_transform = sample_transforms[
            bench_state["adversarial_transform"]
        ]
        adv_test_indices = bench_state["adv_indices_test"]
        eval_from_test_indices = bench_state["eval_test_indices"]
        eval_indices = [adv_test_indices[i] for i in eval_from_test_indices]

        eval_dataset = SingleClassImageDataset(
            root=adversarial_dir,
            label=bench_state["adversarial_label"],
            transform=adversarial_transform,
            indices=eval_indices,
        )

        adv_train_indices = bench_state["adv_indices_train"]

        return obj.assemble(
            model=module,
            checkpoints=bench_state["checkpoints_binary"],
            checkpoints_load_func=bench_load_state_dict,
            base_dataset=bench_state["dataset_str"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            adversarial_dir=adversarial_dir,
            adversarial_label=bench_state["adversarial_label"],
            adversarial_transform=adversarial_transform,
            adv_train_indices = adv_train_indices,
            data_transform=dataset_transform,
            checkpoint_paths=checkpoint_paths,
        )

    @staticmethod
    def _download_adversarial_dataset(
        name: str, adversarial_dir_url: str, cache_dir: str
    ):
        """Download the adversarial dataset.

        Download the adversarial dataset from the given URL and returns the
        path to the downloaded directory.

        Parameters
        ----------
        name: str
            Name of the benchmark.
        adversarial_dir_url: str
            URL to the adversarial dataset.
        cache_dir: str
            Path to the cache directory.

        Returns
        -------
        str
            Path to the downloaded adversarial dataset directory.

        """
        # Download the zip file and extract into cache dir
        adversarial_dir = os.path.join(
            cache_dir, name + "_adversarial_dataset"
        )
        os.makedirs(adversarial_dir, exist_ok=True)

        # download
        adversarial_dir_zip = os.path.join(
            adversarial_dir, "adversarial_dataset.zip"
        )
        with open(adversarial_dir_zip, "wb") as f:
            response = requests.get(adversarial_dir_url)
            f.write(response.content)

        # extract
        with zipfile.ZipFile(adversarial_dir_zip, "r") as zip_ref:
            zip_ref.extractall(adversarial_dir)

        return adversarial_dir

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        eval_dataset: torch.utils.data.Dataset,
        base_dataset: torch.utils.data.Dataset,
        adversarial_dir: str,
        adversarial_label: int,
        adv_train_indices: List[int],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        data_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        adversarial_transform: Optional[Callable] = None,
        dataset_split: str = "train",
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from the given components.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        eval_dataset: torch.utils.data.Dataset
            The dataset containing the adversarial examples used for
            evaluation. They should belong to the same dataset and the same
            class as the samples in the adversarial dataset.
        base_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed,
            it should be a HuggingFace dataset.
        adversarial_dir: str
            Path to the adversarial dataset of a single class.
        adversarial_label: int
            The label to be used for the adversarial dataset.
        adv_train_indices: List[int]
            List of indices of the adversarial dataset used for training.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        data_transform: Optional[Callable], optional
            Transform to be applied to the clean dataset, by default None.
        use_predictions: bool, optional
            Whether to use the model's predictions for generating attributions.
            Defaults to True.
        filter_by_prediction: bool, optional
            Whether to filter the adversarial examples to only use correctly
            predicted test samples. Defaults to True.
        adversarial_transform: Optional[Callable], optional
            Transform to be applied to the adversarial dataset, by default
            None.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MixedDatasets
            The benchmark instance.

        """
        obj = cls()
        obj.model = model
        obj._set_devices(model)
        obj.checkpoints = checkpoints
        obj.checkpoints_load_func = checkpoints_load_func
        obj.base_dataset = obj._process_dataset(
            base_dataset, transform=data_transform, dataset_split=dataset_split
        )
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction
        obj.adversarial_label = adversarial_label

        adversarial_dataset = SingleClassImageDataset(
            root=adversarial_dir,
            label=adversarial_label,
            transform=adversarial_transform,
            indices=adv_train_indices,
        )

        obj.mixed_dataset = torch.utils.data.ConcatDataset(
            [adversarial_dataset, obj.base_dataset]
        )
        obj.adversarial_indices = [1] * ds_len(adversarial_dataset) + [
            0
        ] * ds_len(obj.base_dataset)

        obj._checkpoint_paths = checkpoint_paths

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Evaluate the benchmark using a given explanation method.

        Parameters
        ----------
        explainer_cls: type
            The explanation class inheriting from the base Explainer class to
            be used for evaluation.
        expl_kwargs: Optional[dict], optional
            Keyword arguments for the explainer, by default None.
        batch_size: int, optional
            Batch size for the evaluation, by default 8.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

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
            train_dataset=self.mixed_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )

        adversarial_expl_dl = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=batch_size
        )

        metric = MixedDatasetsMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.mixed_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            adversarial_indices=self.adversarial_indices,
            filter_by_prediction=self.filter_by_prediction,
            adversarial_label=self.adversarial_label,
        )

        pbar = tqdm(adversarial_expl_dl)
        n_batches = len(adversarial_expl_dl)

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
                explanations=explanations,
                test_tensor=inputs,
                test_labels=labels,
            )

        return metric.compute()
