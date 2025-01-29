"""Base class for all benchmarks."""

import os
import zipfile
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Any

import requests
import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
import lightning as L

from quanda.benchmarks.resources import (
    benchmark_urls,
    sample_transforms,
    load_module_from_bench_state,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import get_load_state_dict_func, load_last_checkpoint
from quanda.utils.datasets.image_datasets import (
    HFtoTV,
    SingleClassImageDataset,
)

from quanda.utils.training import BaseTrainer


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str
    eval_args: List = []

    def __init__(self, *args, **kwargs):
        """Initialize the base `Benchmark` class."""
        self.device: Union[str, torch.device]
        self.bench_state: dict
        self._checkpoint_paths: Optional[List[str]] = None
        self._checkpoints_load_func: Optional[Callable[..., Any]] = None
        self._checkpoints: Optional[Union[str, List[str]]] = None

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """Generate the benchmark by specifying parameters.

        The evaluation can then be run using the `evaluate` method.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

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

        Returns
        -------
        Benchmark
            The benchmark instance.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        return obj._parse_bench_state(bench_state, cache_dir, device=device)

    def _get_bench_state(
        self,
        name: str,
        cache_dir: str,
        device: str,
    ):
        """Download a benchmark state dictionary of a benchmark and returns.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components
        device : str
            Device to use with the benchmark components.

        Returns
        -------
        dict
            Benchmark state dictionary.

        """
        os.makedirs(cache_dir, exist_ok=True)
        # check if file exists
        if not os.path.exists(os.path.join(cache_dir, name + ".pth")):
            url = benchmark_urls[name]

            # download to cache_dir
            response = requests.get(url)

            with open(os.path.join(cache_dir, name + ".pth"), "wb") as f:
                f.write(response.content)

        return torch.load(
            os.path.join(cache_dir, name + ".pth"),
            map_location=device,
            weights_only=True,
        )

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """Assembles the benchmark from existing components.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _assemble_common(
        self,
        model: torch.nn.Module,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        use_predictions: bool = True,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.

        Returns
        -------
        None

        """
        self.model = model
        self._set_devices(model)
        self.eval_dataset = eval_dataset
        self.checkpoints = checkpoints
        self.checkpoints_load_func = checkpoints_load_func
        self.use_predictions = use_predictions

    @abstractmethod
    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Run the evaluation using the benchmark.

        Parameters
        ----------
        explainer_cls : type
            The explainer class to be used for evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the explainer, by
            default None.
        batch_size : int, optional
            Batch size for the evaluation, by default 8.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _set_devices(
        self,
        model: torch.nn.Module,
    ):
        """Infer device from model.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.

        """
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    def _process_dataset(
        cls,
        dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
    ) -> torch.utils.data.Dataset:
        """Return the dataset using the given parameters.

        Parameters
        ----------
        dataset : Union[str, torch.utils.data.Dataset]
            The dataset to be processed.
        transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, by default "train", only used for HuggingFace
            datasets.

        Returns
        -------
        torch,utils.data.Dataset
            The dataset.

        """
        cache_dir = os.getenv(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets")
        )

        if isinstance(dataset, str):
            cls.dataset_str = dataset
            return HFtoTV(
                load_dataset(
                    dataset,
                    name="mnist",
                    split=dataset_split,
                    cache_dir=cache_dir,
                ),
                transform=transform,
            )  # TODO: remove name="mnist"
        else:
            return dataset

    def _build_eval_dataset(
        self,
        dataset_str: str,
        eval_indices: List[int],
        transform: Optional[Callable] = None,
        dataset_split: str = "test",
    ):
        """Download the HuggingFace evaluation dataset from given name.

        Parameters
        ----------
        dataset_str : str
            The name of the HuggingFace dataset.
        eval_indices : List[int]
            The indices to be used for evaluation.
        transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, by default "test".

        Returns
        -------
        torch.utils.data.Dataset
            The evaluation dataset.

        """
        cache_dir = os.getenv(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface/datasets")
        )
        test_dataset = HFtoTV(
            load_dataset(
                dataset_str,
                name="mnist",
                split=dataset_split,
                cache_dir=cache_dir,
            ),  # TODO: remove name="mnist"
            transform=transform,
        )
        return torch.utils.data.Subset(test_dataset, eval_indices)

    def get_checkpoint_paths(self) -> List[str]:
        """Return the paths to the checkpoints."""
        assert self._checkpoint_paths is not None, (
            "get_checkpoint_paths can only be called after instantiating a "
            "benchmark using the download method."
        )
        return self._checkpoint_paths

    @property
    def checkpoints_load_func(self):
        """Return the function to load the checkpoints."""
        return self._checkpoints_load_func

    @checkpoints_load_func.setter
    def checkpoints_load_func(self, value):
        """Set the function to load the checkpoints."""
        if self.device is None:
            raise ValueError(
                "The device must be set before setting the "
                "checkpoints_load_func."
            )
        if value is None:
            self._checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self._checkpoints_load_func = value

    @property
    def checkpoints(self):
        """Return the checkpoint paths."""
        return self._checkpoints

    @checkpoints.setter
    def checkpoints(self, value):
        """Set the checkpoint paths."""
        if value is None:
            self._checkpoints = []
        else:
            self._checkpoints = value if isinstance(value, List) else [value]

    def _parse_bench_state(
        self,
        bench_state: dict,
        cache_dir: str,
        model_id: Optional[str] = None,
        device: str = "cpu",
    ):
        """Parse the benchmark state dictionary."""
        # TODO: this should be further refactored after the pipeline is done.
        # TODO: fix this mess.
        checkpoint_paths = []

        assemble_dict = {}

        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        dataset_transform_str = bench_state.get("dataset_transform", None)
        dataset_transform = sample_transforms.get(dataset_transform_str, None)
        sample_fn_str = bench_state.get("sample_fn", None)
        sample_fn = sample_transforms.get(sample_fn_str, None)

        eval_dataset = self._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=dataset_transform
            if self.name != "Shortcut Detection"
            else None,  # TODO: better way to handle this
            dataset_split=bench_state.get("test_split_name", "test"),
        )

        if self.name == "Mixed Datasets":
            adversarial_dir_url = bench_state["adversarial_dir_url"]
            adversarial_dir = self._download_adversarial_dataset(
                adversarial_dir_url=adversarial_dir_url,
                cache_dir=cache_dir,
            )

            adversarial_transform = sample_transforms[
                bench_state["adversarial_transform"]
            ]
            adv_test_indices = bench_state["adv_indices_test"]
            eval_from_test_indices = bench_state["eval_test_indices"]
            eval_indices = [
                adv_test_indices[i] for i in eval_from_test_indices
            ]

            eval_dataset = SingleClassImageDataset(
                root=adversarial_dir,
                label=bench_state["adversarial_label"],
                transform=adversarial_transform,
                indices=eval_indices,
            )

            adv_train_indices = bench_state["adv_indices_train"]
            assemble_dict["adversarial_dir"] = adversarial_dir
            assemble_dict["adv_train_indices"] = adv_train_indices
            assemble_dict["adversarial_transform"] = adversarial_transform

        module = load_module_from_bench_state(bench_state, device)

        # check the type of the instance self

        assemble_dict["model"] = module
        assemble_dict["checkpoints"] = bench_state["checkpoints_binary"]
        assemble_dict["checkpoints_load_func"] = bench_load_state_dict
        assemble_dict["train_dataset"] = bench_state["dataset_str"]
        assemble_dict["base_dataset"] = bench_state[
            "dataset_str"
        ]  # TODO: rename dataset_str to base/train_dataset_str
        assemble_dict["eval_dataset"] = eval_dataset
        assemble_dict["use_predictions"] = bench_state["use_predictions"]
        assemble_dict["checkpoint_paths"] = checkpoint_paths
        assemble_dict["dataset_transform"] = dataset_transform
        assemble_dict["sample_fn"] = sample_fn
        assemble_dict["cache_dir"] = cache_dir
        assemble_dict["model_id"] = model_id

        for el in [
            "n_classes",
            "mislabeling_labels",
            "adversarial_label",
            "global_method",
            "shortcut_indices",
            "shortcut_cls",
            "class_to_group",
        ]:
            if el in bench_state:
                assemble_dict[el] = bench_state[el]

        return self.assemble(**assemble_dict)

    def _evaluate_dataset(
        self,
        eval_dataset: torch.utils.data.Dataset,
        explainer: Explainer,
        metric: Metric,
        batch_size: int,
    ):
        expl_dl = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size
        )

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description(
                "Metric evaluation, batch %d/%d" % (i + 1, n_batches)
            )

            input, labels = input.to(self.device), labels.to(self.device)

            if self.use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            explanations = explainer.explain(
                test_data=input,
                targets=targets,
            )
            data_unit = {
                "test_data": input,
                "test_targets": targets,
                "test_labels": labels,
                "explanations": explanations,
            }

            if hasattr(self, "class_to_group"):
                data_unit["grouped_labels"] = torch.tensor(
                    [self.class_to_group[i.item()] for i in labels],
                    device=labels.device,
                )
                if not self.use_predictions:
                    data_unit["targets"] = data_unit["grouped_labels"]

            eval_unit = {k: data_unit[k] for k in self.eval_args}
            metric.update(**eval_unit)

        return metric.compute()

    def _prepare_explainer(
        self,
        dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
    ):
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
            train_dataset=dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )
        return explainer

    def _train_model(
        self,
        model: torch.nn.Module,
        trainer: Union[L.Trainer, BaseTrainer],
        train_dataset: torch.utils.data.Dataset,
        save_dir: str,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        if val_dataset:
            val_dl = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            val_dl = None

        model.train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )

            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError(
                "Trainer should be a Lightning Trainer or a BaseTrainer"
            )

        # save check point to cache_dir
        # TODO: add model id
        torch.save(
            model.state_dict(),
            save_dir,
        )

        model.to(self.device)
        model.eval()

        return model

    def _download_adversarial_dataset(
        self, adversarial_dir_url: str, cache_dir: str
    ):
        """Download the adversarial dataset.

        Download the adversarial dataset from the given URL and returns the
        path to the downloaded directory.

        Parameters
        ----------
        adversarial_dir_url: str
            URL to the adversarial dataset.
        cache_dir: str
            Path to the cache directory.

        Returns
        -------
        str
            Path to the downloaded adversarial dataset directory.

        """
        name = self.name.replace(" ", "_").lower()
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
