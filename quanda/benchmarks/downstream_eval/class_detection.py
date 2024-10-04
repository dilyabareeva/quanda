import logging
import os
from typing import Callable, List, Optional, Union

import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import sample_transforms
from quanda.benchmarks.resources.modules import load_module_from_bench_state
from quanda.metrics.downstream_eval import ClassDetectionMetric

logger = logging.getLogger(__name__)


class ClassDetection(Benchmark):
    """
    Benchmark for class detection task.
    This benchmark evaluates the effectiveness of an attribution method in detecting the class of a test sample
    from its highest attributed training point.
    Intuitively, a good attribution method should assign the highest attribution to the class of the test sample,
    as argued in Hanawa et al. (2021) and Kwon et al. (2024).

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of similarity-based explanations.
    In International Conference on Learning Representations.

    2) Kwon, Y., Wu, E., Wu, K., Zou, J., (2024). DataInf: Efficiently Estimating Data Influence in
    LoRA-tuned LLMs and Diffusion Models. The Twelfth International Conference on Learning Representations.
    """

    # TODO: remove USES PREDICTED LABELS https://arxiv.org/pdf/2006.04528
    name: str = "Class Detection"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializer for the Class Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        data_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        Generates the benchmark by specifying parameters. The evaluation can then be run using the `evaluate` method.

        Parameters
        ----------
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train `model`. If a string is passed, it should be a HuggingFace dataset name.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        model : torch.nn.Module
            The model used to generate attributions.
        data_transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation. Original paper uses the model's predictions.
            Therefore, by default True.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".

        Returns
        -------
        ClassDetection
            The benchmark instance.
        """

        logger.info(f"Generating {ClassDetection.name} benchmark components based on passed arguments...")
        obj = cls()

        obj.model = model
        obj.eval_dataset = eval_dataset
        obj._set_devices(model)
        obj.train_dataset = obj._process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.use_predictions = use_predictions

        return obj

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        This method downloads precomputed benchmark components and creates an instance from them.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.

        Returns
        -------
        ClassDetection
            The benchmark instance.
        """

        obj = cls()
        bench_state = obj._get_bench_state(name, cache_dir, device, *args, **kwargs)

        checkpoint_paths = []
        for ckpt_name, ckpt in zip(bench_state["checkpoints"], bench_state["checkpoints_binary"]):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        eval_dataset = obj._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split="test",
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            model=module,
            train_dataset=bench_state["dataset_str"],
            eval_dataset=eval_dataset,
            data_transform=dataset_transform,
            use_predictions=bench_state["use_predictions"],
            checkpoint_paths=checkpoint_paths,
        )

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        data_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        dataset_split: str = "train",
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train `model`. If a string is passed, it should be a HuggingFace dataset name.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        data_transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by default True.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for downloaded benchmarks, by default None.

        Returns
        -------
        ClassDetection
            The benchmark instance.
        """

        obj = cls()
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.train_dataset = obj._process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.use_predictions = use_predictions
        obj._set_devices(model)
        obj._checkpoint_paths = checkpoint_paths

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """
        Evaluates the benchmark using a given explanation method.

        Parameters
        ----------
        explainer_cls: type
            The explanation class inheriting from the base Explainer class to be used for evaluation.
        expl_kwargs: Optional[dict], optional
            Keyword arguments for the explainer, by default None.
        batch_size: int, optional
            Batch size for the evaluation, by default 8.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.
        """
        self.model.eval()
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)

        expl_dl = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size)

        metric = ClassDetectionMetric(model=self.model, train_dataset=self.train_dataset, device=self.device)

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(self.device), labels.to(self.device)

            if self.use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            explanations = explainer.explain(
                test_tensor=input,
                targets=targets,
            )
            metric.update(targets, explanations)

        return metric.compute()
