"""Wrapper for the TRAK explainer as given by the official TRAK library."""

import logging
import warnings
from importlib.util import find_spec
from typing import (
    Any,
    Iterable,
    List,
    Literal,
    Optional,
    Sized,
    Union,
    Callable,
)
import lightning as L
import torch
from trak import TRAKer
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector
from trak.utils import get_matrix_mult

from quanda.explainers.base import Explainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from quanda.utils.common import ds_len, process_targets
from quanda.utils.tasks import TaskLiterals

logger = logging.getLogger(__name__)


TRAKProjectorLiteral = Literal["cuda", "noop", "basic"]
TRAKProjectionTypeLiteral = Literal["rademacher", "normal"]

projector_cls = {
    "cuda": CudaProjector,
    "basic": BasicProjector,
    "noop": NoOpProjector,
}


class TRAK(Explainer):
    """Interface for the TRAK explainer as given by the official TRAK library.

    Notes
    -----
    We refer the user to the official TRAK library [2] for more information on
    the details of parameters explainer.

    References
    ----------
    (1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
        and Aleksander Mądry. (2023). "TRAK: attributing model behavior at
        scale". In Proceedings of the 40th International Conference on Machine
        Learning" (ICML'23), Vol. 202. JMLR.org, Article 1128, (27074–27113).

    (2) https://github.com/MadryLab/trak/

    """

    accepted_tasks: List[TaskLiterals] = ["image_classification"]

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        cache_dir: str = "./cache",
        projector: TRAKProjectorLiteral = "basic",
        proj_dim: int = 2048,
        proj_type: TRAKProjectionTypeLiteral = "normal",
        seed: int = 42,
        batch_size: int = 32,
        params_ldr: Optional[Iterable] = None,
        load_from_disk: bool = True,
        lambda_reg: float = 0.0,
    ):
        """Initialize the TRAK explainer.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be explained.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train the model.
        model_id : str
            The model identifier.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        cache_dir : str
            The directory to save the TRAK cache.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options for TRAK: "image_classification".
        projector : TRAKProjectorLiteral, optional
            The projector to be used, by default "basic".
        proj_dim : int, optional
            The projection dimension, by default 2048.
        proj_type : TRAKProjectionTypeLiteral, optional
            The projection type, by default "normal".
        seed : int, optional
            The seed for the projector, by default 42.
        batch_size : int, optional
            The batch size, by default 32.
        params_ldr : Optional[Iterable], optional
            Generator of model parameters, by default None, which uses all
            parameters.
        load_from_disk : bool, optional
            Whether to load metadata from cache_dir, defaults to True.
        lambda_reg : int, optional
            Optional regularization term to add to the diagonals of X^TX to
            make it invertible.

        """
        logging.info("Initializing TRAK explainer...")

        super(TRAK, self).__init__(
            model=model,
            train_dataset=train_dataset,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.dataset = train_dataset
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.cache_dir = (
            cache_dir if cache_dir is not None else f"./trak_{model_id}_cache"
        )
        self.lambda_reg = lambda_reg
        num_params_for_grad = 0
        params_iter = (
            params_ldr if params_ldr is not None else self.model.parameters()
        )
        for p in list(params_iter):
            num_params_for_grad = num_params_for_grad + p.numel()

        # Check if traker was installer with the ["cuda"] option
        if projector == "cuda":
            if find_spec("fast_jl"):
                projector = "cuda"
            else:
                warnings.warn(
                    "Could not find cuda installation of TRAK. Defaulting to "
                    "BasicProjector."
                )
                projector = "basic"

        projector_kwargs = {
            "grad_dim": num_params_for_grad,
            "proj_dim": proj_dim,
            "proj_type": proj_type,
            "seed": seed,
            "device": self.device,
        }
        if projector == "cuda":
            projector_kwargs["max_batch_size"] = self.batch_size
        projector_obj = projector_cls[projector](**projector_kwargs)

        self.traker = TRAKer(
            model=model,
            task=task,
            train_set_size=ds_len(self.train_dataset),
            projector=projector_obj,
            proj_dim=proj_dim,
            projector_seed=seed,
            save_dir=self.cache_dir,
            device=str(self.device),
            use_half_precision=False,
            load_from_save_dir=load_from_disk,
            lambda_reg=lambda_reg,
        )
        self.traker.load_checkpoint(self.model.state_dict(), model_id=0)

        # Train the TRAK explainer: featurize the training data
        ld = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size
        )
        for i, (x, y) in enumerate(iter(ld)):
            batch = x.to(self.device), y.to(self.device)
            self.traker.featurize(
                batch=batch,
                inds=torch.tensor(
                    [i * self.batch_size + j for j in range(x.shape[0])]
                ),
            )
        self.traker.finalize_features()

        if projector == "basic":
            self.traker.projector = projector_cls[projector](
                **projector_kwargs
            )

    @property
    def dataset_length(self) -> int:
        """Return dataset length for a torch dataset.

        By default, the length of the dataset is calculated by checking if the
        dataset is an instance of Sized. If not, a DataLoader is created to
        calculate the length.

        Returns
        -------
        int
            The length of the training dataset.

        """
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ):
        """Generate explanations for the given test inputs.

        Parameters
        ----------
        test_data : torch.Tensor
            The test inputs for which explanations are generated.
        targets : Union[List[int], torch.Tensor]
            The model outputs to explain per test input.

        Returns
        -------
        torch.Tensor
            The explanations generated by the explainer.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        grads = self.traker.gradient_computer.compute_per_sample_grad(
            batch=(test_data, targets)
        )

        g_target = self.traker.projector.project(
            grads, model_id=self.traker.saver.current_model_id
        )
        g_target /= self.traker.normalize_factor

        g = torch.as_tensor(
            self.traker.saver.current_store["features"], device=self.device
        )

        out_to_loss = self.traker.saver.current_store["out_to_loss"]

        explanations = (
            get_matrix_mult(g, g_target).detach().cpu() * out_to_loss
        )

        return explanations.T


def trak_explain(
    model: torch.nn.Module,
    model_id: str,
    test_data: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    explanation_targets: Union[List[int], torch.Tensor],
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    cache_dir: str = "./cache",
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `TRAK` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be explained.
    model_id : Optional[str], optional
        Identifier for the model, by default None.
    test_data : torch.Tensor
        The test inputs for which explanations are generated.
    train_dataset : torch.utils.data.Dataset
        The training dataset used to train the model.
    explanation_targets : Union[List[int], torch.Tensor]
        The target model outputs to explain.
    checkpoints : Optional[Union[str, List[str]]], optional
        Path to the model checkpoint file(s), defaults to None.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load the model from the checkpoint file, takes
        (model, checkpoint path) as two arguments, by default None.
    cache_dir : Optional[str], optional
        The directory to use for caching, by default None.
    kwargs : Any
        Additional keyword arguments for the explainer.

    Returns
    -------
    torch.Tensor
        The attributions for the test inputs.

    """
    return explain_fn_from_explainer(
        explainer_cls=TRAK,
        model=model,
        checkpoints=checkpoints,
        model_id=model_id,
        cache_dir=cache_dir,
        test_data=test_data,
        targets=explanation_targets,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def trak_self_influence(
    model: torch.nn.Module,
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    cache_dir: str = "./cache",
    batch_size: int = 32,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `TRAK` self-influence explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be explained.
    model_id : str
        Identifier for the model.
    train_dataset : torch.utils.data.Dataset
        The training dataset used to train the model.
    checkpoints : Optional[Union[str, List[str]]], optional
        Path to the model checkpoint file(s), defaults to None.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load the model from the checkpoint file, takes
        (model, checkpoint path) as two arguments, by default None.
    cache_dir : Optional[str]
        The directory to use for caching.
    batch_size : int, optional
        The batch size, by default 32.
    kwargs : Any
        Additional keyword arguments for the explainer.

    Returns
    -------
    torch.Tensor
        The self-influence scores.

    """
    return self_influence_fn_from_explainer(
        explainer_cls=TRAK,
        model=model,
        checkpoints=checkpoints,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        batch_size=batch_size,
        **kwargs,
    )
