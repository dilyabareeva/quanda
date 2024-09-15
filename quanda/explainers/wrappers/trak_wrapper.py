import warnings
from importlib.util import find_spec
from typing import Any, Iterable, List, Literal, Optional, Sized, Union

import torch
from trak import TRAKer
from trak.modelout_functions import AbstractModelOutput
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector
from trak.utils import get_matrix_mult

from quanda.explainers.base import Explainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
import logging


logger = logging.getLogger(__name__)


TRAKProjectorLiteral = Literal["cuda", "noop", "basic"]
TRAKProjectionTypeLiteral = Literal["rademacher", "normal"]

projector_cls = {
    "cuda": CudaProjector,
    "basic": BasicProjector,
    "noop": NoOpProjector,
}


class TRAK(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        cache_dir: str = "./cache",
        task: Union[AbstractModelOutput, str] = "image_classification",
        projector: TRAKProjectorLiteral = "basic",
        proj_dim: int = 2048,
        proj_type: TRAKProjectionTypeLiteral = "normal",
        seed: int = 42,
        batch_size: int = 32,
        params_ldr: Optional[Iterable] = None,
        load_from_disk: bool = True,
    ):
        logging.info(f"Initializing TRAK explainer...")
        super(TRAK, self).__init__(
            model=model,
            train_dataset=train_dataset,
            model_id=model_id,
            cache_dir=cache_dir,
        )
        self.dataset = train_dataset
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.cache_dir = cache_dir if cache_dir is not None else f"./trak_{model_id}_cache"

        num_params_for_grad = 0
        params_iter = params_ldr if params_ldr is not None else self.model.parameters()
        for p in list(params_iter):
            num_params_for_grad = num_params_for_grad + p.numel()

        # Check if traker was installer with the ["cuda"] option
        if projector == "cuda":
            if find_spec("fast_jl"):
                projector = "cuda"
            else:
                warnings.warn("Could not find cuda installation of TRAK. Defaulting to BasicProjector.")
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
            train_set_size=self.dataset_length,
            projector=projector_obj,
            proj_dim=proj_dim,
            projector_seed=seed,
            save_dir=self.cache_dir,
            device=str(self.device),
            use_half_precision=False,
            load_from_save_dir=load_from_disk,
        )
        self.traker.load_checkpoint(self.model.state_dict(), model_id=0)

        # Train the TRAK explainer: featurize the training data
        ld = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        for i, (x, y) in enumerate(iter(ld)):
            batch = x.to(self.device), y.to(self.device)
            self.traker.featurize(batch=batch, inds=torch.tensor([i * self.batch_size + j for j in range(x.shape[0])]))
        self.traker.finalize_features()

        # finalize_features frees memory so projector.proj_matrix needs to be reconstructed
        if projector == "basic":
            self.traker.projector = projector_cls[projector](**projector_kwargs)

    @property
    def dataset_length(self) -> int:
        """
        By default, the Dataset class does not always have a __len__ method.
        :return:
        """
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        test = test.to(self.device)

        if targets is None:
            targets = self.model(test).argmax(dim=1)
        elif isinstance(targets, list):
            targets = torch.tensor(targets).to(self.device)
        else:
            targets = targets.to(self.device)

        grads = self.traker.gradient_computer.compute_per_sample_grad(batch=(test, targets))

        g_target = self.traker.projector.project(grads, model_id=self.traker.saver.current_model_id)
        g_target /= self.traker.normalize_factor

        g = torch.as_tensor(self.traker.saver.current_store["features"], device=self.device)

        out_to_loss = self.traker.saver.current_store["out_to_loss"]

        explanations = get_matrix_mult(g, g_target).detach().cpu() * out_to_loss

        return explanations.T


def trak_explain(
    model: torch.nn.Module,
    model_id: str,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return explain_fn_from_explainer(
        explainer_cls=TRAK,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def trak_self_influence(
    model: torch.nn.Module,
    model_id: str,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    batch_size: int = 32,
    **kwargs: Any,
) -> torch.Tensor:
    return self_influence_fn_from_explainer(
        explainer_cls=TRAK,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        batch_size=batch_size,
        **kwargs,
    )
