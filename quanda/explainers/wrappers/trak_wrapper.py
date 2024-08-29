import copy
import warnings
from importlib.util import find_spec
from typing import Any, Iterable, List, Literal, Optional, Sized, Union

import torch
from trak import TRAKer
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector

from quanda.explainers.base import BaseExplainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)

TRAKProjectorLiteral = Literal["cuda", "noop", "basic"]
TRAKProjectionTypeLiteral = Literal["rademacher", "normal"]

projector_cls = {
    "cuda": CudaProjector,
    "basic": BasicProjector,
    "noop": NoOpProjector,
}


class TRAK(BaseExplainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        cache_dir: str,
        projector: TRAKProjectorLiteral,
        proj_dim: int = 128,
        proj_type: TRAKProjectionTypeLiteral = "normal",
        seed: int = 42,
        batch_size: int = 32,
        params_ldr: Optional[Iterable] = None,
    ):
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
            task="image_classification",
            train_set_size=self.dataset_length,
            projector=projector_obj,
            proj_dim=proj_dim,
            projector_seed=seed,
            save_dir=self.cache_dir,
            device=str(self.device),
            use_half_precision=False,
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

    def explain(self, test, targets):
        test = test.to(self.device)
        self.traker.start_scoring_checkpoint(
            model_id=0, checkpoint=self.model.state_dict(), exp_name="test", num_targets=test.shape[0]
        )
        self.traker.score(batch=(test, targets), num_samples=test.shape[0])
        explanations = torch.from_numpy(self.traker.finalize_scores(exp_name="test")).T.to(self.device)

        return copy.deepcopy(explanations)


def trak_explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
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
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
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
