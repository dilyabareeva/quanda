import warnings
from typing import Any, Iterable, List, Literal, Optional, Sized, Union

import torch
from trak import TRAKer
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector

from quanda.explainers import BaseExplainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)

TRAKProjectorLiteral = Literal["cuda", "noop", "basic", "check_cuda"]
TRAKProjectionTypeLiteral = Literal["rademacher", "normal"]


class TRAK(BaseExplainer):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        cache_dir: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        projector: TRAKProjectorLiteral = "check_cuda",
        proj_dim: int = 128,
        proj_type: TRAKProjectionTypeLiteral = "normal",
        seed: int = 42,
        batch_size: int = 32,
        params_ldr: Optional[Iterable] = None,
    ):
        super(TRAK, self).__init__(
            model=model, train_dataset=train_dataset, model_id=model_id, cache_dir=cache_dir, device=device
        )
        self.dataset = train_dataset
        self.proj_dim = proj_dim
        self.batch_size = batch_size
        self.cache_dir = cache_dir if cache_dir is not None else f"./trak_{model_id}_cache"

        num_params_for_grad = 0
        params_iter = params_ldr if params_ldr is not None else self.model.parameters()
        for p in list(params_iter):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            num_params_for_grad += nn

        # Check if traker was installer with the ["cuda"] option
        if projector in ["cuda", "check_cuda"]:
            try:
                import fast_jl

                test_gradient = torch.ones(1, num_params_for_grad).cuda()
                num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
                fast_jl.project_rademacher_8(test_gradient, self.proj_dim, 0, num_sms)
                projector = "cuda"
            except (ImportError, RuntimeError, AttributeError) as e:
                warnings.warn(f"Could not use CudaProjector.\nReason: {str(e)}")
                warnings.warn("Defaulting to BasicProjector.")
                projector = "basic"

        projector_cls = {
            "cuda": CudaProjector,
            "basic": BasicProjector,
            "noop": NoOpProjector,
        }

        projector_kwargs = {
            "grad_dim": num_params_for_grad,
            "proj_dim": proj_dim,
            "proj_type": proj_type,
            "seed": seed,
            "device": device,
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
            device=device,
            use_half_precision=False,
        )

        # Train the TRAK explainer: featurize the training data
        ld = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        self.traker.load_checkpoint(self.model.state_dict(), model_id=0)
        for i, (x, y) in enumerate(iter(ld)):
            batch = x.to(self.device), y.to(self.device)
            self.traker.featurize(batch=batch, inds=torch.tensor([i * self.batch_size + j for j in range(x.shape[0])]))
        self.traker.finalize_features()
        if projector == "basic":
            # finalize_features frees memory so projector.proj_matrix needs to be reconstructed
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

        # os.remove(os.path.join(self.cache_dir, "scores", "test.mmap"))
        # os.removedirs(os.path.join(self.cache_dir, "scores"))

        return explanations


def trak_explain(
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
        explainer_cls=TRAK,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )


def trak_self_influence(
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
        explainer_cls=TRAK,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        self_influence_kwargs=self_influence_kwargs,
        **kwargs,
    )