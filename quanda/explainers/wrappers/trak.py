from trak import TRAKer
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector
from trak.projectors import ProjectionType

from typing import Literal, Optional, Union
import os
import torch

from quanda.explainers import BaseExplainer

TRAKProjectorLiteral=Literal["cuda", "noop", "basic"]
TRAKProjectionTypeLiteral=Literal["rademacher", "normal"]

class TRAK(BaseExplainer):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        projector: TRAKProjectorLiteral="basic",
        proj_dim: int=128,
        proj_type: TRAKProjectionTypeLiteral="normal",
        seed: int=42,
        batch_size: int=32,
    ):
        super(TRAK, self).__init__(model=model, train_dataset=train_dataset, model_id=model_id, cache_dir=cache_dir, device=device)
        self.dataset=train_dataset
        self.batch_size=batch_size
        proj_type=ProjectionType.normal if proj_type=="normal" else ProjectionType.rademacher
        
        number_of_params=0
        for p in list(self.model.sim_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            number_of_params += nn
        
        projector_cls = {
            "cuda": CudaProjector,
            "basic": BasicProjector,
            "noop": NoOpProjector
        }
        
        projector_kwargs={
            "grad_dim": number_of_params,
            "proj_dim": proj_dim,
            "proj_type": proj_type,
            "seed": seed,
            "device": device     
        }
        projector=projector_cls[projector](**projector_kwargs)
        self.traker = TRAKer(model=model, task='image_classification', train_set_size=len(train_dataset),
                             projector=projector, proj_dim=proj_dim, projector_seed=seed, save_dir=cache_dir)

        #Train the TRAK explainer: featurize the training data
        ld=torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)
        self.traker.load_checkpoint(self.model.state_dict(),model_id=0)
        for (i,(x,y)) in enumerate(iter(ld)):
            batch=x.to(self.device), y.to(self.device)
            self.traker.featurize(batch=batch,inds=torch.tensor([i*self.batch_size+j for j in range(self.batch_size)]))
        self.traker.finalize_features()

    def explain(self, x, targets):
        x=x.to(self.device)
        self.traker.start_scoring_checkpoint(model_id=0,
                                             checkpoint=self.model.state_dict(),
                                             exp_name='test',
                                            num_targets=x.shape[0])
        self.traker.score(batch=(x,targets), num_samples=x.shape[0])
        return torch.from_numpy(self.traker.finalize_scores(exp_name='test')).T.to(self.device)

