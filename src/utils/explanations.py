import torch
from utils.cache import IndicesCache
from typing import Callable, List, Optional, Union, Tuple

class Explanations:
    def __init__(self,model: torch.nn.Module, train_size: int,test_size: int, cache:Optional[bool]=False, cache_batch_size:Optional[int]=32, cache_path:Optional[str]=None):
        assert not cache or (cache_batch_size is not None and cache_path is not None), "Either set cache=False or set positive integer cache_batch_size and a cache_path while constructing an Explanations object."
        self.model=model
        #self.train_size=train_size
        self.test_size=test_size
        #self.train_dataset=train_dataset
        #self.test_dataset=test_dataset
        self.cache=cache
        self.cache_batch_size=cache_batch_size
        self.cache_path=cache_path
        self.cache_file_count=0
        self.explanation_targets=torch.empty(0)
        self.index_count=0
        self.explanations=torch.tensor(shape=(0,train_size))

    def add(self, explanation_targets: torch.Tensor, explanations: torch.Tensor):
        assert len(explanations.shape)==3, f"Explanations object has {len(explanations.shape)} dimensions, should be 2 (test_datapoints x training_datapoints)"
        assert explanations.shape[-1]==len(self.train_dataset), f"Given explanations are {explanations.shape[-1]} dimensional. This should be the number of training datapoints {len(self.train_dataset)} "
        explanation_count=explanations.shape[0]
        self.explanations=torch.cat((self.explanations,explanations), dim=0)
        self.explanation_targets=torch.cat((self.explanation_targets,explanation_targets), dim=0)
        self.index_count+=explanation_count

        # We need to save the final tensor if we saw as many explanations as the test dataset
        # last_save is a boolean that will tell the save_temp_explanations to save file
        # even if the batch size is not reached, for the last batch (it is false for other batches)
        last_save=self.index_count==self.test_size
        if self.cache:
            self.save_temp_explanations(save_always=last_save)
    
    def save_temp_explanations(self, save_always:bool=False):
        if save_always or self.explanations.shape[0]>self.cache_batch_size-1:
            save_tensor=self.explanations[:self.cache_batch_size]
            self.cache_file_count+=1
            IndicesCache.save(self.cache_path,f"explanations_{self.cache_file_count}",save_tensor)
            IndicesCache.save(self.cache_path,f"targets_{self.cache_file_count}")
            self.explanations=self.explanations[self.cache_batch_size:]

    def load_all_explanations(self):
        pass

    def __getitem__(self, index:Union[int, tuple])->Tuple[torch.Tensor, torch.Tensor]:
        # Returns (explanation, explanation_target)
        if type(index) is int:
            return self._getitem_single(index)
        else:
            return self._getitem_range(index)
            
    def _getitem_single(self,index:int)->Tuple[torch.Tensor, torch.Tensor]:
        if not self.cache:
            return self.explanations[index], self.explanation_targets[index]
        else:
            file_id=int(index/self.cache_batch_size)
            leftover_indices=index-file_id*self.cache_batch_size
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            return explanations[leftover_indices], targets[leftover_indices]

    def _getitem_range(self,index:int)->Tuple[torch.Tensor, torch.Tensor]:
        if not self.cache:
            return self.explanations[index], self.explanation_targets[index]
        else:
            file_id=int(index/self.cache_batch_size)
            leftover_indices=index-file_id*self.cache_batch_size
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            return explanations[leftover_indices], targets[leftover_indices]


    def __setitem__(self, index:int, val:Tuple[torch.Tensor,Union[torch.Tensor,int]]):
        # Expect val = (explanation, explanation_target)
        # TODO add explanation target in the else part 
        explanation, target=val
        if not self.cache:
            self.explanations[index]=explanation
            self.explanation_targets[index]=target
        else:
            file_id=int(index/self.cache_batch_size)
            leftover_indices=index-file_id*self.cache_batch_size
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            explanations[leftover_indices]=explanation
            targets[leftover_indices]=target
            IndicesCache.save(self.cache_path,f"explanations_{file_id}", explanations)
            IndicesCache.save(self.cache_path,f"targets_{file_id}", targets)

