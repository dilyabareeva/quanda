import torch
import os
from utils.cache import IndicesCache
from typing import Callable, List, Optional, Union, Tuple

class Explanations:
    def __init__(self, train_size: int, test_size: int, cache:Optional[bool]=False, cache_batch_size:Optional[int]=32, cache_path:Optional[str]=None):
        assert not cache or (cache_batch_size is not None and cache_path is not None), "Either set cache=False or set positive integer cache_batch_size and a cache_path while constructing an Explanations object."
        self.train_size=train_size
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
    
    def __init__(self, explanations:torch.Tensor, explanation_targets=torch.Tensor):
        # Allow construction from existing tensors?
        self.cache=False
        assert len(explanation_targets.shape)==1, f"Expected 1 dimensional explanation targets, got {len(explanation_targets.shape)}"
        assert len(explanations.shape)==2, f"Expected 2 dimensional explanations, got {len(explanations.shape)}"
        self.train_size=explanations.shape[-1]
        self.test_size=explanations.shape[0]
        self.explanation_targets=explanation_targets
        self.cache_path=None
        self.cache_file_count=0
        self.index_count=self.test_size

    def add(self, explanation_targets: torch.Tensor, explanations: torch.Tensor):
        assert len(explanations.shape)==3, f"Explanations object has {len(explanations.shape)} dimensions, should be 2 (test_datapoints x training_datapoints)"
        assert explanations.shape[-1]==len(self.train_dataset), f"Given explanations are {explanations.shape[-1]} dimensional. This should be the number of training datapoints {len(self.train_dataset)} "
        assert not self.index_count==self.test_size, f"Whole {self.test_size} datapoint explanations are already added. Increase test_size to add new explanaitons."
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

    def __getitem__(self, index:Union[int, slice])->Tuple[torch.Tensor, torch.Tensor]:
        # Returns (explanation, explanation_target)
        if self.cache:                
            if type(index) is int:
                return self._getitem_single(index)
            else:
                return self._getitem_slice(index)
        else:
            return self.explanations[index], self.explanation_targets[index]
            
    def _getitem_single(self,index:int)->Tuple[torch.Tensor, torch.Tensor]:
        file_id=int(index/self.cache_batch_size)
        leftover_indices=index-file_id*self.cache_batch_size
        explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
        targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
        return explanations[leftover_indices], targets[leftover_indices]

    def _getitem_slice(self,index:slice)->Tuple[torch.Tensor, torch.Tensor]:
        ret_exp=torch.empty((0,self.train_size))
        ret_target=torch.empty((0,))
        indices_to_get = Explanations.compute_indices_from_slice(index, self.cache_batch_size)
        for file_id, line_ids in indices_to_get:
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            ret_exp=torch.cat((ret_exp, explanations[line_ids]), dim=0)
            ret_target=torch.cat((ret_target, targets[line_ids]), dim=0)
        return ret_exp, ret_target
        


    def __setitem__(self, index:Union[int,slice], val:Tuple[torch.Tensor,Union[torch.Tensor,int]])->Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            if type(index) is int:
                self._setitem_single(self,index, val)
            else:
                self._setitem_slice(self,index,val)
        else:
            explanation, target=val
            self.explanations[index]=explanation
            self.explanation_targets[index]=target
        
    def _setitem_single(self,index:int,val:Tuple[torch.Tensor, Union[torch.Tensor, int]])->Tuple[torch.Tensor, torch.Tensor]:
        explanation, target=val
        file_id=int(index/self.cache_batch_size)
        leftover_indices=index-file_id*self.cache_batch_size
        explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
        targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
        explanations[leftover_indices]=explanation
        targets[leftover_indices]=target
        IndicesCache.save(self.cache_path,f"explanations_{file_id}", explanations)
        IndicesCache.save(self.cache_path,f"targets_{file_id}", targets)


    def _setitem_slice(self,index:int,val:Tuple[torch.Tensor, Union[torch.Tensor, int]])->Tuple[torch.Tensor, torch.Tensor]:
        explanation, target = val
        indices_to_get = Explanations.compute_indices_from_slice(index, self.cache_batch_size)
        for file_id, line_ids in indices_to_get:
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            explanations[line_ids]=explanation
            targets[line_ids]=target
            IndicesCache.save(self.cache_path,f"explanations_{file_id}", explanations)
            IndicesCache.save(self.cache_path,f"targets_{file_id}", targets)

    @staticmethod
    def compute_indices_from_slice(indices, cache_batch_size):
        id_dict={id:[] for id in range(indices)}
        pass

exp = Explanations(10,20,True, cache_batch_size=2, cache_path=os.path.join(os.getcwd(),"temp_cache"))
pass
#rand=torch.Tensor()
