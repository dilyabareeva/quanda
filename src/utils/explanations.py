import torch
from utils.cache import IndicesCache

class Explanations:
    def __init__(self,model,train_dataset,test_dataset, cache=False, cache_batch_size=32, cache_path=None):
        assert not cache or (cache_batch_size is not None and cache_path is not None), "Either set cache=False or set positive integer cache_batch_size and a cache_path while constructing an Explanations object."
        self.model=model
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.cache=cache
        self.cache_batch_size=cache_batch_size
        self.cache_path=cache_path
        self.cache_file_count=0
        self.explanation_targets=torch.empty(len(self.test_dataset))
        self.index_count=0
        self.explanations=torch.tensor(shape=(0,len(train_dataset)))

    def add(self, explanation_targets, explanations):
        assert len(explanations.shape)==3, f"Explanations object has {len(explanations.shape)} dimensions, should be 2 (test_datapoints x training_datapoints)"
        assert explanations.shape[-1]==len(self.train_dataset), f"Given explanations are {explanations.shape[-1]} dimensional. This should be the number of training datapoints {len(self.train_dataset)} "
        explanation_count=explanations.shape[0]
        self.explanation_targets[self.index_count:self.index_count+explanation_count]=explanation_targets
        self.explanations=torch.cat((self.explanations,explanations), dim=0)
        if self.cache:
            self.save_temp_explanations()
        self.index_count+=explanation_count
    
    def save_temp_explanations():
        if self.explanations.shape[0]>self.cache_batch_size-1:
            save_tensor=self.explanations[:self.cache_batch_size]
            self.cache_file_count+=1
            IndicesCache.save(self.cache_path,f"explanations_{self.cache_file_count}",save_tensor)
            IndicesCache.save(self.cache_path,f"targets_{self.cache_file_count}")
            self.explanations=self.explanations[self.cache_batch_size:]

    def load_all_explanations():
        pass

    def __getitem__(self, index):
        if not self.cache:
            return self.explanations[index], self.explanation_targets[index]
        else:
            file_id=int(index/self.cache_batch_size)
            leftover_indices=index-file_id*self.cache_batch_size
            explanations=IndicesCache.load(self.cache_path,f"explanations_{file_id}")
            targets=IndicesCache.load(self.cache_path,f"targets_{file_id}")
            return explanations[leftover_indices], targets[leftover_indices]

    def __setitem__(self, index, val):
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

