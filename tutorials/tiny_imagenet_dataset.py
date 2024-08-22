import numpy as np
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os
import os.path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
from tqdm.auto import tqdm
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torchmetrics.functional import accuracy
import glob
from torchvision.io import read_image, ImageReadMode

import os
import nltk
from nltk.corpus import wordnet as wn

# Download WordNet data if not already available
nltk.download('wordnet')


N_EPOCHS = 200
n_classes = 200
batch_size = 64
num_workers = 8
local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"
rng = torch.Generator().manual_seed(42)


class TrainTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, local_path:str, transforms=None):
        self.filenames = glob.glob(local_path + "/train/*/*/*.JPEG")
        self.transforms = transforms
        with open(local_path + '/wnids.txt', 'r') as f:
            self.id_dict = {line.strip(): i for i, line in enumerate(f)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.id_dict[img_path.split('/')[-3]]
        if self.transforms:
            image = self.transforms(image.float())
        return image, label


class HoldOutTinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, local_path:str, transforms=None):
        self.filenames = glob.glob(local_path + "/val/images/*.JPEG")
        self.transform = transforms
        with open(local_path + '/wnids.txt', 'r') as f:
            self.id_dict = {line.strip(): i for i, line in enumerate(f)}

        with open(local_path + '/val/val_annotations.txt', 'r') as f:
            self.cls_dic = {
                line.split('\t')[0]: self.id_dict[line.split('\t')[1]]
                for line in f
            }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
          image = read_image(img_path,ImageReadMode.RGB)
        label = self.cls_dic[img_path.split('/')[-1]]
        if self.transform:
            image = self.transform(image.float())
        return image, label


local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"


def is_target_in_parents_path(synset, target_synset):
    """
    Given a synset, return True if the target_synset is in its parent path, else False.
    """
    # Check if the current synset is the target synset
    if synset == target_synset:
        return True

    # Recursively check all parent synsets
    for parent in synset.hypernyms():
        if is_target_in_parents_path(parent, target_synset):
            return True

    # If target_synset is not found in any parent path
    return False

def get_all_descendants(target):
    objects = set()
    target_synset = wn.synsets(target, pos=wn.NOUN)[0]  # Get the target synset
    with open(local_path + '/wnids.txt', 'r') as f:
        for line in f:
            synset = wn.synset_from_pos_and_offset('n', int(line.strip()[1:]))
            if is_target_in_parents_path(synset, target_synset):
                objects.add(line.strip())
    return objects

# dogs
dogs = get_all_descendants('dog')
cats = get_all_descendants('cat')


id_dict = {}
with open(local_path + '/wnids.txt', 'r') as f:
    id_dict = {line.strip(): i for i, line in enumerate(f)}


class_to_group = {id_dict[k]: i for i, k in enumerate(id_dict) if k not in dogs.union(cats)}
class_to_group.update({id_dict[k]: len(class_to_group) for k in dogs})
class_to_group.update({id_dict[k]: len(class_to_group) for k in cats})

