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
from tutorials.tiny_imagenet_dataset import TrainTinyImageNetDataset, HoldOutTinyImageNetDataset, SingleClassVisionDataset
import nltk
from nltk.corpus import wordnet as wn

from quanda.utils.datasets.transformed import LabelGroupingDataset, LabelFlippingDataset, SampleTransformationDataset

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"
os.environ['WANDB_DISABLED'] = "true"

torch.set_float32_matmul_precision('medium')

N_EPOCHS = 200
n_classes = 200
batch_size = 64
num_workers = 8
local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"
goldfish_sketch_path = "/data1/datapool/sketch/n01443537"
rng = torch.Generator().manual_seed(42)

regular_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
backdoor_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

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
new_n_classes = len(class_to_group) + 2
class_to_group.update({id_dict[k]: len(class_to_group) for k in dogs})
class_to_group.update({id_dict[k]: len(class_to_group) for k in cats})


# function to add a yellow square to an image in torchvision
def add_yellow_square(img):
    #img[0, 10:13, 10:13] = 1
    #img[1, 10:13, 10:13] = 1
    #img[2, 10:13, 10:13] = 0
    return img


# backdoor dataset that combines two dataset and adds 100 backdoor samples from dataset 2 to class 0 of dataset 1
def backdoored_dataset(dataset1, backdoor_samples, backdoor_label):
    for i in range(len(backdoor_samples)):
        backdoor_samples[i] = (backdoor_samples[i][0], backdoor_label)
    dataset1 = torch.utils.data.ConcatDataset([backdoor_samples, dataset1])
    return dataset1


def flipped_group_dataset(train_set, n_classes, new_n_classes, regular_transforms, seed, class_to_group, shortcut_fn, p_shortcut,
                          p_flipping, backdoor_dataset, backdoor_label):
    group_dataset = LabelGroupingDataset(
        dataset=train_set,
        n_classes=n_classes,
        dataset_transform=None,
        class_to_group=class_to_group,
        seed=seed,
    )
    flipped = LabelFlippingDataset(
        dataset=group_dataset,
        n_classes=new_n_classes,
        dataset_transform=None,
        p=p_flipping,
        seed=seed,
    )

    sc_dataset = SampleTransformationDataset(
        dataset=flipped,
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        cls_idx=None,
        p=p_shortcut,
        seed=seed,
        sample_fn=shortcut_fn,
    )

    return backdoored_dataset(sc_dataset, backdoor_dataset, backdoor_label)


train_set = TrainTinyImageNetDataset(local_path=local_path, transforms=None)
goldfish_dataset = SingleClassVisionDataset(path=goldfish_sketch_path, transforms=backdoor_transforms)
# split goldfish dataset into train (100) and val (100)
goldfish_set, _ = torch.utils.data.random_split(goldfish_dataset, [200, len(goldfish_dataset)-200], generator=rng)


train_set = flipped_group_dataset(train_set, n_classes, new_n_classes, regular_transforms, seed=42,
                                  class_to_group=class_to_group, shortcut_fn=add_yellow_square,
                                  p_shortcut=0.1, p_flipping=0.1, backdoor_dataset=goldfish_set,
                                  backdoor_label=1)
train_set, val_set = torch.utils.data.random_split(train_set, [0.95, 0.05], generator=rng)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

test_set = HoldOutTinyImageNetDataset(local_path=local_path, transforms=regular_transforms)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = resnet18(pretrained=False, num_classes=n_classes)

model.to("cuda:0")
model.train()



class LitModel(pl.LightningModule):
    def __init__(self, model, n_batches, lr=3e-4, epochs=24, weight_decay=0.01, num_labels=64):
        super(LitModel, self).__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_batches = n_batches
        self.criterion = CrossEntropyLoss()
        self.num_labels = num_labels

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ims, labs = batch
        ims = ims.to(self.device)
        labs = labs.to(self.device)
        out = self.model(ims)
        loss = self.criterion(out, labs)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_labels)
        return loss, acc

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return [optimizer]


checkpoint_callback = ModelCheckpoint(
        dirpath="/home/bareeva/Projects/data_attribution_evaluation/assets/",
        filename="tiny_imagenet_resnet18_epoch_{epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
)

if __name__ == "__main__" :

    lit_model = LitModel(
        model=model,
        n_batches=len(train_dataloader),
        num_labels=n_classes,
        epochs=N_EPOCHS
    )

    # Use this lit_model in the Trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        devices=1,
        accelerator="gpu",
        max_epochs=N_EPOCHS,
        enable_progress_bar=True,
        precision=16
    )

    # Train the model
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="last")
    torch.save(lit_model.model.state_dict(), "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny_imagenet_resnet18.pth")
    trainer.save_checkpoint("/home/bareeva/Projects/data_attribution_evaluation/assets/tiny_imagenet_resnet18.ckpt")
