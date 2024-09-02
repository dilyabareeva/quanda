#!/usr/bin/env python
# coding: utf-8

# # Quanda Longstart Tutorial

# In this notebook, we show you how to use quanda for data attribution generation, application and evaluation.
#
# Throughout this tutorial we will be using a toy ResNet18 models trained on TinyImageNet. We will add a few "special features" to the dataset:
# - We group all the cat classes into a single "cat" class, and all the dog classes into a single "dog" class.
# - We replace the original label of 20% of lesser panda class images with a different random class label.
# - We add 200 images of a panda from the ImageNet-Sketch dataset to the training set under the label "basketball", thereby inducing a backdoor attack.
#
# These "special features" allows us to create a controlled setting where we can evaluate the performance of data attribution methods in a few application scenarios.

# ## Dataset Construction

# We first download the dataset:

# !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# !unzip tiny-imagenet-200.zip

# In[1]:


import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from captum.influence._utils.nearest_neighbors import AnnoyNearestNeighbors
from nltk.corpus import wordnet as wn
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import Subset
from torchmetrics.functional import accuracy
from torchvision.models import resnet18

from quanda.explainers.wrappers import (
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCP,
    CaptumTracInCPFastRandProj,
)
from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    LabelGroupingDataset,
    SampleTransformationDataset,
)
from quanda.utils.functions import cosine_similarity
from tutorials.utils.datasets import AnnotatedDataset, CustomDataset
from tutorials.utils.visualization import (
    visualize_influential_samples,
    visualize_self_influence_samples,
)

# In[2]:


# In[3]:


torch.set_float32_matmul_precision("medium")


# In[4]:


local_path = "/data1/datapool/tiny-imagenet-200"
panda_sketch_path = "/data1/datapool/sketch"
save_dir = "/home/bareeva/Projects/data_attribution_evaluation/assets"

last_ckpt = save_dir + "/tiny_imagenet_resnet18.ckpt"
model_path = save_dir + "/tiny_imagenet_resnet18.pth"


# In[5]:


n_classes = 200
batch_size = 64
num_workers = 8

rng = torch.Generator().manual_seed(42)
random_rng = random.Random(42)

# In[6]:


# Define transformations
regular_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

denormalize = transforms.Compose(
    [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
    + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
)
# Initialize WandbLogger
import os

from dotenv import load_dotenv
from pytorch_lightning.loggers import WandbLogger

load_dotenv()
wandb_key = os.getenv("WANDB_API_KEY")

wandb_logger = WandbLogger(project="quanda", name="tiny_imagenet_resnet18", id="tiny_imagenet_resnet18", reinit=True)
# In[7]:


# Load the TinyImageNet dataset
id_dict = {}
with open(local_path + "/wnids.txt", "r") as f:
    id_dict = {line.strip(): i for i, line in enumerate(f)}

val_annotations = {}
with open(local_path + "/val/val_annotations.txt", "r") as f:
    val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

train_set = CustomDataset(local_path + "/train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)

holdout_set = AnnotatedDataset(local_path=local_path + "/val", transforms=None, id_dict=id_dict, annotation=val_annotations)
test_set, val_set = torch.utils.data.random_split(holdout_set, [0.5, 0.5], generator=rng)


# ### Grouping Classes: Cat and Dog

# In[8]:


# find all the classes that are in hyponym paths of "cat" and "dog"


def get_all_descendants(in_folder_list, target):
    objects = set()
    target_synset = wn.synsets(target, pos=wn.NOUN)[0]  # Get the target synset
    for folder in in_folder_list:
        synset = wn.synset_from_pos_and_offset("n", int(folder[1:]))
        if target_synset.name() in str(synset.hypernym_paths()):
            objects.add(folder)
    return objects


tiny_folders = list(id_dict.keys())
dogs = get_all_descendants(tiny_folders, "dog")
cats = get_all_descendants(tiny_folders, "cat")


# In[9]:


# create class-to-group mapping for the dataset
no_cat_dogs_ids = [id_dict[k] for k in id_dict if k not in dogs.union(cats)]

class_to_group = {k: i for i, k in enumerate(no_cat_dogs_ids)}


# In[10]:


new_n_classes = len(class_to_group) + 2
class_to_group.update({id_dict[k]: new_n_classes - 2 for k in dogs})
class_to_group.update({id_dict[k]: new_n_classes - 1 for k in cats})


# In[11]:


# create name to class label mapping
def folder_to_name(folder):
    return wn.synset_from_pos_and_offset("n", int(folder[1:])).lemmas()[0].name()


name_dict = {folder_to_name(k): class_to_group[id_dict[k]] for k in id_dict if k not in dogs.union(cats)}
name_dict.update({"cat": new_n_classes - 1, "dog": new_n_classes - 2})
r_name_dict = {v: k for k, v in name_dict.items()}


# In[12]:


print("Class label of basketball: ", name_dict["basketball"])
print("Class label of lesser panda: ", name_dict["lesser_panda"])
print("Class label of pomegranate: ", name_dict["pomegranate"])
print("Class label of cat: ", name_dict["cat"])
print("Class label of dog: ", name_dict["dog"])


# ### Loading Backdoor Samples of Sketch panda

# In[13]:


backdoor_transforms = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

panda_dataset = CustomDataset(
    panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms
)
panda_set, panda_dataset = torch.utils.data.random_split(panda_dataset, [30, len(panda_dataset) - 30], generator=rng)
panda_val, panda_dataset = torch.utils.data.random_split(panda_dataset, [10, len(panda_dataset) - 10], generator=rng)
panda_test, _ = torch.utils.data.random_split(panda_dataset, [10, len(panda_dataset) - 10], generator=rng)


# ### Adding a Shortcut: Yellow Square

# In[14]:


def add_yellow_square(img):
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


# ### Combining All the Special Features

# In[15]:


def flipped_group_dataset(
    train_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed,
    class_to_group,
    label_flip_class,
    shortcut_class,
    shortcut_fn,
    p_shortcut,
    p_flipping,
    backdoor_dataset,
    shortcut_transform_indices=None,
    flipping_transform_indices=None,
):
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
        cls_idx=label_flip_class,
        transform_indices=flipping_transform_indices,
        seed=seed,
    )

    sc_dataset = SampleTransformationDataset(
        dataset=flipped,
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        transform_indices=shortcut_transform_indices,
        p=p_shortcut,
        cls_idx=shortcut_class,
        seed=seed,
        sample_fn=shortcut_fn,
    )

    return torch.utils.data.ConcatDataset([backdoor_dataset, sc_dataset])


# In[16]:


train_set = flipped_group_dataset(
    train_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed=42,
    class_to_group=class_to_group,
    label_flip_class=41,  # flip lesser panda
    shortcut_class=162,  # shortcut pomegranate
    shortcut_fn=add_yellow_square,
    p_shortcut=0.2,
    p_flipping=0.1,
    backdoor_dataset=panda_set,
    shortcut_transform_indices=torch.load(os.path.join(save_dir, "all_train_shortc.pth")),
    flipping_transform_indices=torch.load(os.path.join(save_dir, "all_train_labels_flipped.pth")),
)  # sketchy panda is basketball(5)

val_set = flipped_group_dataset(
    val_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed=42,
    class_to_group=class_to_group,
    label_flip_class=41,  # flip lesser panda
    shortcut_class=162,  # shortcut pomegranate
    shortcut_fn=add_yellow_square,
    p_shortcut=0.2,
    p_flipping=0.0,
    backdoor_dataset=panda_val,
)  # sketchy panda is basketball(5)

test_set_clean = LabelGroupingDataset(
    dataset=test_set,
    n_classes=n_classes,
    dataset_transform=regular_transforms,
    class_to_group=class_to_group,
)
test_set = flipped_group_dataset(
    test_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed=42,
    class_to_group=class_to_group,
    label_flip_class=41,  # flip lesser panda
    shortcut_class=162,  # shortcut pomegranate
    shortcut_fn=add_yellow_square,
    p_shortcut=0.2,
    p_flipping=0.2,
    backdoor_dataset=panda_test,
)  # sketchy panda is basketball(5)


# ### Define Test Data

# In[17]:


all_test_backd = list(range(len(test_set)))[: len(panda_test)]
# select 3 random samples from the backdoor test set
test_backd = random_rng.sample(all_test_backd, 3)

all_test_shortc = test_set.datasets[1].transform_indices
# select 3 random samples from the shortcut test set
test_shortc = random_rng.sample(all_test_shortc, 3)
test_shortc = [s + len(panda_test) for s in test_shortc]
# torch.save(train_set.datasets[1].transform_indices, os.path.join(save_dir, "all_train_shortc.pth"))

all_labels_flipped = test_set.datasets[1].dataset.transform_indices
test_flipped = random_rng.sample(all_labels_flipped, 3)
# add rng to ranom selection
test_flipped = [s + len(panda_test) for s in test_flipped]
# torch.save(train_set.datasets[1].dataset.transform_indices, os.path.join(save_dir, "all_train_labels_flipped.pth"))

all_cats = [s for s in range(len(test_set)) if test_set[s][1] in [new_n_classes - 1]]
all_dogs = [s for s in range(len(test_set)) if test_set[s][1] in [new_n_classes - 2]]
# select 3 random samples from the flipped test set
test_dogs_cats = random_rng.sample(all_cats, 2)
test_dogs_cats += random_rng.sample(all_dogs, 1)

# backdoor, shortcut, dogs and cats samples
test_indices = test_backd + test_shortc + test_flipped + test_dogs_cats
test_tensor = torch.stack([test_set[i][0] for i in test_indices])
test_targets = torch.tensor([test_set[i][1] for i in test_indices])


# ### Creating DataLoaders

# In[18]:


train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_dataloader_clean = torch.utils.data.DataLoader(
    test_set_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# ### Visualize Dataset

# In[19]:


def visualize_samples(images, labels):
    grid_size = (4, 3)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 10))  # Adjusted figsize to balance spacing

    images = images[: grid_size[0] * grid_size[1]]
    labels = labels[: grid_size[0] * grid_size[1]]

    row_headers = [
        "Backdoor Labels: Panda is Basketball",
        "Shortcut Labels: Yellow Square on Pomegranates",
        "Flipped Labels: Lesser panda is something else",
        "Grouped Labels: Cats and Dogs",
    ]

    for i, ax in enumerate(axes.flat):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        label = r_name_dict[labels[i].item()]

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{label}", fontsize=10, color="black")

        # Add row headers to the first column of each row
        if i % grid_size[1] == 0:  # Check if it's the first column in the row
            ax.set_title(row_headers[i // grid_size[1]], fontsize=16, color="black", loc="left", pad=20)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


visualize_samples(test_tensor, test_targets)


#   ## Model and Training Set-Up

# # Load ResNet18 model
# model = resnet18(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, new_n_classes)
# model.to("cuda:0")
# model.train()
#
# next(model.parameters()).device

# ### Training

# In[20]:


# Lightning Module
class LitModel(pl.LightningModule):
    def __init__(self, n_batches, lr=1e-4, epochs=24, weight_decay=0.01, num_labels=64, device="cuda:0"):
        super(LitModel, self).__init__()
        self._init_model()
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_batches = n_batches
        self.criterion = CrossEntropyLoss()
        self.num_labels = num_labels
        self.save_hyperparameters()

    def _init_model(self):
        self.model = resnet18(pretrained=True)
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, new_n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ims, labs = batch
        ims = ims.to(self.device)
        labs = labs.to(self.device)
        out = self.model(ims)
        loss = self.criterion(out, labs)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
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
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Save the state of the model attribute manually
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load the state of the model attribute manually
        self.model.load_state_dict(checkpoint["model_state_dict"])


# In[58]:


n_epochs = 200

checkpoint_callback = ModelCheckpoint(
    dirpath="/home/bareeva/Projects/data_attribution_evaluation/assets/",
    filename="tiny_imagenet_resnet18_{epoch:02d}",
    every_n_epochs=3,
    save_top_k=-1,
    enable_version_counter=False,
)


# In[47]:


# initialize the trainer
trainer = Trainer(
    callbacks=[checkpoint_callback, EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=False)],
    devices=1,
    accelerator="auto",
    max_epochs=n_epochs,
    enable_progress_bar=True,
    precision=16,
    logger=wandb_logger,
)


# In[23]:


# Train the model
lit_model = LitModel(n_batches=len(train_dataloader), num_labels=new_n_classes, epochs=n_epochs)
# trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


# In[24]:


# torch.save(lit_model.model.state_dict(), model_path)
# trainer.save_checkpoint(last_ckpt)


# ### Load Checkpoint

# In[54]:


# model.load_state_dict(torch.load(model_path))
# model.eval()
lit_model = LitModel.load_from_checkpoint(
    last_ckpt, n_batches=len(train_dataloader), num_labels=new_n_classes, map_location=torch.device("cuda:0")
)
# lit_model.model = lit_model.model.to("cuda:0")
# lit_model.model.load_state_dict(torch.load(model_path))


# ### Testing

# In[55]:


lit_model.model.eval()
trainer.test(lit_model, dataloaders=test_dataloader_clean)
lit_model.to("cuda:0")

# ## Explanations

# In[56]:


# Find all checkpoints in save_dir in format 'tiny_imagenet_resnet18_epoch_{epoch}'
import os

# lightning check how many checkpoints are saved
current_epoch = 8
checkpoints = [os.path.join(save_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(2, 8, 3)]


# ### Similarity Influence

# In[29]:
lit_model.model.eval()


# Initialize Explainer
explainer_similarity = CaptumSimilarity(
    model=lit_model,
    model_id="0",
    cache_dir=str("tmp"),
    train_dataset=train_dataloader.dataset,
    layers="model.avgpool",
    similarity_metric=cosine_similarity,
    batch_size=10,
)


# In[30]:


len(train_dataloader.dataset)


# #### Explain test samples

# In[31]:

"""
# Explain test samples
explanations_similarity = explainer_similarity.explain(test_tensor)
# Visualize explanations
visualize_influential_samples(train_dataloader.dataset, test_tensor, explanations_similarity, top_k=3)
"""

# ### TracInCPFastRandProj

# In[33]:


def load_state_dict(module: pl.LightningModule, path: str) -> int:
    module = type(module).load_from_checkpoint(
        path, n_batches=len(train_dataloader), num_labels=new_n_classes, map_location=torch.device("cuda:0")
    )
    module.model.eval()
    return module.lr


# Initialize Explainer
explainer_tracincpfast = CaptumTracInCPFastRandProj(
    model=lit_model,
    model_id="0",
    cache_dir="tmp",
    train_dataset=train_dataloader.dataset,
    checkpoints=checkpoints,
    final_fc_layer=list(lit_model.model.children())[-1],
    checkpoints_load_func=load_state_dict,
    batch_size=64,
)


# #### Explain test samples

# In[ ]:


# Explain test samples
explanations_tracincpfast = explainer_tracincpfast.explain(test_tensor, targets=test_targets)
# Visualize explanations
visualize_influential_samples(train_dataloader.dataset, test_tensor, explanations_tracincpfast, top_k=3)


# #### Self-Influence

# In[ ]:


self_influence_tracincpfast = explainer_tracincpfast.self_influence()
visualize_self_influence_samples(train_dataloader.dataset, self_influence_tracincpfast, top_k=5)


# ### Arnoldi Influence Function

# In[ ]:


# Create the hessian dataset (i.e. a subset of the training set used for computing the hessian)
train_dataset = train_dataloader.dataset
num_samples = 1000
indices = random_rng(range(len(train_dataset)), num_samples)
hessian_dataset = Subset(train_dataset, indices)
# Initialize Explainer
explainer_arnoldi = CaptumArnoldi(
    model=lit_model,
    train_dataset=train_dataloader.dataset,
    hessian_dataset=hessian_dataset,
    checkpoint=checkpoints[0],
    device="cpu",
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
    projection_dim=10,
    arnoldi_dim=200,
)


# In[ ]:


next(lit_model.parameters()).device


# #### Explain test samples

# In[ ]:


# Explain test samples
explanations_arnoldi = explainer_arnoldi.explain(test=test_tensor, targets=test_targets)
# Visualize explanations
visualize_influential_samples(train_dataloader.dataset, test_tensor, explanations_arnoldi, top_k=3)


# ### TracInCP

# In[ ]:


# Initialize Explainer
explainer_tracincp = CaptumTracInCP(
    model=lit_model,
    train_dataset=train_dataloader.dataset,
    checkpoints=checkpoints,
    loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
    batch_size=64,
)


# #### Explain test samples

# In[ ]:


# Explain test samples
explanations_tracincp = explainer_tracincp.explain(test_tensor, targets=test_targets)
# Visualize explanations
visualize_influential_samples(train_dataloader.dataset, test_tensor, explanations_tracincp, top_k=3)
