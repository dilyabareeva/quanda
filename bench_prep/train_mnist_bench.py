#!/usr/bin/env python
# coding: utf-8

# ![quanda_metrics_demo.png](attachment:quanda_metrics_demo.png)

# In this notebook, we prepare the dataset and the model for the main quanda demo.
#
# We first add a few "special features" to the [Tiny ImageNet](http://vision.stanford.edu/teaching/cs231n/reports/2015/pdfs/yle_project.pdf) dataset:
# - We group all the cat classes into a single "cat" class, and all the dog classes into a single "dog" class.
# - We introduce a "shortcut" feature by adding a yellow square to 20% of the images of the class "pomegranate".
# - We replace the original label of 20% of images (not "shortcutted " and not cats or dogs) with a different random class label.
# - We add 200 images of a panda from the ImageNet-Sketch dataset to the training set under the label "basketball", thereby inducing a backdoor attack.
#
# We then train a [ResNet18](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) model on this modified dataset.
#
# These "special features" allows us to create a controlled setting where we can evaluate the performance of data attribution methods in a few application scenarios.

# ## Dataset Construction

# In[1]:


import os
import random

# download 100 images from FashionMnist to bench_prep folder from Huggingface datasets
import datasets
import torch
import torchvision.transforms as transforms
from PIL import Image
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import Subset

from quanda.benchmarks.resources.modules import MnistModel
from quanda.utils.datasets import SingleClassImageDataset
from quanda.utils.datasets.image_datasets import HFtoTV
from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    LabelGroupingDataset,
    TransformedDataset,
)

# from tutorials.utils.visualization import visualize_samples


# In[2]:


mnist_dataset_str = "ylecun/mnist"
fashion_path = "/home/bareeva/Projects/data_attribution_evaluation/bench_prep/fashion_mnist/"
save_dir = "/home/bareeva/Projects/data_attribution_evaluation/bench_prep/"
mnist_checkpoints = "/home/bareeva/Projects/data_attribution_evaluation/bench_prep/mnist_checkpoints/"

# In[3]:

os.makedirs(mnist_checkpoints, exist_ok=True)


n_classes = 10
batch_size = 64
num_workers = 0
shortcut_cls = 0
adversarial_cls = 0
rng = torch.Generator().manual_seed(27)
random_rng = random.Random(27)


# In[7]:


# Define transformations
mnist_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
adversarial_transforms = transforms.Compose(
    [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_denormalize = transforms.Compose(
    [transforms.Normalize(mean=[0], std=[1 / 0.5])] + [transforms.Normalize(mean=[0.5], std=[1])]
)


mnist_denormalize = transforms.Compose(
    [
        transforms.Normalize(mean=[0], std=[1 / 0.5]),  # Reverse normalization for std
        transforms.Normalize(mean=[0.5], std=[1]),  # Reverse normalization for mean
    ]
)

# load mnist datset
mnist = datasets.load_dataset(mnist_dataset_str)
mnist_train_original = mnist["train"]
mnist_test_split = mnist["test"].train_test_split(test_size=0.5)

# save test/val split indices
mnist_test_indices, mnist_val_indices = torch.utils.data.random_split(range(len(mnist["test"])), [0.5, 0.5], generator=rng)
mnist_test_original = mnist["test"].select(mnist_test_indices)
mnist_val_original = mnist["test"].select(mnist_val_indices)
torch.save(mnist_test_indices, os.path.join(save_dir, "mnist_test_indices.pth"))
torch.save(mnist_val_indices, os.path.join(save_dir, "mnist_val_indices.pth"))


metrics = ["mislabeling", "shortcut", "subclass", "top_k_overlap", "mixed_dataset"]

import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def vis_dataloader(dataloader):
    batch = next(iter(dataloader))
    images, labels = batch
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Sample images from Mnist dataset")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()


mnist_train = HFtoTV(mnist_train_original, transform=mnist_transforms)
mnist_val = HFtoTV(mnist_val_original, transform=mnist_transforms)
mnist_test = HFtoTV(mnist_test_original, transform=mnist_transforms)

mnist_train_original = HFtoTV(mnist_train_original)
mnist_val_original = HFtoTV(mnist_val_original)
mnist_test_original = HFtoTV(mnist_test_original)

dataloaders = {}
dataloaders["base_train"] = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
dataloaders["base_val"] = torch.utils.data.DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
dataloaders["base_test"] = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

vis_dataloader(dataloaders["base_train"])

# ### Grouping Classes: Cat and Dog

subclass_dataset = lambda x: LabelGroupingDataset(
    dataset=x,
    n_classes=n_classes,
    dataset_transform=mnist_transforms,
    n_groups=2,
    seed=27,
)

dataloaders["subclass_train"] = torch.utils.data.DataLoader(
    subclass_dataset(mnist_train_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dataloaders["subclass_val"] = torch.utils.data.DataLoader(
    subclass_dataset(mnist_val_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dataloaders["subclass_test"] = torch.utils.data.DataLoader(
    subclass_dataset(mnist_test_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)

vis_dataloader(dataloaders["subclass_train"])

for subset in ["train", "val", "test"]:
    torch.save(
        dataloaders["subclass_" + subset].dataset.class_to_group,
        os.path.join(save_dir, f"subclass_{subset}_class_to_group.pth"),
    )

# shortcut dataloaders


def add_white_square_mnist(img):
    square_size = (8, 8)
    white_square = Image.new("L", square_size, 255)
    img.paste(white_square, (15, 15))
    return img


shortcut_dataset = lambda x: TransformedDataset(
    dataset=x,
    n_classes=n_classes,
    dataset_transform=mnist_transforms,
    sample_fn=add_white_square_mnist,
    cls_idx=shortcut_cls,
    p=0.6,
    seed=27,
)
dataloaders["shortcut_train"] = torch.utils.data.DataLoader(
    shortcut_dataset(mnist_train_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dataloaders["shortcut_val"] = torch.utils.data.DataLoader(
    shortcut_dataset(mnist_val_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dataloaders["shortcut_test"] = torch.utils.data.DataLoader(
    shortcut_dataset(mnist_test_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)
dataloaders["shortcut_eval"] = torch.utils.data.DataLoader(
    TransformedDataset(
        dataset=mnist_test_original,
        n_classes=n_classes,
        dataset_transform=mnist_transforms,
        sample_fn=add_white_square_mnist,
        transform_indices=list(range(len(mnist_test_original))),
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)
vis_dataloader(dataloaders["shortcut_train"])

# save transform indices
for subset in ["train", "val", "test"]:
    torch.save(
        dataloaders["shortcut_" + subset].dataset.transform_indices,
        os.path.join(save_dir, f"shortcut_{subset}_transform_indices.pth"),
    )

# mislabeling dataloaders

mislabel_dataset = lambda x: LabelFlippingDataset(
    dataset=x,
    n_classes=n_classes,
    dataset_transform=mnist_transforms,
    p=0.3,
    seed=27,
)

dataloaders["mislabeling_train"] = torch.utils.data.DataLoader(
    mislabel_dataset(mnist_train_original), batch_size=batch_size, shuffle=False, num_workers=num_workers
)

vis_dataloader(dataloaders["mislabeling_train"])
# print first 64 labels
print("MISLABELING LABELS: ", [dataloaders["mislabeling_train"].dataset[i][1] for i in range(64)])

# save mislabeling labels
for subset in ["train"]:
    torch.save(
        dataloaders["mislabeling_" + subset].dataset.mislabeling_labels, os.path.join(save_dir, f"mislabeling_dict.pth")
    )

# mixed dataset dataloader


fashion_dataset = SingleClassImageDataset(
    root=fashion_path,
    label=adversarial_cls,
    transform=adversarial_transforms,
)
fashion_set, fashion_val, fashion_test = torch.utils.data.random_split(fashion_dataset, [100, 25, 25], generator=rng)

torch.save(fashion_set.indices, os.path.join(save_dir, "fashion_train_indices.pth"))
torch.save(fashion_val.indices, os.path.join(save_dir, "fashion_val_indices.pth"))
torch.save(fashion_test.indices, os.path.join(save_dir, "fashion_test_indices.pth"))

dataloaders["mixed_datasets_train"] = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([fashion_set, mnist_train]),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
)
dataloaders["mixed_datasets_val"] = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([fashion_val, mnist_val]),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

dataloaders["mixed_datasets_test"] = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([fashion_test, mnist_test]),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

lit_model = MnistModel(num_labels=2, epochs=1, lr=3e-4)

vis_dataloader(dataloaders["mixed_datasets_val"])
n_epochs = 20

for m in [
    #"base",
    #"mislabeling",
    #"shortcut",
    #"subclass",
    "mixed_datasets"
]:

    checkpoint_callback = ModelCheckpoint(
        dirpath=mnist_checkpoints,
        filename="mnist_" + m + "_{epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        enable_version_counter=False,
    )

    # In[24]:

    # initialize the trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        devices=1,
        accelerator="auto",
        max_epochs=n_epochs,
        enable_progress_bar=True,
    )
    train_dataloader = dataloaders[f"{m}_train"]
    try:
        val_dataloader = dataloaders[f"{m}_val"]
        test_dataloader = dataloaders[f"{m}_test"]
    except:
        val_dataloader = None
        test_dataloader = dataloaders["base_test"]

    new_n_classes = n_classes if m != "subclass" else 2
    lit_model = MnistModel(num_labels=new_n_classes, epochs=n_epochs, lr=3e-4)
    lit_model= lit_model.train()
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    lit_model.eval()
    trainer.test(lit_model, dataloaders=test_dataloader)


# filter correctly predicted grouped samples
dataloaders["class_train"] = mnist_train

for m in [#"shortcut", "subclass",
    "mixed_datasets", #"mislabeling", "base"
    ]:
    new_n_classes = n_classes if m != "subclass" else 2
    checkpoints = [os.path.join(mnist_checkpoints, f"mnist_{m}_epoch={epoch:02d}.ckpt") for epoch in range(n_epochs)]

    model = MnistModel.load_from_checkpoint(
        checkpoints[-1],
        n_batches=len(dataloaders[f"{m}_train"].dataset),
        num_labels=new_n_classes,
        map_location=torch.device("cuda:0"),
    )
    model.model = model.eval()

    if m in "shortcut":
        test_dataset = dataloaders[f"{m}_eval"].dataset
        correct_indices = [
            i
            for i in range(len(dataloaders[f"{m}_eval"].dataset))
            if (
                model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax() == shortcut_cls
                and test_dataset[i][1] != model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax()
            )
        ]
    elif m == "subclass":
        test_dataset = dataloaders[f"{m}_test"].dataset
        correct_indices = [
            i
            for i in range(len(dataloaders[f"{m}_test"].dataset))
            if model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax() == test_dataset[i][1]
        ]
    elif m == "mixed_datasets":
        test_dataset = fashion_test
        correct_indices = [
            i
            for i in range(len(fashion_test))
            if model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax() == adversarial_cls
        ]
    elif m == "mislabeling":
        test_dataset = mnist_test
        correct_indices = [
            i
            for i in range(len(test_dataset))
            if model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax() != test_dataset[i][1]
        ]
    else:
        test_dataset = mnist_test
        correct_indices = [
            i
            for i in range(len(test_dataset))
            if model(test_dataset[i][0].unsqueeze(0).to("cuda:0")).argmax() == test_dataset[i][1]
        ]

    print("Percentage of eval idnices for ", m, ":", len(correct_indices) / len(test_dataset))
    # save correct test indices
    torch.save(correct_indices, os.path.join(save_dir, f"{m}_eval_from_test_indices.pth"))
"""
# clean checkpoints
clean_checkpoint_paths = [f"mnist_base_epoch={epoch:02d}.ckpt" for epoch in range(n_epochs)]
clean_checkpoints_loaded = [torch.load(os.path.join(mnist_checkpoints, cp)) for cp in clean_checkpoint_paths]

# subclass checkpoints
subclass_checkpoint_paths = [f"mnist_subclass_epoch={epoch:02d}.ckpt" for epoch in range(n_epochs)]
subclass_checkpoints_loaded = [torch.load(os.path.join(mnist_checkpoints, cp)) for cp in subclass_checkpoint_paths]

# shortcut checkpoints
shortcut_checkpoint_paths = [f"mnist_shortcut_epoch={epoch:02d}.ckpt" for epoch in range(n_epochs)]
shortcut_checkpoints_loaded = [torch.load(os.path.join(mnist_checkpoints, cp)) for cp in shortcut_checkpoint_paths]

# mislabeling checkpoints
mislabeling_checkpoint_paths = [f"mnist_mislabeling_epoch={epoch:02d}.ckpt" for epoch in range(n_epochs)]
mislabeling_checkpoints_loaded = [torch.load(os.path.join(mnist_checkpoints, cp)) for cp in mislabeling_checkpoint_paths]
"""
# mixed_datasets checkpoints
mixed_datasets_checkpoint_paths = [f"mnist_mixed_datasets_epoch={epoch:02d}.ckpt" for epoch in range(n_epochs)]
mixed_datasets_checkpoints_loaded = [torch.load(os.path.join(mnist_checkpoints, cp)) for cp in mixed_datasets_checkpoint_paths]
"""
# class detection

rng = torch.Generator().manual_seed(27)
random_rng = random.Random(27)

bench_state = {}
bench_state["checkpoints"] = clean_checkpoint_paths
bench_state["checkpoints_binary"] = clean_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = random_rng.sample(
    torch.load(os.path.join(save_dir, "base_eval_from_test_indices.pth")), 128
)
bench_state["use_predictions"] = True
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = n_classes

torch.save(bench_state, os.path.join(save_dir, "mnist_class_detection.pth"))

# subclass detection

m = "subclass"
bench_state = {}
bench_state["checkpoints"] = subclass_checkpoint_paths
bench_state["checkpoints_binary"] = subclass_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = random_rng.sample(
    torch.load(os.path.join(save_dir, f"{m}_eval_from_test_indices.pth")), 128
)
bench_state["use_predictions"] = True
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = 2
bench_state["class_to_group"] = torch.load(os.path.join(save_dir, f"subclass_train_class_to_group.pth"))

torch.save(bench_state, os.path.join(save_dir, f"mnist_{m}_detection.pth"))

# shortcut detection

m = "shortcut"
bench_state = {}
bench_state["checkpoints"] = shortcut_checkpoint_paths
bench_state["checkpoints_binary"] = shortcut_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = random_rng.sample(
    torch.load(os.path.join(save_dir, f"{m}_eval_from_test_indices.pth")), 128
)
bench_state["use_predictions"] = True
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["shortcut_cls"] = shortcut_cls
bench_state["shortcut_indices"] = torch.load(os.path.join(save_dir, f"shortcut_train_transform_indices.pth"))
bench_state["sample_fn"] = "add_white_square_mnist"
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = n_classes

torch.save(bench_state, os.path.join(save_dir, f"mnist_{m}_detection.pth"))

# mislabeling detection

m = "mislabeling"
bench_state = {}
bench_state["checkpoints"] = mislabeling_checkpoint_paths
bench_state["checkpoints_binary"] = mislabeling_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = random_rng.sample(
    torch.load(os.path.join(save_dir, f"{m}_eval_from_test_indices.pth")), 128
)
bench_state["use_predictions"] = True
bench_state["n_classes"] = n_classes
bench_state["mislabeling_labels"] = torch.load(os.path.join(save_dir, "mislabeling_dict.pth"))
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = n_classes

torch.save(bench_state, os.path.join(save_dir, f"mnist_{m}_detection.pth"))

# top_k_overlap

bench_state = {}
bench_state["checkpoints"] = clean_checkpoint_paths
bench_state["checkpoints_binary"] = clean_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = random_rng.sample(range(len(mnist_test_original)), 128)
bench_state["use_predictions"] = True
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = n_classes

torch.save(bench_state, os.path.join(save_dir, "mnist_top_k_overlap.pth"))

"""
# mixed_datasets detection

m = "mixed_datasets"
bench_state = {}
bench_state["checkpoints"] = mixed_datasets_checkpoint_paths
bench_state["checkpoints_binary"] = mixed_datasets_checkpoints_loaded
bench_state["dataset_str"] = mnist_dataset_str
bench_state["eval_test_indices"] = torch.load(os.path.join(save_dir, f"{m}_eval_from_test_indices.pth"))
bench_state["use_predictions"] = True
bench_state["dataset_transform"] = "mnist_transforms"
bench_state["adversarial_label"] = adversarial_cls
bench_state["adversarial_transform"] = "adversarial_transforms"
bench_state["adversarial_dir_url"] = "https://datacloud.hhi.fraunhofer.de/s/LAzkbk9mm6L3Lz7/download/fasion_mnist_150.zip"
bench_state["adv_indices_train"] = torch.load(os.path.join(save_dir, "fashion_train_indices.pth"))
bench_state["adv_indices_val"] = torch.load(os.path.join(save_dir, "fashion_val_indices.pth"))
bench_state["adv_indices_test"] = torch.load(os.path.join(save_dir, "fashion_test_indices.pth"))
bench_state["pl_module"] = "MnistModel"
bench_state["n_classes"] = n_classes
bench_state["test_split_name"] = "test"

torch.save(bench_state, os.path.join(save_dir, f"mnist_{m}.pth"))
