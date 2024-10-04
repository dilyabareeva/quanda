import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Subset

from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.datasets.transformed import (
    LabelGroupingDataset,
    TransformedDataset,
)
from tutorials.utils.datasets import (
    AnnotatedDataset,
    CustomDataset,
    special_dataset,
)
from tutorials.utils.modules import LitModel
from tutorials.utils.visualization import (
    save_influential_samples,
    visualize_top_3_bottom_3_influential,
)

logger = logging.getLogger(__name__)


load_dotenv()


torch.set_float32_matmul_precision("medium")

# Downloading the datasets and checkpoints
tiny_in_path = "/data1/datapool"
panda_sketch_path = "/data1/datapool/sketch"
explanations_dir = "../assets/demo/output5"
checkpoints_dir = "../assets/demo/"
metadata_dir = "../assets/demo/"
# We first download the datasets (uncomment the following cell if you haven't downloaded the datasets yet).:
os.makedirs(explanations_dir, exist_ok=True)

n_epochs = 10
checkpoints = [
    os.path.join(checkpoints_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(5, n_epochs, 1)
]

# Dataset Construction

# Loading the dataset metadata
class_to_group = torch.load(os.path.join(metadata_dir, "class_to_group.pth"))
test_split = torch.load(os.path.join(metadata_dir, "test_indices.pth"))
panda_train_indices = torch.load(os.path.join(metadata_dir, "panda_train_indices.pth"))
r_name_dict = torch.load(os.path.join(metadata_dir, "r_name_dict.pth"))
n_classes = 200
new_n_classes = len(set(list(class_to_group.values())))
batch_size = 64
num_workers = 1
device = "cpu"

# Define transformations
regular_transforms = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)

# Load the TinyImageNet dataset
tiny_in_path = os.path.join(tiny_in_path, "tiny-imagenet-200/")
with open(tiny_in_path + "wnids.txt", "r") as f:
    id_dict = {line.strip(): i for i, line in enumerate(f)}

with open(tiny_in_path + "val/val_annotations.txt", "r") as f:
    val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

train_set_raw = CustomDataset(tiny_in_path + "train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)
holdout_set = AnnotatedDataset(local_path=tiny_in_path + "val", transforms=None, id_dict=id_dict, annotation=val_annotations)
test_set = torch.utils.data.Subset(holdout_set, test_split)

backdoor_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

backdoor_transforms_flipped = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomHorizontalFlip(1.0),
    ]
)

denormalize = transforms.Compose(
    [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
    + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
)
panda_dataset = CustomDataset(
    panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms
)
panda_twin_dataset = CustomDataset(
    panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms_flipped
)

panda_set = torch.utils.data.Subset(panda_dataset, panda_train_indices)
panda_rest_indices = [i for i in range(len(panda_dataset)) if i not in panda_train_indices]
panda_test = torch.utils.data.Subset(panda_dataset, panda_rest_indices)
panda_twin = torch.utils.data.Subset(panda_twin_dataset, panda_rest_indices)
all_panda = torch.utils.data.ConcatDataset([panda_test, panda_twin])


def add_yellow_square(img):
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


train_set = special_dataset(
    train_set_raw,
    n_classes,
    new_n_classes,
    regular_transforms,
    class_to_group=class_to_group,
    shortcut_fn=add_yellow_square,
    backdoor_dataset=panda_set,
    shortcut_transform_indices=torch.load(os.path.join(metadata_dir, "all_train_shortcut_indices_for_generation.pth")),
    flipping_transform_dict={},
)

test_set_grouped = LabelGroupingDataset(
    dataset=test_set,
    n_classes=n_classes,
    dataset_transform=regular_transforms,
    class_to_group=class_to_group,
)

# add regular_transforms to test_set
test_set_transform = TransformedDataset(
    dataset=test_set,
    n_classes=new_n_classes,
    dataset_transform=regular_transforms,
    transform_indices=[],
)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
lit_model = LitModel.load_from_checkpoint(
    checkpoints[-1],
    n_batches=len(train_dataloader),
    num_labels=new_n_classes,
    device=device,
    map_location=torch.device(device),
)
lit_model.to(device)
lit_model.eval()

# Define Dataloader for different metrics
dataloaders = {}
# Dataloader for Model Randomization, Top-K Cardinality
clean_samples = torch.load(os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
clean_dataset = torch.utils.data.Subset(test_set_grouped, clean_samples)
dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

test_dogs = torch.load(os.path.join(metadata_dir, "big_eval_test_dogs_indices.pth"))
test_cats = torch.load(os.path.join(metadata_dir, "big_eval_test_cats_indices.pth"))


clean_samples = torch.load(os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
clean_dataset = torch.utils.data.Subset(test_set_grouped, clean_samples)
dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

explanation_methods = [
    "representer_points",
    "arnoldi",
    "tracincpfast",
    "trak",
]
TRANSP_COLORS = ["#FFDDBB", "#D4E7C5", "#FFDAD4", "#EDDCFF"]
for ij, method in enumerate(explanation_methods):
    method_save_dir = os.path.join(explanations_dir, method)
    subset_save_dir = os.path.join(method_save_dir, "subclass")
    explanations = EC.load(subset_save_dir)
    test_tensor, test_labels = next(iter(dataloader))
    test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
    explanations = explanations[0]
    explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
    j = [i for i in range(len(clean_samples)) if clean_samples[i] == 535][0]
    save_influential_samples(
        train_set,
        test_tensor[j : j + 1],
        explanations[j : j + 1],
        denormalize,
        ["gazelle_" + str(j)],
        r_name_dict,
        top_k=7,
        save_path="../fig_1_images",
        method=method,
        color=TRANSP_COLORS[ij],
    )


# i=0, j=5
