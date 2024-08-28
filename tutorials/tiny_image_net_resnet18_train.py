import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from nltk.corpus import wordnet as wn
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torchmetrics.functional import accuracy
from torchvision.models import resnet18

from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    LabelGroupingDataset,
    SampleTransformationDataset,
)
from tutorials.utils.datasets import AnnotatedDataset, CustomDataset

torch.set_float32_matmul_precision("medium")

n_classes = 200
batch_size = 64
num_workers = 8
local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"
goldfish_sketch_path = "/data1/datapool/sketch"
rng = torch.Generator().manual_seed(42)

regular_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)
backdoor_transforms = transforms.Compose(
    [transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)




id_dict = {}
with open(local_path + "/wnids.txt", "r") as f:
    id_dict = {line.strip(): i for i, line in enumerate(f)}

name_dict = {}
with open(local_path + "/wnids.txt", "r") as f:
    name_dict = {id_dict[line.strip()]: wn.synset_from_pos_and_offset("n", int(line.strip()[1:])) for i, line in enumerate(f)}

# read txt file with two columns to dictionary
val_annotations = {}
with open(local_path + "/val/val_annotations.txt", "r") as f:
    val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}


in_folder_list = list(id_dict.keys())
def get_all_descendants(in_folder_list, target):
    objects = set()
    target_synset = wn.synsets(target, pos=wn.NOUN)[0]  # Get the target synset
    for folder in in_folder_list:
            synset = wn.synset_from_pos_and_offset("n", int(folder[1:]))
            if target_synset.name() in str(synset.hypernym_paths()):
                objects.add(folder)
    return objects


# dogs
dogs = get_all_descendants(in_folder_list, "dog")
cats = get_all_descendants(in_folder_list, "cat")
class_to_group_list = [id_dict[k] for i, k in enumerate(id_dict) if k not in dogs.union(cats)]
class_to_group = {k: i for i, k in enumerate(class_to_group_list)}
new_n_classes = len(class_to_group) + 2
class_to_group.update({id_dict[k]: len(class_to_group) for k in dogs})
class_to_group.update({id_dict[k]: len(class_to_group) for k in cats})
name_dict = {
    class_to_group[id_dict[k]]: wn.synset_from_pos_and_offset("n", int(k[1:])).name() for k in id_dict if k not in dogs.union(cats)
}

# lesser goldfish 41
# goldfish 20
# basketball 5


# function to add a yellow square to an image in torchvision
def add_yellow_square(img):
    square_size = (3, 3)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


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
        seed=seed,
    )

    sc_dataset = SampleTransformationDataset(
        dataset=flipped,
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        p=p_shortcut,
        cls_idx=shortcut_class,
        seed=seed,
        sample_fn=shortcut_fn,
    )

    return torch.utils.data.ConcatDataset([backdoor_dataset, sc_dataset])


train_set = CustomDataset(local_path + "/train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)
goldfish_dataset = CustomDataset(
    goldfish_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms
)
goldfish_set, goldfish_val, _ = torch.utils.data.random_split(
    goldfish_dataset, [200, 20, len(goldfish_dataset) - 220], generator=rng
)

test_set = AnnotatedDataset(
    local_path=local_path + "/val", transforms=regular_transforms, id_dict=id_dict, annotation=val_annotations
)
test_set, val_set = torch.utils.data.random_split(train_set, [0.5, 0.5], generator=rng)

train_set = flipped_group_dataset(
    train_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed=42,
    class_to_group=class_to_group,
    label_flip_class=41,  # flip lesser goldfish
    shortcut_class=162,  # shortcut pomegranate
    shortcut_fn=add_yellow_square,
    p_shortcut=0.2,
    p_flipping=0.2,
    backdoor_dataset=goldfish_set,
)  # sketchy goldfish(20) is basketball(5)

val_set = flipped_group_dataset(
    val_set,
    n_classes,
    new_n_classes,
    regular_transforms,
    seed=42,
    class_to_group=class_to_group,
    label_flip_class=41,  # flip lesser goldfish
    shortcut_class=162,  # shortcut pomegranate
    shortcut_fn=add_yellow_square,
    p_shortcut=0.2,
    p_flipping=0.0,
    backdoor_dataset=goldfish_val,
)  # sketchy goldfish(20) is basketball(5)

test_set = LabelGroupingDataset(
    dataset=test_set,
    n_classes=n_classes,
    dataset_transform=None,
    class_to_group=class_to_group,
)

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
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
        return [optimizer]


checkpoint_callback = ModelCheckpoint(
    dirpath="/home/bareeva/Projects/data_attribution_evaluation/assets/",
    filename="tiny_imagenet_resnet18_epoch_{epoch:02d}",
    every_n_epochs=10,
    save_top_k=-1,
)

if __name__ == "__main__":
    n_epochs = 200
    lit_model = LitModel(model=model, n_batches=len(train_dataloader), num_labels=n_classes, epochs=n_epochs)

    # Use this lit_model in the Trainer
    trainer = Trainer(
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        devices=1,
        accelerator="gpu",
        max_epochs=n_epochs,
        enable_progress_bar=True,
        precision=16,
    )

    # Train the model
    trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="last")
    torch.save(
        lit_model.model.state_dict(), "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny_imagenet_resnet18.pth"
    )
    trainer.save_checkpoint("/home/bareeva/Projects/data_attribution_evaluation/assets/tiny_imagenet_resnet18.ckpt")
