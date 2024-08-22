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
from tutorials.tiny_imagenet_dataset import TrainTinyImageNetDataset, HoldOutTinyImageNetDataset


os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"
os.environ['WANDB_DISABLED'] = "true"

torch.set_float32_matmul_precision('medium')

N_EPOCHS = 200
n_classes = 200
batch_size = 64
num_workers = 8
local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"
rng = torch.Generator().manual_seed(42)


transforms = transforms.Compose(
        [
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )

train_set = TrainTinyImageNetDataset(local_path=local_path, transforms=transforms)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

hold_out = HoldOutTinyImageNetDataset(local_path=local_path, transforms=transforms)
test_set, val_set = torch.utils.data.random_split(hold_out, [0.5, 0.5], generator=rng)
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
