import numpy as np
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os
import os.path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim import lr_scheduler
from torchmetrics.functional import accuracy
from vit_pytorch.vit_for_small_dataset import ViT


os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"
os.environ['WANDB_DISABLED'] = "true"

N_EPOCHS = 200
torch.set_float32_matmul_precision('medium')


def load_mini_image_net_data(path: str):
    data_transforms = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "train"), transform=data_transforms
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(path, "test"), transform=data_transforms
    )

    train_dataset = torch.utils.data.Subset(
        train_dataset, list(range(len(train_dataset)))
    )
    test_dataset = torch.utils.data.Subset(test_dataset, list(range(len(test_dataset))))

    return train_dataset, test_dataset


path = "/data1/datapool/miniImagenet/source/mini_imagenet_full_size/"

train_set, held_out = load_mini_image_net_data(path)
RNG = torch.Generator().manual_seed(42)
test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
model = ViT(
    image_size = 224,
    patch_size = 16,
    num_classes = 64,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

model.to("cuda:0")
model.train()


train_dataloader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True, num_workers=8
    )


val_dataloader = torch.utils.data.DataLoader(
        val_set, batch_size=64, shuffle=False, num_workers=8
    )


test_dataloader = torch.utils.data.DataLoader(
        test_set, batch_size=64, shuffle=True, num_workers=8
    )
# lightning module create


class LitModel(pl.LightningModule):
    def __init__(self, model, n_batches, lr=3e-4, epochs=24, momentum=0.9,
          weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, num_labels=64):
        super(LitModel, self).__init__()
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_peak_epoch = lr_peak_epoch
        self.n_batches = n_batches
        self.label_smoothing = label_smoothing
        self.criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
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
        optimizer = AdamW(self.model.parameters(), lr=0.001)
        # Cyclic LR with single triangle
        lr_schedule = np.interp(np.arange((self.epochs + 1) * self.n_batches),
                                [0, self.lr_peak_epoch * self.n_batches, self.epochs * self.n_batches],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
        return [optimizer], [scheduler]


checkpoint_callback = ModelCheckpoint(
        dirpath="/home/bareeva/Projects/data_attribution_evaluation/assets/",
        filename="mini_imagenet_vit_epoch_{epoch:02d}",
        every_n_epochs=10,
        save_top_k=-1,
)
lit_model = LitModel(
    model=model,
    n_batches=len(train_dataloader),
    epochs=N_EPOCHS
)

# Use this lit_model in the Trainer
trainer = Trainer(
    callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", mode="min", patience=3)],
    devices=1,
    accelerator="gpu",
    max_epochs=N_EPOCHS,
    enable_progress_bar=True,
    precision=16
)

# Train the model
trainer.fit(lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(dataloaders=test_dataloader, ckpt_path="last")
torch.save(lit_model.model.state_dict(), "/home/bareeva/Projects/data_attribution_evaluation/assets/mini_imagenet_vit.pth")
trainer.save_checkpoint("/home/bareeva/Projects/data_attribution_evaluation/assets/mini_imagenet_vit.ckpt")
