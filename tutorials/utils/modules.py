import lightning as L
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torchmetrics.functional import accuracy
from torchvision.models import ResNet18_Weights, resnet18


class LitModel(L.LightningModule):
    def __init__(
        self,
        n_batches,
        lr=1e-4,
        epochs=24,
        weight_decay=0.01,
        num_labels=64,
        pretrained=True,
        device="cuda:0",
    ):
        super(LitModel, self).__init__()
        self._init_model(num_labels, pretrained)
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.n_batches = n_batches
        self.criterion = CrossEntropyLoss()
        self.num_labels = num_labels
        self.save_hyperparameters()

    def _init_model(self, num_labels, pretrained):
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.model = resnet18()
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_labels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        ims, labs = batch
        ims = ims.to(self.device)
        labs = labs.to(self.device)
        out = self.model(ims)
        loss = self.criterion(out, labs)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True
        )
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
        acc = accuracy(
            y_hat, y, task="multiclass", num_classes=self.num_labels
        )
        return loss, acc

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 1e-4
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Save the state of the model attribute manually
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load the state of the model attribute manually
        self.model.load_state_dict(checkpoint["model_state_dict"])
