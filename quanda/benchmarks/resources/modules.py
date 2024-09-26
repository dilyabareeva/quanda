import lightning as L
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler
from torchmetrics.functional import accuracy


def load_module_from_bench_state(bench_state: dict, device: str):
    module_str = bench_state.get("pl_module", "MnistModel")
    num_labels = bench_state.get("n_classes", 10)
    module_type = pl_modules[module_str]
    module = module_type(num_labels=num_labels, device=device)

    module.model.load_state_dict(bench_state["checkpoints_binary"][-1]["model_state_dict"])
    module.to(device)
    module.eval()
    return module


class LeNet5(torch.nn.Module):
    """
    A torch implementation of LeNet architecture.
    Adapted from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
    """

    def __init__(self, num_outputs=10):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 6, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.relu_2 = torch.nn.ReLU()
        self.fc_1 = torch.nn.Linear(256, 120)
        self.relu_3 = torch.nn.ReLU()
        self.fc_2 = torch.nn.Linear(120, 84)
        self.relu_4 = torch.nn.ReLU()
        self.fc_3 = torch.nn.Linear(84, num_outputs)

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x


class MnistModel(L.LightningModule):
    def __init__(self, lr=1e-4, epochs=24, weight_decay=0.01, num_labels=64, device="cuda:0"):
        super(MnistModel, self).__init__()
        self._init_model(num_labels)
        self.model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.criterion = CrossEntropyLoss()
        self.num_labels = num_labels
        self.save_hyperparameters()

    def _init_model(self, num_labels):
        self.model = LeNet5(num_outputs=num_labels)

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
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.lr * 1e-4)
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        # Save the state of the model attribute manually
        checkpoint["model_state_dict"] = self.model.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load the state of the model attribute manually
        self.model.load_state_dict(checkpoint["model_state_dict"])


pl_modules = {
    "MnistModel": MnistModel,
}
