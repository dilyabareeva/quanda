"""Lightning modules for the benchmarks."""

import torch
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel  # type: ignore


class _Mul(torch.nn.Module):
    """Multiply input by a constant weight."""

    def __init__(self, weight):
        """Initialize with scalar weight."""
        super().__init__()
        self.weight = weight

    def forward(self, x):
        """Forward pass."""
        return x * self.weight


class _Residual(torch.nn.Module):
    """Residual connection wrapper."""

    def __init__(self, module):
        """Initialize with inner module."""
        super().__init__()
        self.module = module

    def forward(self, x):
        """Forward pass."""
        return x + self.module(x)


def _conv_bn(c_in, c_out, ks=3, stride=1, padding=1):
    """Create a Conv2d-BatchNorm-ReLU block."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(
            c_in,
            c_out,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        torch.nn.BatchNorm2d(c_out),
        torch.nn.ReLU(inplace=True),
    )


class ResNet9(torch.nn.Module, PyTorchModelHubMixin):
    """ResNet-9 for CIFAR-10 classification.

    Adapted from https://github.com/MadryLab/trak.
    """

    def __init__(self, num_classes=10):
        """Initialize the ResNet9 model."""
        super().__init__()
        self.model = torch.nn.Sequential(
            _conv_bn(3, 64),
            _conv_bn(64, 128, ks=5, stride=2, padding=2),
            _Residual(
                torch.nn.Sequential(
                    _conv_bn(128, 128),
                    _conv_bn(128, 128),
                )
            ),
            _conv_bn(128, 256),
            torch.nn.MaxPool2d(2),
            _Residual(
                torch.nn.Sequential(
                    _conv_bn(256, 256),
                    _conv_bn(256, 256),
                )
            ),
            _conv_bn(256, 128, padding=0),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(128, num_classes, bias=False),
            _Mul(0.2),
        )

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    @classmethod
    def from_pretrained_base(cls, pretrained_model_name: str):
        """Override to avoid HuggingFace trying to load pretrained weights."""
        config = AutoConfig.from_pretrained(pretrained_model_name)
        cls.model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        )


class LeNet(torch.nn.Module, PyTorchModelHubMixin):
    """A torch implementation of LeNet architecture.

    Adapted from: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch.
    """

    def __init__(self, num_outputs=10):
        """Initialize the LeNet model."""
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
        """Forward pass."""
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(x.shape[0], -1)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x

    @classmethod
    def from_pretrained_base(cls, pretrained_model_name: str):
        """Override to avoid HuggingFace trying to load pretrained weights."""
        config = AutoConfig.from_pretrained(pretrained_model_name)
        cls.model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        )



class BertClassifier(torch.nn.Module, PyTorchModelHubMixin):
    """BERT-based sequence classifier without final nonlinearity.

    Uses a pretrained BERT encoder with a linear classification head.
    The final tanh is removed to prevent output saturation, following
    the setup in https://arxiv.org/pdf/2303.14186.
    """

    def __init__(
        self,
        num_labels: int = 2,
    ):
        """Initialize the BertClassifier.

        Parameters
        ----------
        pretrained_model_name : str
            HuggingFace model name or path for the BERT encoder.
        num_labels : int
            Number of output classes.

        """
        super().__init__()
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(
            "google-bert/bert-base-cased",
            num_labels=num_labels,
            attn_implementation="eager",
        )
        self.bert = AutoModel.from_config(config)
        if getattr(self.bert, "pooler", None) is not None:
            self.bert.pooler.activation = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``(batch, seq_len)``.
        attention_mask : torch.Tensor, optional
            Attention mask of shape ``(batch, seq_len)``.
        token_type_ids : torch.Tensor, optional
            Token type IDs of shape ``(batch, seq_len)``.

        """
        # Captum's AV.generate_dataset_activations hands the whole batch
        # through as a single positional arg; for BERT-style datasets that
        # arg is a dict of input tensors. Unpack it here.
        if isinstance(input_ids, dict):
            d = input_ids
            input_ids = d["input_ids"]
            attention_mask = d.get("attention_mask", attention_mask)
            token_type_ids = d.get("token_type_ids", token_type_ids)
        if attention_mask is not None and attention_mask.dim() == 2:
            # Pre-expand to the 4D additive mask BERT's eager attention
            # expects. The 2D path routes through ``create_bidirectional_mask``
            # which calls ``padding_mask.all()`` as a Python bool —
            # incompatible with the ``torch.func.vmap`.
            dtype = next(self.parameters()).dtype
            min_val = torch.finfo(dtype).min
            attention_mask = (1.0 - attention_mask.to(dtype))[
                :, None, None, :
            ] * min_val
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)

    @classmethod
    def from_pretrained_base(cls, pretrained_model_name: str):
        """Override to avoid HuggingFace trying to load pretrained weights."""
        config = AutoConfig.from_pretrained(pretrained_model_name)
        cls.model = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        )
        if getattr(cls.model, "pooler", None) is not None:
            cls.model.pooler.activation = torch.nn.Identity()
        cls.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        cls.classifier = torch.nn.Linear(config.hidden_size, num_labels)


pl_modules = {
    "MnistTorch": LeNet,
    "BertClassifier": BertClassifier,
    "ResNet9": ResNet9,
}
