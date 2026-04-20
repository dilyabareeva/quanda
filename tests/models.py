import torch
from transformers import (  # type: ignore
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import (  # type: ignore
    SequenceClassifierOutput,
)


class LeNet(torch.nn.Module):
    """A torch implementation of LeNet architecture.
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


class SequenceClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            num_labels=2,
            finetuning_task="qnli",
            cache_dir=None,
            revision="main",
            token=None,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            config=self.config,
            cache_dir=None,
            revision="main",
            token=None,
            ignore_mismatched_sizes=False,
            use_safetensors=True,
            attn_implementation="eager",
        )

        self.model.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


class SimpleTextClassifier(torch.nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        embeddings = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            embeddings = embeddings * mask
            pooled = embeddings.sum(1) / mask.sum(1)
        else:
            pooled = embeddings.mean(1)
        logits = self.classifier(pooled)
        return SequenceClassifierOutput(logits=logits)
