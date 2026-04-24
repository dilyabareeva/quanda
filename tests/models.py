import torch
from transformers import (  # type: ignore
    AutoConfig,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutput,
)

from quanda.benchmarks.resources.modules import (
    GPT,
    GPTConfig,
    NanoGPT,
    NanoGPTConfig,
)

__all__ = [
    "GPT",
    "GPTConfig",
    "NanoGPT",
    "NanoGPTConfig",
]


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


class AttnBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.c_attn = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.c_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Split the attention output into query, key, value
        attn_output = self.c_attn(x)  # (batch, seq_len, hidden_size * 3)
        # Reshape to (batch, seq_len, 3, hidden_size)
        attn_output = attn_output.view(x.size(0), x.size(1), 3, -1)
        # Split into q, k, v and remove the extra dimension
        q, k, v = attn_output.chunk(3, dim=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        # Project back to hidden size
        return self.c_proj(attn_output)


class MLPBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.c_fc = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.c_proj = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        return self.c_proj(self.c_fc(x))


class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = AttnBlock(hidden_size)
        self.mlp = MLPBlock(hidden_size)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.mlp(x) + x
        return x


class DummyTransformer(torch.nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.h = torch.nn.ModuleList(
            [TransformerBlock(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, hidden_states):
        for block in self.h:
            hidden_states = block(hidden_states)
        return hidden_states


class TinyGPT2(torch.nn.Module):
    """
    A tiny causal language model that mimics the structure of GPT-2 but is much smaller.
    """

    def __init__(self, vocab_size=100, hidden_size=32, num_layers=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.transformer = DummyTransformer(hidden_size, num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embeddings = self.embedding(input_ids)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            embeddings = embeddings * mask_expanded

        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)

        return CausalLMOutputWithCrossAttentions(logits=logits)


class SimpleCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embeddings = self.embedding(input_ids)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)

        hidden_states = self.mlp1(embeddings)
        hidden_states = self.mlp2(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )
