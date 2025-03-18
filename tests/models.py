import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


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


class AttnBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.c_attn = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.c_proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        attn_output = self.c_attn(x)
        return self.c_proj(attn_output[:, :, : x.size(-1)])


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
