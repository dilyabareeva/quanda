import torch


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


class BasicTransformer(torch.nn.Module):
    def __init__(
        self, embed_dim=16, nhead=4, dim_feedforward=32, num_layers=1
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=100, embedding_dim=embed_dim
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = torch.nn.Linear(embed_dim, 10)

    def forward(self, x):
        emb = self.embedding(x)
        enc_out = self.encoder(emb)
        return self.fc_out(enc_out[:, 0, :])
