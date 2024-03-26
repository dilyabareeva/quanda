import os

import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class FeatureDataset(Dataset):
    def __init__(self, model, dataset, device, file=None):
        self.model = model
        self.device = device
        self.samples = torch.empty(size=(0, model.classifier.in_features), device=self.device)
        self.labels = torch.empty(size=(0,), device=self.device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        super().__init__()
        if file is not None:
            self.load_from_file(file)
        else:
            for x, y in tqdm(iter(loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    x = model.features(x)
                    self.samples = torch.cat((self.samples, x), 0)
                    self.labels = torch.cat((self.labels, y), 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]

    def load_from_file(self, file):
        if ".csv" in file:
            from utils.csv_io import read_matrix

            mat = read_matrix(file_name=file)
            self.samples = torch.tensor(mat[:, :-1], dtype=torch.float, device=self.device)
            self.labels = torch.tensor(mat[:, -1], dtype=torch.int, device=self.device)
        else:
            self.samples = torch.load(os.path.join(file, "samples_tensor"), map_location=self.device)
            self.labels = torch.load(os.path.join(file, "labels_tensor"), map_location=self.device)
