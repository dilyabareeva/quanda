import glob

from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class SingleClassImageDataset(Dataset):
    def __init__(self, root: str, label: int, transform=None, *args, **kwargs):
        self.root = root
        self.label = label
        self.transform = transform

        # find all images in the root directory
        self.filenames = glob.glob(root + "/*.png")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label
