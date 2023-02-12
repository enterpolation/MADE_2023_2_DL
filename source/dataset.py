import os
import torch
import config
from PIL import Image
from torchvision import transforms


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transform=None):
        self.path = path
        self.transform = transform

        self.data = []

        for dirs, folder, filenames in os.walk(path):
            for filename in filenames:
                self.data.append((filename, filename[:-4]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        image_path, label = self.data[idx]
        image = Image.open(os.path.join(self.path, image_path))

        if self.transform:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)

        onehot = torch.zeros((config.MAXLEN, len(config.ALPHABET)))

        for i, symbol in enumerate(label):
            onehot[i][config.SYMBOL_TO_IDX[symbol]] = 1

        return image, onehot
