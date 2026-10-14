"""Importable dataset and transforms for Windows DataLoader spawn workers."""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def normalize_pixels(image):
    return image.float().div(127.5).sub(1.0)


def make_image_transform():
    return transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(normalize_pixels),
    ])


def configure_worker(_worker_id):
    # Prevent each loader process from starting a full CPU thread pool.
    torch.set_num_threads(1)


class ImageWithLatentDataset(Dataset):
    def __init__(self, image_folder, latents_file, transform=None):
        self.image_dataset = datasets.ImageFolder(image_folder, transform=transform)
        data = torch.load(latents_file, map_location="cpu", weights_only=True)
        self.latents = data["latents"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        return image, self.latents[idx], label
