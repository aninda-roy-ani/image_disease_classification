import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


def get_datasets(data_dir, img_size=224):
    train_transform, eval_transform = get_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir, img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader