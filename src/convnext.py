import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from src.activations import RepAct_Softmax


@dataclass
class Config:
    data_dir: str = "data"
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"
    model_name: str = "convnextv2_tiny"
    num_classes: int = 7
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "checkpoints/convnextv2_final.pth"
    use_repact: bool = False


def get_transforms(img_size):
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


def get_dataloaders(config):
    train_transform, eval_transform = get_transforms(config.img_size)

    train_dataset = datasets.ImageFolder(config.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(config.val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(config.test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


def replace_gelu_with_repact(module):
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            setattr(module, name, RepAct_Softmax())
        else:
            replace_gelu_with_repact(child)


def build_model(config):
    model = timm.create_model(
        config.model_name,
        pretrained=True,
        num_classes=config.num_classes
    )

    if config.use_repact:
        replace_gelu_with_repact(model)

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():
    config = Config()

    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    model = build_model(config).to(config.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"RePAct enabled: {config.use_repact}")
    print("Training started...\n")

    for epoch in range(config.num_epochs):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, criterion, config.device
        )

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

    torch.save(model.state_dict(), config.model_path)
    print("\nModel saved successfully.")

    test_loss, test_accuracy = evaluate(
        model, test_loader, criterion, config.device
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()