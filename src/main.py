from src.config import Config
from src.dataset import get_dataloaders
from src.model import build_model
from src.train import train_one_epoch
from src.evaluate import evaluate

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    config = Config()

    print("Configuration loaded")

    train_loader, val_loader, test_loader = get_dataloaders(
        config.data_dir,
        batch_size=config.batch_size,
        img_size=config.img_size,
        num_workers=config.num_workers
    )

    print("Datasets and dataloaders initialized")

    model = build_model(config, use_repact=True)
    model = model.to(config.device)

    print(f"Model loaded and moved to {config.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    print("Loss function and optimizer initialized")
    print("Starting training...\n")

    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            config.device
        )

        print(
            f"Epoch [{epoch + 1}/{config.num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

    print("\nTraining completed")

    torch.save(model.state_dict(), config.best_model_path)
    print("Model saved successfully.")

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        config.device
    )

    print(
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()