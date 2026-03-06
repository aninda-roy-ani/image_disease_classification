from src.config import Config
from src.dataset import get_dataloaders
from src.model import build_model


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

    model = build_model(config)
    model = model.to(config.device)

    print("Model loaded and moved to device:", config.device)


if __name__ == "__main__":
    main()