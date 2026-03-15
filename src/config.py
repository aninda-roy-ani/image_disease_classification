from dataclasses import dataclass
import torch


@dataclass
class Config:
    data_dir: str = "data"
    train_dir: str = "data/train"
    val_dir: str = "data/val"
    test_dir: str = "data/test"

    model_name: str = "swin_tiny_patch4_window7_224"
    num_classes: int = 7
    img_size: int = 224

    batch_size: int = 32
    num_workers: int = 4

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    best_model_path: str = "checkpoints/best_model.pth"