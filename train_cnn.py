from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from helper_lib.data_loader import get_data_loaders
from helper_lib.evaluator import evaluate_model
from helper_lib.model import AssignmentCNN
from helper_lib.trainer import train_model


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    train_loader, test_loader = get_data_loaders(
        data_dir="data",
        batch_size=64,
    )

    model = AssignmentCNN()

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5,
    )

    evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        device=device,
    )

    model_directory = Path("models")
    model_directory.mkdir(exist_ok=True)

    model_path = model_directory / "cifar10_cnn.pth"

    torch.save(
        trained_model.state_dict(),
        model_path,
    )

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()