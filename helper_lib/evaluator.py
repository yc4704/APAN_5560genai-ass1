import torch


def evaluate_model(model, test_loader, device):
    model = model.to(device)
    model.eval()

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted_classes = outputs.argmax(dim=1)

            correct_predictions += (
                predicted_classes == labels
            ).sum().item()
            total_predictions += labels.size(0)

    accuracy = (
        100 * correct_predictions / total_predictions
    )

    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy