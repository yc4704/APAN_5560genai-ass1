import torch


def train_model(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            predicted_classes = outputs.argmax(dim=1)
            correct_predictions += (
                predicted_classes == labels
            ).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / total_predictions
        epoch_accuracy = (
            100 * correct_predictions / total_predictions
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {epoch_loss:.4f} "
            f"- Accuracy: {epoch_accuracy:.2f}%"
        )

    return model