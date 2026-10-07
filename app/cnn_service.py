from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.model import AssignmentCNN


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIRECTORY / "models" / "cifar10_cnn.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


image_transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)


model = AssignmentCNN()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model = model.to(DEVICE)
model.eval()


def predict_image(image: Image.Image):
    image = image.convert("RGB")
    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

    class_index = predicted_index.item()
    class_name = CLASS_NAMES[class_index]

    return {
        "class_index": class_index,
        "class_name": class_name,
        "confidence": confidence.item(),
    }