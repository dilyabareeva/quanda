"""Torchvision transforms for benchmarks."""

import torchvision.transforms as transforms  # type: ignore
from PIL import Image


def add_white_square_mnist(img):
    """Add a white square to the center of the image."""
    square_size = (8, 8)
    white_square = Image.new("L", square_size, 255)
    img.paste(white_square, (15, 15))
    return img


def add_yellow_square(img):
    """Add a yellow square to a fixed location on the image."""
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new(
        "RGB", square_size, (255, 255, 0)
    )  # Create a yellow square
    img.paste(
        yellow_square, (10, 10)
    )  # Paste it onto the image at the specified position
    return img


sample_transforms = {
    "mnist_transforms": transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "adversarial_transforms": transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    ),
    "mnist_denormalize": transforms.Compose(
        [transforms.Normalize(mean=[0], std=[1 / 0.5])]
        + [transforms.Normalize(mean=[0.5], std=[1])]
    ),
    "add_white_square_mnist": add_white_square_mnist,
    "tiny_imagenet_transforms": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "tiny_imagener_adversarial_transforms": transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    ),
    "add_yellow_square": add_yellow_square,
}
