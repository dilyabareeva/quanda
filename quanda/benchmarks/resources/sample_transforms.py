"""Torchvision transforms for benchmarks."""

import torchvision.transforms as transforms  # type: ignore
from PIL import Image, ImageDraw


def add_white_square_mnist(img):
    """Add a white square to the top-left corner of the image."""
    square_size = (12, 12)
    white_square = Image.new("L", square_size, 255)
    img.paste(white_square, (0, 0))
    return img


def add_yellow_square(img):
    """Add a yellow square to a fixed location on the image."""
    square_size = (15, 15)
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))
    img.paste(yellow_square, (10, 10))
    return img


def add_pink_frame(img):
    """Draw a pink frame inside the central square preserved by
    Resize(256)+CenterCrop(224).

    Positioning is centered (not edge-inset) so the full rectangle survives
    the crop regardless of source aspect ratio.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size
    m = min(w, h)
    half = 0.40 * m
    width = max(3, int(0.04 * m))
    cx, cy = w / 2, h / 2
    draw.rectangle(
        [cx - half, cy - half, cx + half, cy + half],
        outline=(255, 105, 180),
        width=width,
    )
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
    "add_pink_frame": add_pink_frame,
    "cifar10_transforms": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    ),
    "cifar10_adversarial_transforms": transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ]
    ),
    "awa2_train_transforms": transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    ),
    "awa2_transforms": transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    ),
    "awa2_adversarial_transforms": transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    ),
}
