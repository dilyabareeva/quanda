import torchvision.transforms as transforms  # type: ignore
from PIL import Image


def add_white_square_mnist(img):
    square_size = (4, 4)
    white_square = Image.new("L", square_size, 255)
    img.paste(white_square, (20, 20))
    return img


sample_transforms = {
    "mnist_transforms": transforms.Compose(
        [transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
    "fashion_mnist_transforms": transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
    "mnist_denormalize": transforms.Compose(
        [transforms.Normalize(mean=[0], std=[1 / 0.5])] + [transforms.Normalize(mean=[0.5], std=[1])]
    ),
    "add_white_square_mnist": add_white_square_mnist,
}
