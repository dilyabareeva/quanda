import torch


def mark_image_contour_and_square(x):
    mask = torch.zeros_like(x[0])
    mid = int(x.shape[-1] / 2)
    mask[mid - 3 : mid + 3, mid - 3 : mid + 3] = 1.0
    mask[:2, :] = 1.0
    mask[-2:, :] = 1.0
    mask[:, -2:] = 1.0
    mask[:, :2] = 1.0
    x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
    if x.shape[0] > 1:
        x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
    return x.numpy().transpose(1, 2, 0)


def mark_image_middle_square(x):
    mask = torch.zeros_like(x[0])
    mid = int(x.shape[-1] / 2)
    mask[(mid - 4) : (mid + 4), (mid - 4) : (mid + 4)] = 1.0
    x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
    if x.shape[0] > 1:
        x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
    return x.numpy().transpose(1, 2, 0)


def mark_image_contour(x):
    # TODO: make controur, middle square and combined masks a constant somewhere else
    mask = torch.zeros_like(x[0])
    mask[:2, :] = 1.0
    mask[-2:, :] = 1.0
    mask[:, -2:] = 1.0
    mask[:, :2] = 1.0
    x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
    if x.shape[0] > 1:
        x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)

    return x.numpy().transpose(1, 2, 0)
