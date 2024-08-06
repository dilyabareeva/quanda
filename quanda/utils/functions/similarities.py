import torch
from torch import Tensor


def cosine_similarity(test, train, replace_nan=0) -> Tensor:
    """
    Compute cosine similarity between test and train activations.

    :param test:
    :param train:
    :param replace_nan:
    :return:
    """
    # TODO: I don't know why Captum return test activations as a list
    if isinstance(test, list):
        test = torch.cat(test)
    test = test.view(test.shape[0], -1)
    train = train.view(train.shape[0], -1)

    test_norm = torch.linalg.norm(test, ord=2, dim=1, keepdim=True)
    train_norm = torch.linalg.norm(train, ord=2, dim=1, keepdim=True)

    test = torch.where(test_norm != 0.0, test / test_norm, Tensor([replace_nan]))
    train = torch.where(train_norm != 0.0, train / train_norm, Tensor([replace_nan])).T

    similarity = torch.mm(test, train)
    return similarity


def dot_product_similarity(test, train, replace_nan=0) -> Tensor:
    """
    Compute cosine similarity between test and train activations.

    :param test:
    :param train:
    :param replace_nan:
    :return:
    """
    # TODO: I don't know why Captum return test activations as a list
    if isinstance(test, list):
        test = torch.cat(test)
    test = test.view(test.shape[0], -1)
    train = train.view(train.shape[0], -1)

    similarity = torch.mm(test, train.T)
    return similarity
