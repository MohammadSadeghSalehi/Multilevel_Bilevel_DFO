from .datasets import BSDS500Dataset, BSD68, LoDoPaB, LoDoPaBCustom
from torchvision.transforms.v2 import ToImage, Compose, Grayscale, ToDtype
import torch
import os


def get_dataset(key, test=False, transform=None, root=None):
    if root is not None:
        location = os.path.join(root, key)
    else:
        location = key
    if key == "BSDS500":
        location = location[:-5]
        if transform is None:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale()]
            )
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale(), transform]
            )
        return BSDS500Dataset(
            root=location, download=True, test=test, transform=transforms
        )
    elif key == "BSDS500_color":
        location = "BS" if root is None else os.path.join(root, "BS")
        if transform is None:
            transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        else:
            transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True), transform])
        return BSDS500Dataset(
            root=location, download=True, test=test, transform=transforms
        )
    elif key == "BSD68":
        if transform is None:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale()]
            )
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), Grayscale(), transform]
            )
        return BSD68(transform=transforms)
    if key == "LoDoPaB":
        if transform is None:
            transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
        else:
            transforms = Compose(
                [ToImage(), ToDtype(torch.float32, scale=True), transform]
            )
        return LoDoPaB(root=location, download=True, test=test, transform=transforms)
    elif key == "LoDoPaBCustom":
        return LoDoPaBCustom(root=location, split="test" if test else "train", transform=transform)
    else:
        raise NameError("Unknown dataset!")
