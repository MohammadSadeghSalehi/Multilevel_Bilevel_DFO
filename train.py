#%%
import torch
from deepinv.physics import Denoising, GaussianNoise, Inpainting, Blur
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose, Resize
import numpy as np
import sys, os
import torch.nn as nn
from torchvision.transforms import Grayscale
import matplotlib.pyplot as plt
from deepinv.loss.metric import PSNR
from torchvision.transforms import Compose, Grayscale
from Regulariser.TV import *
from Regulariser.FoE import *
from GD import reconstruct_gd
from DFO import gaussian_smoothing_step

from torchvision.transforms import (
    RandomCrop,
    RandomVerticalFlip,
    Compose,
    RandomHorizontalFlip,
    CenterCrop,
    RandomRotation,
)

if torch.backends.mps.is_available():
    # mps backend is used in Apple Silicon chips
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

problem = "Denoising" #Inpainting
regulariser_type = "TV" # "FoE"
test_len = 16
train_len = 128
img_size_x = 96
img_size_y = 96

logs_dict = {
    "loss": [],
    "psnr": [],
    "cost": [],
    "sigma": None,
    "N": None,
    "lr": None,
    "algo": "Adam",
    "warm_start": True,
    "reg": regulariser_type,
    "problem": problem,
}

train_dataset = get_dataset("BSDS500", test=False
                          , transform=Compose([CenterCrop((img_size_x, img_size_y)),
                                                Grayscale(num_output_channels=1,),
                                                ]))
val_dataset = get_dataset("BSDS500", test=True
                          , transform=Compose([CenterCrop((img_size_x, img_size_y)),
                                                Grayscale(num_output_channels=1,),
                                                ]))
# splitting in training and validation set
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(test_len))

print(f"test size: {test_len}, train size: {train_len}")
# create dataloaders
val_dataloader = torch.utils.data.DataLoader(
    val_set, batch_size=test_len, shuffle=False, drop_last=True
)

# problem dependent parameters
physics = None
if problem == "Denoising":
    noise_level = 25/255  # noise level for the Gaussian noise model
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
elif problem == "Inpainting":
    noise_level = 5/255  # noise level for the Gaussian noise model
    physics = Inpainting(mask=0.5, tensor_size= (1, img_size_x, img_size_y), noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
data_fidelity = L2(sigma=1.0)
lmbd = 1.0

psnr = PSNR()
if regulariser_type == "TV":
    regularizer = SmoothedTV(tv_weight=0.001, smoothing=1e-1, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
elif regulariser_type == "FoE":
    regularizer = FieldOfExperts(
        num_experts=4,
        device=device
    ).to(device)
for gt in val_dataloader:
    y = physics(gt.to(device))
    print("PSNR before reconstruction: ", psnr(gt.to(device), y))
   
# Bilevel Monte Carlo


num_epochs = 100
total_iterations = 0
for epoch in range(num_epochs):
    for gt in val_dataloader:
        gt = gt.to(device)
        loss, psnr_values, iterations = gaussian_smoothing_step(model=regularizer, physics=physics, data_fidelity=data_fidelity, gt=gt, lmbd=lmbd, device=  device, optim_algo="SGD", warm_start=True, sigma=1e-1, N=20, lr=1e-1, verbose=True)
        logs_dict["loss"].append(loss.item())
        logs_dict["psnr"].append(psnr_values.item())
        total_iterations += iterations
        logs_dict["cost"].append(total_iterations)
# %%
