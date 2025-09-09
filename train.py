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
from SGD import reconstruct_sgd
from DFO import MultilevelSmoothingOptimizer
import glob

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

# set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Calculating L and N_l given lower tolerance epsilon
def calculate_levels(epsilon, alg = "SGD", rho = 1.0):
    if alg == "SGD":
        L = int(2 * epsilon**(-1))
        N_l = [int(4 * epsilon**(-1) * (l + 1)**(-1)) for l in range(L)]
    elif alg == "GD":
        L = int(np.log(3 * epsilon**(-1)) / np.log(rho**(-1)))
        N_l = [int(epsilon**(-1)*(l+1)**2 * rho**(l+1) * np.pi**2/2) for l in range(L)]
    return L, N_l

problem = "Denoising" #Inpainting
regulariser_type = "TV" # "FoE"
test_len = 1
train_len = 300
batch_size = 300
img_size_x = 64
img_size_y = 64
lower_level_alg = "SGD"
lower_level_step = 1e-1

logs_dict = {
    "loss": [],
    "psnr": [],
    "val_psnr": [],
    "cost": [],
    "sigma": None,
    "N": None,
    "lr": None,
    "algo": "SGD",
    "warm_start": True,
    "reg": regulariser_type,
    "problem": problem,
    "ML" : True,
    "saved_params": None,
    "epsilon": None,
}

train_dataset = get_dataset("BSDS500", test=False
                          , transform=Compose([RandomCrop((img_size_x, img_size_y)),
                                                Grayscale(num_output_channels=1,),
                                                ]))
val_dataset = get_dataset("BSDS500", test=True
                          , transform=Compose([CenterCrop((320, 320)),
                                                Grayscale(num_output_channels=1,),
                                                ]))
# splitting in training and validation set
train_set = torch.utils.data.Subset(train_dataset, range(train_len))
val_set = torch.utils.data.Subset(val_dataset, range(96, 97))

print(f"test size: {test_len}, train size: {train_len}")
# create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=False
)
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
    physics = Inpainting(mask=0.7, tensor_size= (1, img_size_x, img_size_y), noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
data_fidelity = L2(sigma=1.0)
lmbd = 1.0

psnr = PSNR()
for gt in val_dataloader:
    y = physics(gt.to(device))
    print("PSNR before reconstruction: ", psnr(gt.to(device), y))
   
# Bilevel Monte Carlo
epsilon = 1e-1
upper_lr = 1e-2
L , N_l = calculate_levels(epsilon=epsilon, alg=lower_level_alg)
max_lower_iter = [20 for _ in range(len(N_l))]
N_per_level_upper = [1 for _ in range(len(N_l))]
print(f"Number of levels: {L}, Number of samples per level: {N_l}")
num_epochs = 100
settings = [
    {"level": max_lower_iter, "N_per_level_upper": N_per_level_upper, "N_per_level_lower": N_l, "ML": True},
    {"level": max_lower_iter, "N_per_level_upper": N_per_level_upper, "N_per_level_lower": [max(N_l) for _ in range(len(N_l))], "ML": False},
]
for sets in settings:
    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "FoE":
        regularizer = FieldOfExperts(
            num_experts=4,
            device=device
        ).to(device)
    opt = MultilevelSmoothingOptimizer(
        model=regularizer,
        physics=physics,
        data_fidelity=data_fidelity,
        ground_truth=gt,
        lmbd=lmbd,
        lr = upper_lr,
        lower_tol=epsilon,
        optimizer_type="SGD",  # or "SGD"
        device=device,
        lower_alg=lower_level_alg,
        lower_lr=lower_level_step
    )
    # compute the initial PSNR and loss of noise
    y = physics(gt.to(device))
    x_hat = physics.A_dagger(y)
    mse = torch.mean(torch.norm(x_hat - gt.to(device), p=2, dim=(1, 2, 3)) ** 2)
    psnr_value = psnr(gt.to(device), x_hat)
    psnr_val = 0 
    for val_gt in val_dataloader:
        gt_val = val_gt.to(device)
        y_val = physics(gt_val)
        x_hat_val = physics.A_dagger(y_val)
        psnr_val += psnr(gt_val, x_hat_val).mean().item()
    logs_dict = {
    "loss": [mse.item()],
    "psnr": [psnr_value.mean().item()],
    "val_psnr": [psnr_val / len(val_dataloader)],
    "cost": [0],
    "sigma": None,
    "N": None,
    "lr": upper_lr,
    "algo": "SGD",
    "upper_algo": "SGD",
    "warm_start": True,
    "reg": regulariser_type,
    "problem": problem,
    "ML" : sets["ML"],
    "saved_params": None,
    }
    logs_dict["epsilon"] = epsilon
    if logs_dict['ML']:
        logs_dict["L"] = L
    else:
        logs_dict["L"] = None
    total_iterations = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        for gt in train_dataloader:
            gt = gt.to(device)
            # Multilevel step (εₗ ~ iter level)
            levels = sets["level"]
            N_per_level_upper = sets["N_per_level_upper"]
            N_per_level_lower = sets["N_per_level_lower"]
            opt.gt = gt
            loss, psnr_values, iterations  = opt.multilevel_gradient_step(levels=levels, N_per_level_upper=N_per_level_upper, N_per_level_lower=N_per_level_lower, warm_start=False, normalize_grad=False, verbose=True)
            logs_dict["loss"].append(loss.item())
            logs_dict["psnr"].append(psnr_values.item())
            total_iterations += iterations
            logs_dict["cost"].append(total_iterations)
            epoch_loss += loss.item()
        for gt in val_dataloader:
            gt = gt.to(device)
            y = physics(gt)
            if lower_level_alg == "SGD":
                x_hat_out, _ = reconstruct_gd(
                    y=y,
                    physics=physics,
                    data_fidelity=data_fidelity,
                    regularizer=opt.model,
                    lamda=lmbd,
                    step_size=lower_level_step,
                    max_iter=200,
                    tol=1e-6,
                    detach_grads=True,
                    verbose=False,
                    return_stats=False,
                    BB=False,
                    LS=True,
                )
            elif lower_level_alg == "GD":
                x_hat_out, _ = reconstruct_sgd(
                    y=y,
                    physics=physics,
                    data_fidelity=data_fidelity,
                    regularizer=opt.model,
                    lamda=lmbd,
                    step_size=lower_level_step,
                    max_iter=500,
                    tol=1e-6,
                    detach_grads=True,
                    verbose=False,
                    return_stats=False,
                )
            psnr_val = psnr(gt, x_hat_out)
            average_loss = epoch_loss / len(train_dataloader)
            logs_dict["val_psnr"].append(psnr_val.mean().item())
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Val PSNR: {psnr_val.mean().item()}")
            if epoch % 1 == 0:
                plt.figure(figsize=(18, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(gt[0,0].cpu().numpy(), cmap='gray')
                plt.title("Ground Truth Image")
                plt.axis('off')
                plt.subplot(1, 3, 2)
                plt.imshow(y[0,0].cpu().numpy(), cmap='gray')
                plt.title("Noisy Image")
                plt.axis('off')
                plt.subplot(1, 3, 3)
                plt.imshow(x_hat_out[0,0].cpu().detach().numpy(), cmap='gray')
                plt.title("Reconstructed Image")
                plt.axis('off')
                psnr_plot = psnr(gt[0], x_hat_out[0]).item()
                plt.suptitle(f"Reconstructed Image - PSNR: {psnr_val.mean().item():.2f} dB")
                if epoch == num_epochs - 1:
                    print(f"Saving reconstructed image for {regulariser_type} {problem} at level {sets['level']}")
                    os.makedirs("logs", exist_ok=True)
                    
                    plt.savefig(f"logs/reconstructed_image_{regulariser_type}_{problem}_{lower_level_alg}_{sets['ML']}_{epsilon}.png", bbox_inches='tight', dpi=300)
                    plt.close()
                plt.close()
            logs_dict["saved_params"] = opt.model.state_dict()
        # save the model and logs
        torch.save(logs_dict, f"logs/logs_{regulariser_type}_{problem}_{lower_level_alg}_{sets['ML']}_{epsilon}.pt")


# %%
