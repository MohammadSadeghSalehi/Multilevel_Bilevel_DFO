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



problem = "Denoising" #Inpainting
regulariser_type = "TV" # "FoE"
test_len = 1
train_len = 300
batch_size = 300
img_size_x = 256
img_size_y = 256
lower_level_alg = "SGD"
lower_level_step = 1e-2


val_dataset = get_dataset("BSDS500", test=True
                          , transform=Compose([Resize((img_size_x, img_size_y)),
                                                Grayscale(num_output_channels=1,),
                                                ]))
val_set = torch.utils.data.Subset(val_dataset, range(96, 97))

# create dataloader
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
   
# load trained model
# -- Load the logs
logs_files = glob.glob("logs/*.pt")
all_logs = []
for file in logs_files:
    logs = torch.load(file, map_location="cpu")
    logs["file"] = file
    all_logs.append(logs)
for log in all_logs:
    print(f"Evaluating log file: {log['file']}")
    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "FoE":
        regularizer = FieldOfExperts(
            num_experts=4,
            device=device
        ).to(device)
    
    regularizer.load_state_dict(log['saved_params'])
    # compute the initial PSNR and loss of noise
    y = physics(gt.to(device))
    x_hat = physics.A_dagger(y)
    mse = torch.mean(torch.norm(x_hat - gt.to(device), p=2, dim=(1, 2, 3)) ** 2)
    psnr_value = psnr(gt.to(device), x_hat)
  
 
      
    for gt in val_dataloader:
        gt = gt.to(device)
        y = physics(gt)
        if lower_level_alg == "SGD" or lower_level_alg == "GD":
            x_hat_out, _ = reconstruct_gd(
                y=y,
                physics=physics,
                data_fidelity=data_fidelity,
                regularizer=regularizer,
                lamda=lmbd,
                step_size=lower_level_step,
                max_iter=1000,
                tol=1e-6,
                detach_grads=True,
                verbose=False,
                return_stats=False,
                BB=False,
                LS=True,
            )
        x_hat_out = torch.clamp(x_hat_out, 0, 1)
        psnr_val = psnr(gt, x_hat_out)
        plt.figure(figsize=(6, 6))
        plt.imshow(gt[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title("Ground Truth Image")
        plt.axis('off')
        plt.savefig(f"logs/gt_image_{regulariser_type}_{problem}_{lower_level_alg}.png", bbox_inches='tight', dpi=300)
        plt.close()
        plt.figure(figsize=(6, 6))
        plt.imshow(y[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title("Noisy Image, {:.2f} dB".format(psnr(gt, y).item()))
        plt.axis('off')
        plt.savefig(f"logs/noisy_image_{regulariser_type}_{problem}_{lower_level_alg}.png", bbox_inches='tight', dpi=300)
        plt.close()
        plt.figure(figsize=(6, 6))
        plt.imshow(x_hat_out[0,0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title(f"{psnr_val.mean().item():.2f} dB")
        plt.axis('off')
        psnr_plot = psnr(gt[0], x_hat_out[0]).item()
        print(f"Saving reconstructed image for {regulariser_type} {problem} at level {log['L']}")
        os.makedirs("logs", exist_ok=True)

        plt.savefig(f"logs/reconstructed_image_{regulariser_type}_{problem}_{lower_level_alg}_{log['ML']}_{log['epsilon']}.png", bbox_inches='tight', dpi=300)
        plt.close()
        

