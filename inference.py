#%%
import torch
from deepinv.physics import Denoising, GaussianNoise, Inpainting, Tomography
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
from FISTA import reconstruct_fista
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
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.manual_seed(0)
np.random.seed(0)


def reconstruct_from_logs(y, gt, physics, data_fidelity, regularizer, lmbd,
                           lower_level_alg, lower_level_step, max_iter=1000):
    """Run reconstruction using the specified lower-level algorithm."""
    if lower_level_alg in ("GD", "SGD"):
        x_hat, _ = reconstruct_gd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=1e-6, detach_grads=True, verbose=False,
            return_stats=False, BB=False, LS=True,
        )
    elif lower_level_alg == "FISTA":
        x_hat, _ = reconstruct_fista(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=1e-6, detach_grads=True, verbose=False,
            return_stats=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {lower_level_alg}")
    return x_hat


def run_inference(problem="Denoising", regulariser_type="TV", lower_level_alg="SGD",
                   lower_level_step=1e-2, img_size_x=256, img_size_y=256):
    """Run inference on saved model checkpoints."""

    # Setup physics
    if problem == "Denoising":
        noise_level = 25 / 255
        physics = Denoising(noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        physics.A_dagger = physics.A_adjoint
    elif problem == "Inpainting":
        noise_level = 5 / 255
        physics = Inpainting(mask=0.7, tensor_size=(1, img_size_x, img_size_y),
                             noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        physics.A_dagger = physics.A_adjoint
    elif problem == "CT":
        from problems.phantoms import PhantomDataset
        noise_level = 0.01
        angles = torch.linspace(0, 180, steps=7, device=device)[:-1]
        physics = Tomography(angles=angles, img_width=64, circle=False, normalize=True,
                             noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    data_fidelity = L2(sigma=1.0)
    lmbd = 1.0
    psnr = PSNR()

    # Setup validation data
    if problem == "CT":
        from problems.phantoms import PhantomDataset
        val_dataset = PhantomDataset(n=64, length=10, phantom_type="triangle")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    else:
        val_dataset = get_dataset("BSDS500", test=True,
                                  transform=Compose([Resize((img_size_x, img_size_y)),
                                                      Grayscale(num_output_channels=1)]))
        val_set = torch.utils.data.Subset(val_dataset, range(96, 97))
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    for gt in val_dataloader:
        y = physics(gt.to(device))
        print("PSNR before reconstruction: ", psnr(gt.to(device), y))

    # Load trained models
    logs_files = glob.glob("logs/*.pt")
    all_logs = []
    for file in logs_files:
        logs = torch.load(file, map_location="cpu")
        logs["file"] = file
        # Filter by problem type
        if logs.get("problem") == problem:
            all_logs.append(logs)

    for log in all_logs:
        print(f"Evaluating log file: {log['file']}")
        if regulariser_type == "TV":
            regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True,
                                     learn_smoothing=True, device=device).to(device)
        elif regulariser_type == "FoE":
            regularizer = FieldOfExperts(num_experts=4, device=device).to(device)
        elif regulariser_type == "CT_TV":
            from problems.ct_radon import CTSmoothedTV
            regularizer = CTSmoothedTV(img_size=64, device=device).to(device)
        else:
            raise ValueError(f"Unknown regulariser: {regulariser_type}")

        if log.get('saved_params') is not None:
            regularizer.load_state_dict(log['saved_params'])

        for gt in val_dataloader:
            gt = gt.to(device)
            y = physics(gt)
            x_hat_out = reconstruct_from_logs(
                y, gt, physics, data_fidelity, regularizer, lmbd,
                lower_level_alg, lower_level_step
            )
            x_hat_out = torch.clamp(x_hat_out, 0, 1)
            psnr_val = psnr(gt, x_hat_out)

            # Save images
            plt.figure(figsize=(6, 6))
            plt.imshow(gt[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.title("Ground Truth Image")
            plt.axis('off')
            plt.savefig(f"logs/gt_image_{regulariser_type}_{problem}_{lower_level_alg}.png",
                        bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(y[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.title("Noisy Image, {:.2f} dB".format(psnr(gt, y).item()))
            plt.axis('off')
            plt.savefig(f"logs/noisy_image_{regulariser_type}_{problem}_{lower_level_alg}.png",
                        bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.imshow(x_hat_out[0, 0].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            plt.title(f"{psnr_val.mean().item():.2f} dB")
            plt.axis('off')
            print(f"Saving reconstructed image for {regulariser_type} {problem} at level {log.get('L')}")
            os.makedirs("logs", exist_ok=True)
            plt.savefig(f"logs/reconstructed_image_{regulariser_type}_{problem}_{lower_level_alg}_{log['ML']}_{log['epsilon']}.png",
                        bbox_inches='tight', dpi=300)
            plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference for Multilevel Bilevel DFO")
    parser.add_argument("--problem", type=str, default="Denoising",
                        choices=["Denoising", "Inpainting", "CT"])
    parser.add_argument("--regulariser", type=str, default="TV",
                        choices=["TV", "FoE", "CT_TV"])
    parser.add_argument("--lower_alg", type=str, default="SGD",
                        choices=["GD", "SGD", "FISTA"])
    parser.add_argument("--lower_step", type=float, default=1e-2)
    args = parser.parse_args()

    run_inference(
        problem=args.problem,
        regulariser_type=args.regulariser,
        lower_level_alg=args.lower_alg,
        lower_level_step=args.lower_step,
    )
