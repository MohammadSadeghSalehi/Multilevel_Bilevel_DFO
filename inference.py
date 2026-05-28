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
from deepinv.loss.metric import PSNR, SSIM
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
                           lower_level_alg, lower_level_step, max_iter=1000,
                           fista_mode="legacy"):
    """Run reconstruction using the specified lower-level algorithm."""
    if lower_level_alg == "GD":
        x_hat, _ = reconstruct_gd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=1e-6, detach_grads=True, verbose=False,
            return_stats=False, BB=False, LS=True,
        )
    elif lower_level_alg == "SGD":
        patch_size = (gt.shape[2] // 2, gt.shape[3] // 2)
        x_hat, _ = reconstruct_sgd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=1e-6, patch_size=patch_size,
            detach_grads=True, verbose=False, return_stats=False,
        )
    elif lower_level_alg == "FISTA":
        x_hat, _ = reconstruct_fista(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=1e-6, detach_grads=True, verbose=False,
            return_stats=False, mode=fista_mode,
        )
    else:
        raise ValueError(f"Unknown algorithm: {lower_level_alg}")
    return x_hat


def build_regularizer(regulariser_type, img_size):
    if regulariser_type == "TV":
        return SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True,
                          learn_smoothing=True, device=device).to(device)
    if regulariser_type == "FoE":
        return FieldOfExperts(num_experts=4, device=device).to(device)
    if regulariser_type == "CT_TV":
        from problems.ct_radon import CTSmoothedTV
        return CTSmoothedTV(img_size=img_size, device=device).to(device)
    if regulariser_type == "CT_FoE":
        from problems.ct_radon import CTFieldOfExperts
        return CTFieldOfExperts(img_size=img_size, num_experts=4, kernel_size=5, device=device).to(device)
    raise ValueError(f"Unknown regulariser: {regulariser_type}")


def setup_problem(problem, img_size, num_angles, noise_level, subset_fraction):
    if problem == "Denoising":
        physics = Denoising(noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        physics.A_dagger = physics.A_adjoint
        val_dataset = get_dataset("BSDS500", test=True,
                                  transform=Compose([Resize((img_size, img_size)),
                                                      Grayscale(num_output_channels=1)]))
        val_set = torch.utils.data.Subset(val_dataset, range(96, 97))
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)
    elif problem == "Inpainting":
        physics = Inpainting(mask=0.7, tensor_size=(1, img_size, img_size),
                             noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        physics.A_dagger = physics.A_adjoint
        val_dataset = get_dataset("BSDS500", test=True,
                                  transform=Compose([Resize((img_size, img_size)),
                                                      Grayscale(num_output_channels=1)]))
        val_set = torch.utils.data.Subset(val_dataset, range(96, 97))
        val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)
    elif problem == "CT":
        from problems.phantoms import PhantomDataset
        angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1]
        physics = Tomography(angles=angles, img_width=img_size, circle=False, normalize=True,
                             noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        val_dataset = PhantomDataset(n=img_size, length=10, phantom_type="triangle")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    elif problem == "CT_LoDo":
        from dataset.datasets import LoDoPaBCustom
        angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1]
        physics = Tomography(angles=angles, img_width=img_size, circle=False, normalize=True,
                             noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
        val_dataset = LoDoPaBCustom(
            root="dataset/LoDoPaB-CT",
            split="val",
            transform=Compose([Resize((img_size, img_size))]),
            subset_fraction=subset_fraction,
        )
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=True)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    return physics, val_dataloader


def stable_lower_step(physics, regularizer, val_dataloader, requested_step):
    """Mirror the training step-size guard for stable post-training inference."""
    try:
        sample_gt = next(iter(val_dataloader)).to(device)
        dummy_x = torch.ones((1, 1, sample_gt.shape[2], sample_gt.shape[3]), device=device)
        p_norm = physics.compute_norm(dummy_x, tol=1e-3, max_iter=20).item()
        l_data = p_norm ** 2
        l_reg = 0.0

        if hasattr(regularizer, "alpha") and hasattr(regularizer, "nu"):
            l_reg = 8.0 * regularizer.alpha.item() / regularizer.nu.item()
            l_reg *= 5.0
        elif hasattr(regularizer, "tv_weight") and hasattr(regularizer, "smoothing"):
            l_reg = 8.0 * regularizer.tv_weight.item() / regularizer.smoothing.item()

        max_step = 0.99 / max(l_data + l_reg, 1e-12)
        return min(requested_step, max_step)
    except Exception as exc:
        print(f"Could not compute stable lower step automatically: {exc}")
        return requested_step


def run_inference(problem="Denoising", regulariser_type="TV", lower_level_alg="SGD",
                   lower_level_step=1e-1, img_size=96, num_angles=16, noise_level=0.03,
                   subset_fraction=1.0, max_iter=None, max_images=1, image_idx=0,
                   log_file=None, fista_mode=None):
    """Run inference on saved model checkpoints."""
    if problem in ("Denoising", "Inpainting") and noise_level == 0.03:
        noise_level = 25 / 255 if problem == "Denoising" else 5 / 255
    if problem == "CT" and img_size == 96:
        img_size = 64
        num_angles = 6
        noise_level = 0.01

    physics, val_dataloader = setup_problem(problem, img_size, num_angles, noise_level, subset_fraction)
    data_fidelity = L2(sigma=1.0)
    lmbd = 1.0
    psnr = PSNR()
    ssim = SSIM()
    if max_iter is None:
        max_iter = 2000 if "CT" in problem else 1000

    for gt in val_dataloader:
        gt = gt.to(device)
        y = physics(gt.to(device))
        x_init = physics.A_dagger(y)
        print("PSNR before reconstruction: ", psnr(gt, x_init).mean().item())
        print("SSIM before reconstruction: ", ssim(gt, x_init).mean().item())
        break

    # Load trained models. If a log file is supplied, evaluate only that run.
    logs_files = [log_file] if log_file else glob.glob("logs/*.pt")
    all_logs = []
    for file in logs_files:
        logs = torch.load(file, map_location="cpu", weights_only=False)
        logs["file"] = file
        # Filter by problem type
        if (logs.get("problem") == problem and logs.get("reg") == regulariser_type
                and logs.get("algo") == lower_level_alg):
            all_logs.append(logs)

    if not all_logs:
        print(f"No logs found for problem={problem}, regulariser={regulariser_type}, lower_alg={lower_level_alg}.")
        return

    start_image_idx = max(0, int(image_idx))
    max_images = max(1, int(max_images))

    for log in all_logs:
        print(f"Evaluating log file: {log['file']}")
        regularizer = build_regularizer(regulariser_type, img_size)
        run_fista_mode = fista_mode or log.get("fista_mode", "legacy")

        if log.get('saved_params') is not None:
            load_result = regularizer.load_state_dict(log['saved_params'], strict=False)
            if load_result.missing_keys or load_result.unexpected_keys:
                print(f"Loaded with missing keys={load_result.missing_keys}, "
                      f"unexpected keys={load_result.unexpected_keys}")
        else:
            print("Warning: this log file does not contain saved_params; using the regularizer initialization.")

        stable_step = stable_lower_step(physics, regularizer, val_dataloader, lower_level_step)
        if stable_step < lower_level_step:
            print(f"Adjusted lower_step from {lower_level_step:.2e} to {stable_step:.2e} for inference stability.")

        saved_count = 0
        for current_image_idx, gt in enumerate(val_dataloader):
            if current_image_idx < start_image_idx:
                continue
            if saved_count >= max_images:
                break
            gt = gt.to(device)
            y = physics(gt)
            x_init = physics.A_dagger(y)
            x_hat_out = reconstruct_from_logs(
                y, gt, physics, data_fidelity, regularizer, lmbd,
                lower_level_alg, stable_step, max_iter=max_iter,
                fista_mode=run_fista_mode,
            )
            x_hat_out = torch.clamp(x_hat_out, 0, 1)
            psnr_val = psnr(gt, x_hat_out)
            ssim_val = ssim(gt, x_hat_out)
            psnr_init = psnr(gt, x_init)

            os.makedirs("logs", exist_ok=True)
            vmax = gt[0, 0].max().item()
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(gt[0, 0].cpu().numpy(), cmap='bone', vmin=0, vmax=vmax)
            axs[0].set_title("Ground Truth")
            axs[1].imshow(x_init[0, 0].cpu().detach().numpy(), cmap='bone', vmin=0, vmax=vmax)
            axs[1].set_title(f"$A^\\dagger D$ ({psnr_init.mean().item():.2f} dB)")
            axs[2].imshow(x_hat_out[0, 0].cpu().detach().numpy(), cmap='bone', vmin=0, vmax=vmax)
            axs[2].set_title(f"Reconstruction ({psnr_val.mean().item():.2f} dB, SSIM {ssim_val.mean().item():.3f})")
            for ax in axs:
                ax.axis('off')
            fig.suptitle(
                f"{problem} | {regulariser_type} | {'MLMC' if log['ML'] else 'SLMC'} | "
                f"eps={log['epsilon']} | FISTA={run_fista_mode}"
            )
            plt.tight_layout()
            print(f"Saving reconstructed image for {regulariser_type} {problem} at level {log.get('L')}")
            plt.savefig(f"logs/inference_{regulariser_type}_{problem}_{lower_level_alg}_{log['ML']}_{log['epsilon']}_{current_image_idx}.png",
                        bbox_inches='tight', dpi=300)
            plt.close()
            saved_count += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference for Multilevel Bilevel DFO")
    parser.add_argument("--problem", type=str, default="Denoising",
                        choices=["Denoising", "Inpainting", "CT", "CT_LoDo"])
    parser.add_argument("--regulariser", type=str, default="TV",
                        choices=["TV", "FoE", "CT_TV", "CT_FoE"])
    parser.add_argument("--lower_alg", type=str, default="SGD",
                        choices=["GD", "SGD", "FISTA"])
    parser.add_argument("--lower_step", type=float, default=1e-1)
    parser.add_argument("--img_size", type=int, default=96)
    parser.add_argument("--num_angles", type=int, default=16)
    parser.add_argument("--noise_level", type=float, default=0.03)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=1)
    parser.add_argument("--image_idx", type=int, default=0,
                        help="Validation image index to start from.")
    parser.add_argument("--log_file", type=str, default="",
                        help="Optional exact training log .pt file to load.")
    parser.add_argument("--fista_mode", type=str, default="",
                        choices=["", "legacy", "restart", "monotone"],
                        help="Override FISTA mode for inference. Empty uses the mode saved in the log.")
    args = parser.parse_args()

    run_inference(
        problem=args.problem,
        regulariser_type=args.regulariser,
        lower_level_alg=args.lower_alg,
        lower_level_step=args.lower_step,
        img_size=args.img_size,
        num_angles=args.num_angles,
        noise_level=args.noise_level,
        subset_fraction=args.subset,
        max_iter=args.max_iter or None,
        max_images=args.max_images,
        image_idx=args.image_idx,
        log_file=args.log_file or None,
        fista_mode=args.fista_mode or None,
    )
