import torch
from deepinv.physics import Denoising, GaussianNoise, Inpainting, Tomography
from deepinv.optim import L2
from dataset import get_dataset
from torchvision.transforms import RandomCrop, CenterCrop, Compose, Resize
import numpy as np
import sys, os
import argparse
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

# set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# Level schedule calculations
def calculate_levels(epsilon, alg="SGD", rho=0.3):
    """
    Calculate number of levels L and samples per level N_l
    for the multilevel estimator, given a target accuracy epsilon.

    Matches Table 1 in the paper.
    """
    if alg == "SGD":
        L = max(1, int(np.ceil(2 * epsilon ** (-1))))
        N_l = [int(4 * epsilon ** (-1) * (l + 1) ** (-1)) for l in range(L)]
    elif alg == "GD":
        L = max(1, int(np.ceil(np.log(3 * epsilon ** (-1)) / np.log(rho ** (-1)))))
        N_l = [int(epsilon ** (-1) * (l + 1) ** 2 * rho ** (l + 1) * np.pi ** 2 / 2) for l in range(L)]
    elif alg == "FISTA":
        # From Theorem 2.3:
        # Total levels (L_math + 1) >= sqrt(2) * eps^{-1/2}
        L = max(3, int(np.ceil(np.sqrt(2) * epsilon ** (-0.5))))

        N_l = []
        for l in range(L):
            if l == 0:
                # N_0 >= 3 * eps^{-1}
                N_l.append(max(1, int(np.ceil(3 * epsilon ** (-1)))))
            else:
                # Use fewer samples on progressively finer levels.
                N_l.append(max(1, int(np.ceil(4 * epsilon ** (-1) / (l + 1)))))
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    # Ensure at least 1 sample per level
    N_l = [max(1, n) for n in N_l]
    L = max(1, L)
    return L, N_l


def build_lower_iteration_levels(level_count, finest_iter, alg="GD", min_iter=50):
    """Build the lower-level iteration ladder used by the estimator."""
    finest_iter = max(int(finest_iter), 1)
    level_count = max(int(level_count), 1)
    if level_count == 1:
        return [finest_iter]

    if alg == "FISTA":
        start = min(max(1, min_iter), finest_iter)
        raw_levels = np.geomspace(start, finest_iter, num=level_count)
    else:
        raw_levels = np.linspace(max(1, finest_iter // level_count), finest_iter, num=level_count)

    levels = []
    previous = 0
    for value in raw_levels:
        level = max(previous + 1, int(np.ceil(value)))
        level = min(level, finest_iter)
        levels.append(level)
        previous = level

    levels[-1] = finest_iter
    return levels


def required_training_batch_size(N_l, compare_single_level=True):
    """Smallest data batch that can supply the lower-sample pools used per oracle step."""
    if not N_l:
        return 1
    if compare_single_level:
        return max(1, int(np.sum(N_l)))
    return max(1, int(max(N_l)))


def limited_batches(dataloader, max_batches=0):
    """Yield at most max_batches batches; max_batches<=0 means a full pass."""
    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break
        yield batch_idx, batch


# Experiment setup
def setup_denoising_experiment(regulariser_type, img_size_x=128, img_size_y=128,
                                train_len=300, test_len=1, batch_size=300, val_batch_size=1,
                                color=False, freeze_foe_kernels=False, foe_num_experts=None):
    """Set up the TV denoising experiment (existing experiment)."""
    noise_level = 25 / 255
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
    data_fidelity = L2(sigma=1.0)

    dataset_key = "BSDS500_color" if color else "BSDS500"
    if color:
        train_transform = Compose([RandomCrop((img_size_x, img_size_y))])
        val_transform = Compose([Resize((img_size_x, img_size_y))])
        num_channels = 3
    else:
        train_transform = Compose([RandomCrop((img_size_x, img_size_y)),
                                   Grayscale(num_output_channels=1)])
        val_transform = Compose([Resize((img_size_x, img_size_y)),
                                 Grayscale(num_output_channels=1)])
        num_channels = 1

    train_dataset = get_dataset(dataset_key, test=False, transform=train_transform)
    val_dataset = get_dataset(dataset_key, test=True, transform=val_transform)
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_start = 96 if len(val_dataset) > 96 else 0
    val_stop = min(len(val_dataset), val_start + max(1, int(test_len)))
    val_set = torch.utils.data.Subset(val_dataset, range(val_start, val_stop))

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False)

    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "FoE":
        num_experts = foe_num_experts or (8 if color else 4)
        regularizer = FieldOfExperts(
            num_experts=num_experts,
            kernel_size=5,
            num_channels=num_channels,
            colored=color,
            device=device,
        ).to(device)
        if freeze_foe_kernels:
            for param in regularizer.conv_experts.parameters():
                param.requires_grad_(False)
    else:
        raise ValueError(f"Unknown regulariser: {regulariser_type}")

    return {
        "physics": physics,
        "data_fidelity": data_fidelity,
        "regularizer": regularizer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "lmbd": 1.0,
        "noise_level": noise_level,
        "color": color,
        "num_channels": num_channels,
        "foe_num_experts": foe_num_experts or (8 if color else 4),
        "freeze_foe_kernels": freeze_foe_kernels,
    }


def setup_inpainting_experiment(regulariser_type, img_size_x=128, img_size_y=128,
                                 train_len=300, test_len=1, batch_size=300, val_batch_size=1):
    """Set up the inpainting experiment."""
    noise_level = 5 / 255
    physics = Inpainting(mask=0.7, tensor_size=(1, img_size_x, img_size_y),
                         noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
    data_fidelity = L2(sigma=1.0)

    train_dataset = get_dataset("BSDS500", test=False,
                                transform=Compose([RandomCrop((img_size_x, img_size_y)),
                                                    Grayscale(num_output_channels=1)]))
    val_dataset = get_dataset("BSDS500", test=True,
                              transform=Compose([Resize((128, 128)),
                                                  Grayscale(num_output_channels=1)]))
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_start = 96 if len(val_dataset) > 96 else 0
    val_stop = min(len(val_dataset), val_start + max(1, int(test_len)))
    val_set = torch.utils.data.Subset(val_dataset, range(val_start, val_stop))

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=val_batch_size, shuffle=False, drop_last=False)

    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "FoE":
        regularizer = FieldOfExperts(num_experts=4, device=device).to(device)
    else:
        raise ValueError(f"Unknown regulariser: {regulariser_type}")

    return {
        "physics": physics,
        "data_fidelity": data_fidelity,
        "regularizer": regularizer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "lmbd": 1.0,
        "noise_level": noise_level,
    }


def setup_ct_experiment_full(regulariser_type="TV", img_size=64, num_angles=6,
                              noise_level=0.01, train_len=300, test_len=1, batch_size=50,
                              val_batch_size=1):
    """Set up the CT/Radon transform experiment using deepinv.physics.Tomography."""
    from problems.ct_radon import CTSmoothedTV
    from problems.phantoms import PhantomDataset

    # Set up angles uniformly in [0, 180).
    angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1]

    physics = Tomography(
        angles=angles,
        img_width=img_size,
        circle=False,
        normalize=True,
        noise_model=GaussianNoise(sigma=noise_level),
        device=device,
    ).to(device)

    data_fidelity = L2(sigma=1.0)

    train_dataset = PhantomDataset(n=img_size, length=train_len, phantom_type="triangle")
    val_dataset = PhantomDataset(n=img_size, length=10, phantom_type="triangle")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)

    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "CT_TV":
        regularizer = CTSmoothedTV(img_size=img_size, device=device).to(device)
    elif regulariser_type == "CT_FoE":
        from problems.ct_radon import CTFieldOfExperts
        regularizer = CTFieldOfExperts(img_size=img_size, num_experts=4, kernel_size=5, device=device).to(device)
    elif regulariser_type == "FoE":
        regularizer = FieldOfExperts(num_experts=4, device=device).to(device)
    else:
        raise ValueError(f"Unknown regulariser: {regulariser_type}")

    return {
        "physics": physics,
        "data_fidelity": data_fidelity,
        "regularizer": regularizer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "lmbd": 1.0,
        "noise_level": noise_level,
        "img_size": img_size,
        "num_angles": num_angles,
    }

def setup_lodopab_ct_experiment(regulariser_type="CT_TV", img_size=96, num_angles=16,
                                noise_level=0.03, batch_size=50, val_batch_size=5,
                                subset_fraction=1.0):
    """Set up the CT experiment using the LoDoPaB dataset."""
    from dataset.datasets import LoDoPaBCustom
    from problems.ct_radon import CTSmoothedTV

    # 180 degrees mapping
    angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1]

    # Calculate unnormalized forward model so we can measure the exact operator jump dynamically
    # and then scale the true 0.7 physical constraint organically against whatever angles/resolution you set!
    physics_unnorm = Tomography(
        angles=angles,
        img_width=img_size,
        circle=False,
        normalize=False,
        device=device
    ).to(device)

    # Dynamically read what the norm shifted to (e.g 61 for 128x30, or ~125 for 256x64)
    # Note: Tomography natively computes this via eigenvalues when doing normalized matrices.
    with torch.no_grad():
        dummy = torch.ones(1, 1, img_size, img_size, device=device)
        norm_val = physics_unnorm.compute_norm(dummy, tol=1e-3, max_iter=20).item()

    physics = Tomography(
        angles=angles,
        img_width=img_size,
        circle=False,
        normalize=True,
        noise_model=GaussianNoise(sigma=noise_level),
        device=device,
    ).to(device)

    data_fidelity = L2(sigma=1.0)

    train_transform = Compose([
        Resize((img_size, img_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip()
    ])
    val_transform = Compose([
        Resize((img_size, img_size))
    ])

    root_dir = "dataset/LoDoPaB-CT"
    train_dataset = LoDoPaBCustom(root=root_dir, split="train", transform=train_transform, subset_fraction=subset_fraction)
    val_dataset = LoDoPaBCustom(root=root_dir, split="val", transform=val_transform, subset_fraction=subset_fraction)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_batch_size = min(max(1, val_batch_size), max(1, len(val_dataset)))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, drop_last=False)

    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "CT_TV":
        regularizer = CTSmoothedTV(img_size=img_size, device=device).to(device)
    elif regulariser_type == "CT_FoE":
        from problems.ct_radon import CTFieldOfExperts
        regularizer = CTFieldOfExperts(img_size=img_size, num_experts=4, kernel_size=5, device=device).to(device)

        # Lower baseline initialization to prevent over-smoothing early blocks
        # with torch.no_grad():
        #     regularizer.log10_alpha.data = torch.tensor(-5.0, dtype=torch.float32, device=device)
        #     regularizer.log10_lambda.data = torch.tensor(-4.0, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unknown regulariser: {regulariser_type}")

    return {
        "physics": physics,
        "data_fidelity": data_fidelity,
        "regularizer": regularizer,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "lmbd": 1.0,
        "noise_level": noise_level,
        "img_size": img_size,
        "num_angles": num_angles,
        "subset_fraction": subset_fraction,
    }


# Reconstruction dispatch
def reconstruct(y, physics, data_fidelity, regularizer, lmbd, lower_level_alg,
                lower_level_step, max_iter=500, tol=1e-4,
                verbose=False, return_stats=False, fista_mode="legacy",
                sgd_step_decay="none", sgd_patch_mode="patch",
                sgd_patch_fraction=0.5):
    """Dispatch to the correct lower-level solver."""
    if lower_level_alg == "GD":
        x_hat, iters = reconstruct_gd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, detach_grads=True, verbose=verbose,
            return_stats=return_stats, BB=False, LS=True,
        )
    elif lower_level_alg == "SGD":
        patch_fraction = min(max(float(sgd_patch_fraction), 1e-4), 1.0)
        patch_size = (
            max(1, int(round(y.shape[2] * patch_fraction))),
            max(1, int(round(y.shape[3] * patch_fraction))),
        )
        x_hat, iters = reconstruct_sgd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, patch_size=patch_size,
            detach_grads=True, verbose=verbose, return_stats=return_stats,
            step_decay=sgd_step_decay,
            patch_mode=sgd_patch_mode,
            mask_probability=patch_fraction * patch_fraction,
        )
    elif lower_level_alg == "FISTA":
        x_hat, iters = reconstruct_fista(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, detach_grads=True, verbose=verbose,
            return_stats=return_stats,
            mode=fista_mode,
        )
    else:
        raise ValueError(f"Unknown lower-level algorithm: {lower_level_alg}")
    return x_hat, iters


# Main training loop
def run_experiment(
    problem="Denoising",
    regulariser_type="TV",
    lower_level_alg="SGD",
    lower_level_step=1e-1,
    epsilon=5e-1,
    budget=1e7,
    upper_lr=2e-3,
    num_epochs=200,
    rho=0.3,
    subset_fraction=1.0,
    img_size=None,
    num_angles=None,
    noise_level=None,
    batch_size=None,
    val_batch_size=None,
    val_len=1,
    train_batches=0,
    val_batches=0,
    upper_directions=1,
    fista_diagnostics=False,
    fista_mode="legacy",
    lower_max_iter=None,
    color=False,
    sgd_step_decay="none",
    sgd_patch_mode="patch",
    sgd_patch_fraction=0.5,
    freeze_foe_kernels=False,
    foe_num_experts=None,
    run_tag="",
):
    """
    Run the bilevel optimization experiment.

    Args:
        problem: 'Denoising', 'Inpainting', or 'CT'
        regulariser_type: 'TV', 'FoE', or 'CT_TV'
        lower_level_alg: 'GD', 'SGD', or 'FISTA'
        lower_level_step: Step size for lower-level solver
        epsilon: Target accuracy
        budget: Computational budget
        upper_lr: Upper-level learning rate
        num_epochs: Number of epochs
        rho: Contraction rate (for GD level schedule)
    """
    print(f"{'='*60}")
    print(f"  Experiment: {problem} | Regularizer: {regulariser_type}")
    print(f"  Lower-level: {lower_level_alg} | epsilon: {epsilon}")
    if problem == "Denoising":
        print(f"  Image mode: {'RGB' if color else 'grayscale'}")
        if regulariser_type == "FoE":
            print(f"  FoE experts: {foe_num_experts or (8 if color else 4)}")
    print(f"{'='*60}")

    _, schedule_N_l = calculate_levels(epsilon=epsilon, alg=lower_level_alg, rho=rho)
    auto_batch_size = required_training_batch_size(schedule_N_l, compare_single_level=True)
    effective_batch_size = max(1, int(batch_size)) if batch_size else auto_batch_size
    if effective_batch_size < auto_batch_size:
        print(f"Warning: batch_size={effective_batch_size} is smaller than required sample pool "
              f"{auto_batch_size}; using {auto_batch_size}.")
        effective_batch_size = auto_batch_size

    effective_val_batch_size = max(1, int(val_batch_size)) if val_batch_size else min(5, effective_batch_size)
    print(f"[DataLoader] train batch size: {effective_batch_size} | val batch size: {effective_val_batch_size} | "
          f"train batches/epoch: {'all' if train_batches <= 0 else train_batches} | "
          f"val batches: {'all' if val_batches <= 0 else val_batches}")

    # Setup experiment
    if problem == "Denoising":
        exp = setup_denoising_experiment(
            regulariser_type,
            img_size_x=img_size or 128,
            img_size_y=img_size or 128,
            batch_size=effective_batch_size,
            val_batch_size=effective_val_batch_size,
            test_len=val_len,
            color=color,
            freeze_foe_kernels=freeze_foe_kernels,
            foe_num_experts=foe_num_experts,
        )
    elif problem == "Inpainting":
        exp = setup_inpainting_experiment(
            regulariser_type,
            img_size_x=img_size or 128,
            img_size_y=img_size or 128,
            batch_size=effective_batch_size,
            val_batch_size=effective_val_batch_size,
            test_len=val_len,
        )
    elif problem == "CT":
        exp = setup_ct_experiment_full(
            regulariser_type,
            img_size=img_size or 64,
            num_angles=num_angles or 6,
            noise_level=0.01 if noise_level is None else noise_level,
            batch_size=effective_batch_size,
            val_batch_size=effective_val_batch_size,
        )
    elif problem == "CT_LoDo":
        exp = setup_lodopab_ct_experiment(
            regulariser_type,
            img_size=img_size or 96,
            num_angles=num_angles or 16,
            noise_level=0.03 if noise_level is None else noise_level,
            batch_size=effective_batch_size,
            val_batch_size=effective_val_batch_size,
            subset_fraction=subset_fraction,
        )
    else:
        raise ValueError(f"Unknown problem: {problem}")

    physics = exp["physics"]
    data_fidelity = exp["data_fidelity"]
    lmbd = exp["lmbd"]
    train_dataloader = exp["train_dataloader"]
    val_dataloader = exp["val_dataloader"]
    regularizer = exp["regularizer"]

    psnr = PSNR()
    ssim = SSIM()

    # Compute stable step size bound (1/L) for the forward physics operator + regularizer
    L_total_norm = 1.0 # Fallback in case of exception
    L_data = 1.0
    try:
        sample_gt = next(iter(val_dataloader))
        dummy_x = torch.ones_like(sample_gt[:1]).to(device)
        p_norm = physics.compute_norm(dummy_x, tol=1e-3, max_iter=20).item()

        L_data = p_norm ** 2
        L_reg = 0.0

        # Approximate the Lipschitz constant of the regularizer to prevent FISTA divergence
        if hasattr(regularizer, 'alpha') and hasattr(regularizer, 'nu'):
            # L_reg for SmoothedTV is roughly 8 * alpha / nu
            L_reg = 8.0 * regularizer.alpha.item() / regularizer.nu.item()

            # Since DFO will actively optimize/increase alpha during training, we add a
            # 5x safety factor so the step size remains stable across the hyperparameter space.
            L_reg *= 5.0

        L_total = L_data + L_reg
        L_total_norm = L_total # Save out of try block for MLMC theory calculation

        max_step = 0.99 / L_total

        if lower_level_step > max_step:
            print(f"Warning: Auto-adjusting lower_step from {lower_level_step} to {max_step:.2e} "
                  f"(< 1/L_total) to account for structural curvature and prevent {lower_level_alg} divergence!")
            lower_level_step = max_step
    except Exception as e:
        print(f"Could not compute physics norm automatically: {e}")

    # Print initial quality
    for _, gt in limited_batches(val_dataloader, val_batches):
        y = physics(gt.to(device))
        x_init = physics.A_dagger(y)
        print("PSNR before reconstruction: ", psnr(gt.to(device), x_init).mean().item())
        print("SSIM before reconstruction: ", ssim(gt.to(device), x_init).mean().item())

    # Level schedule
    L, N_l = calculate_levels(epsilon=epsilon, alg=lower_level_alg, rho=rho)
    L_levels = len(N_l)

    # Theory driven step size: s is proportional to sqrt(gamma) i.e. sqrt(L_total_norm)
    # The multiplier acts as the substitution for bounding the initial discrepancy distance
    condition_multiplier = 100 if problem in ["CT", "CT_LoDo"] else 40

    # FISTA theoretically achieves O(1/k^2) convergence compared to GD O(1/k).
    # Thus, it structurally requires far fewer iterations to guarantee epsilon-accuracy bounds!
    if lower_level_alg == "FISTA":
        condition_multiplier = max(10, condition_multiplier)

    s_theory = int(np.sqrt(L_total_norm) * condition_multiplier)

    # Safeguard bounds: Guarantee that the finest MLMC level stays controlled
    # unless the user explicitly overrides it for high-resolution CT runs.
    global_iter_ceiling = 2000 if problem in ["CT", "CT_LoDo"] else 500
    max_allowable_s = max(50, global_iter_ceiling // L_levels)
    if lower_max_iter is not None and lower_max_iter > 0:
        finest_lower_iter = int(lower_max_iter)
        s_step = max(1, int(np.ceil(finest_lower_iter / L_levels)))
    else:
        s_step = max(50, min(s_theory, max_allowable_s))
        finest_lower_iter = s_step * L_levels

    print(f"[MLMC Theory Schedule] Gamma Norm: {L_total_norm:.4f} | Base Step (s): {s_step} | Max Iter: {finest_lower_iter}")

    max_lower_iter = build_lower_iteration_levels(L_levels, finest_lower_iter, alg=lower_level_alg)
    upper_directions = max(1, int(upper_directions))
    N_per_level_upper = [upper_directions for _ in range(len(N_l))]
    print(f"Number of levels: {L}, Number of samples per level: {N_l}")
    print(f"Gaussian directions averaged per level: {upper_directions}")
    print(f"Lower-level iteration ladder: {max_lower_iter}")

    # Settings: ML vs SL comparison
    single_level_batch = max(1, int(np.sum(N_l)))
    settings = [
        {"level": max_lower_iter, "N_per_level_upper": N_per_level_upper, "N_per_level_lower": N_l, "ML": True},
        {"level": [max_lower_iter[-1]], "N_per_level_upper": [upper_directions],
         "N_per_level_lower": [single_level_batch], "ML": False},
    ]

    # Extract the exact image dimension dictated by the dataset setup
    sample_val = next(iter(val_dataloader))
    actual_channels = sample_val.shape[1]
    actual_img_size = sample_val.shape[2]
    run_problem_tag = "ColorDenoising" if problem == "Denoising" and color else problem
    filename_suffix = f"_{run_tag}" if run_tag else ""

    for sets in settings:
        # Re-initialize regularizer for fair comparison
        if regulariser_type == "TV":
            regularizer = SmoothedTV(tv_weight=1e-3, smoothing=1e-3, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
        elif regulariser_type == "FoE":
            num_experts = foe_num_experts or (8 if actual_channels == 3 else 4)
            regularizer = FieldOfExperts(
                num_experts=num_experts,
                kernel_size=5,
                num_channels=actual_channels,
                colored=actual_channels == 3,
                device=device,
            ).to(device)
            if freeze_foe_kernels:
                for param in regularizer.conv_experts.parameters():
                    param.requires_grad_(False)
        elif regulariser_type == "CT_TV":
            from problems.ct_radon import CTSmoothedTV
            regularizer = CTSmoothedTV(img_size=actual_img_size, device=device).to(device)
        elif regulariser_type == "CT_FoE":
            from problems.ct_radon import CTFieldOfExperts
            regularizer = CTFieldOfExperts(img_size=actual_img_size, num_experts=4, kernel_size=5, device=device).to(device)
        else:
            raise ValueError(f"Unknown regulariser: {regulariser_type}")

        opt = MultilevelSmoothingOptimizer(
            model=regularizer,
            physics=physics,
            data_fidelity=data_fidelity,
            ground_truth=gt,
            lmbd=lmbd,
            lr=upper_lr,
            lower_tol=epsilon,
            optimizer_type="SGD",
            device=device,
            lower_alg=lower_level_alg,
            lower_lr=lower_level_step,
            fista_mode=fista_mode,
            sgd_step_decay=sgd_step_decay,
            sgd_patch_mode=sgd_patch_mode,
            sgd_patch_fraction=sgd_patch_fraction,
        )

        # Compute initial PSNR and loss
        y = physics(gt.to(device))
        x_hat = physics.A_dagger(y)
        mse = torch.mean(torch.norm(x_hat - gt.to(device), p=2, dim=(1, 2, 3)) ** 2)
        psnr_value = psnr(gt.to(device), x_hat)
        ssim_value = ssim(gt.to(device), x_hat)
        psnr_val = 0
        ssim_val = 0
        initial_val_batches = 0
        for _, val_gt in limited_batches(val_dataloader, val_batches):
            gt_val = val_gt.to(device)
            y_val = physics(gt_val)
            x_hat_val = physics.A_dagger(y_val)
            psnr_val += psnr(gt_val, x_hat_val).mean().item()
            ssim_val += ssim(gt_val, x_hat_val).mean().item()
            initial_val_batches += 1

        # Pre-capture the explicit initialization parameters so cost=0 maps accurately on the plot
        initial_params = {k: v.detach().clone().cpu().numpy() for k, v in opt.model.state_dict().items()}

        logs_dict = {
            "loss": [mse.item()],
            "psnr": [psnr_value.mean().item()],
            "ssim": [ssim_value.mean().item()],
            "val_psnr": [psnr_val / max(1, initial_val_batches)],
            "val_ssim": [ssim_val / max(1, initial_val_batches)],
            "param_history": [initial_params],
            "cost": [0],
            "sigma": None,
            "N": None,
            "lr": upper_lr,
            "algo": lower_level_alg,
            "upper_algo": "SGD",
            "warm_start": True,
            "reg": regulariser_type,
            "problem": problem,
            "run_problem_tag": run_problem_tag,
            "color": bool(color),
            "num_channels": actual_channels,
            "foe_num_experts": foe_num_experts or (8 if actual_channels == 3 else 4),
            "freeze_foe_kernels": freeze_foe_kernels,
            "run_tag": run_tag,
            "ML": sets["ML"],
            "saved_params": None,
            "epsilon": epsilon,
            "img_size": exp.get("img_size"),
            "num_angles": exp.get("num_angles"),
            "noise_level": exp.get("noise_level"),
            "subset_fraction": exp.get("subset_fraction", subset_fraction),
            "N_l": N_l,
            "max_lower_iter": sets["level"],
            "estimator": "MLMC" if sets["ML"] else "SLMC",
            "batch_size": effective_batch_size,
            "val_batch_size": effective_val_batch_size,
            "val_len": val_len,
            "train_batches": train_batches,
            "val_batches": val_batches,
            "upper_directions": upper_directions,
            "fista_diagnostics": fista_diagnostics,
            "fista_mode": fista_mode,
            "lower_max_iter": lower_max_iter,
            "sgd_step_decay": sgd_step_decay,
            "sgd_patch_mode": sgd_patch_mode,
            "sgd_patch_fraction": sgd_patch_fraction,
        }
        if logs_dict['ML']:
            logs_dict["L"] = L
        else:
            logs_dict["L"] = None

        total_iterations = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_batches = 0
            for _, gt in limited_batches(train_dataloader, train_batches):
                gt = gt.to(device)
                epoch_batches += 1

                # Formulate Dynamic Accuracy bound
                with torch.no_grad():
                    if regulariser_type in ["CT_TV", "CT_FoE"]:
                        dyn_L_reg = 8.0 * opt.model.alpha.item() / opt.model.nu.item()
                    elif regulariser_type == "TV":
                        dyn_L_reg = 8.0 * opt.model.tv_weight.item() / opt.model.smoothing.item()
                    else:
                        dyn_L_reg = 0.0

                    dyn_L_total = L_data + dyn_L_reg
                    if lower_max_iter is not None and lower_max_iter > 0:
                        dyn_finest_iter = int(lower_max_iter)
                    else:
                        dyn_s_theory = int(np.sqrt(dyn_L_total) * condition_multiplier)
                        dyn_s_step = max(50, min(dyn_s_theory, max_allowable_s))
                        dyn_finest_iter = dyn_s_step * L_levels
                    levels = build_lower_iteration_levels(L_levels, dyn_finest_iter, alg=lower_level_alg)

                N_per_level_upper = sets["N_per_level_upper"]
                N_per_level_lower = sets["N_per_level_lower"]
                opt.gt = gt
                if sets["ML"]:
                    loss, psnr_values, ssim_values, iterations = opt.multilevel_gradient_step(
                        levels=levels, N_per_level_upper=N_per_level_upper,
                        N_per_level_lower=N_per_level_lower, sigma=1e-1,
                        warm_start=False, normalize_grad=False, verbose=True
                    )
                else:
                    loss, psnr_values, ssim_values, iterations = opt.single_level_gradient_step(
                        level=levels[-1], N_upper=N_per_level_upper[0],
                        N_lower=N_per_level_lower[0], sigma=1e-1,
                        warm_start=False, normalize_grad=False, verbose=True
                    )
                logs_dict["loss"].append(loss.item() if hasattr(loss, 'item') else float(loss))
                logs_dict["psnr"].append(psnr_values.item() if hasattr(psnr_values, 'item') else float(psnr_values))
                logs_dict["ssim"].append(ssim_values.item() if hasattr(ssim_values, 'item') else float(ssim_values))
                total_iterations += iterations
                logs_dict["cost"].append(total_iterations)
                epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)

            # Validation
            val_psnrs = []
            val_ssims = []
            fista_diag = []
            plot_gt = None
            plot_x_init = None
            plot_x_hat_out = None
            plot_init_psnr = None
            plot_recon_psnr = None

            # Validate at the same finest accuracy used by the training
            # estimator. A much longer fixed-step patch-SGD validation run can
            # over-regularize FoE reconstructions and report the wrong signal.
            val_max_iter = 2000 if "CT" in problem else int(sets["level"][-1])

            for _, gt in limited_batches(val_dataloader, val_batches):
                gt = gt.to(device)
                y = physics(gt)
                x_hat_out, solver_info = reconstruct(
                    y=y, physics=physics, data_fidelity=data_fidelity,
                    regularizer=opt.model, lmbd=lmbd,
                    lower_level_alg=lower_level_alg,
                    lower_level_step=lower_level_step,
                    max_iter=val_max_iter, tol=1e-4,
                    verbose=False,
                    return_stats=fista_diagnostics and lower_level_alg == "FISTA",
                    fista_mode=fista_mode,
                    sgd_step_decay=sgd_step_decay,
                    sgd_patch_mode=sgd_patch_mode,
                    sgd_patch_fraction=sgd_patch_fraction,
                )
                if fista_diagnostics and lower_level_alg == "FISTA" and isinstance(solver_info, dict):
                    fista_diag.append(solver_info)
                x_init = physics.A_dagger(y)
                psnr_val = psnr(gt, x_hat_out)
                ssim_val = ssim(gt, x_hat_out)
                psnr_fbp = psnr(gt, x_init)

                val_psnrs.append(psnr_val.mean().item())
                val_ssims.append(ssim_val.mean().item())
                if plot_gt is None:
                    plot_gt = gt
                    plot_x_init = x_init
                    plot_x_hat_out = x_hat_out
                    plot_init_psnr = psnr(gt[:1], x_init[:1]).mean().item()
                    plot_recon_psnr = psnr(gt[:1], x_hat_out[:1]).mean().item()

            # Average metrics across all validation batches
            avg_val_psnr = sum(val_psnrs) / len(val_psnrs)
            avg_val_ssim = sum(val_ssims) / len(val_ssims)
            average_loss = epoch_loss / max(1, epoch_batches)

            logs_dict["val_psnr"].append(avg_val_psnr)
            logs_dict["val_ssim"].append(avg_val_ssim)

            diag_msg = ""
            if fista_diag:
                avg_L = sum(float(d["L"].item() if hasattr(d["L"], "item") else d["L"]) for d in fista_diag) / len(fista_diag)
                avg_restarts = sum(float(d.get("restarts", 0)) for d in fista_diag) / len(fista_diag)
                avg_backtracks = sum(float(d.get("backtracks", 0)) for d in fista_diag) / len(fista_diag)
                diag_msg = (
                    f", FISTA L: {avg_L:.2e}, restarts: {avg_restarts:.1f}, "
                    f"backtracks: {avg_backtracks:.1f}"
                )
                if all("rel_change" in d for d in fista_diag):
                    avg_rel = sum(float(d["rel_change"].item() if hasattr(d.get("rel_change"), "item") else d["rel_change"]) for d in fista_diag) / len(fista_diag)
                    diag_msg += f", rel: {avg_rel:.2e}"

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}, Val PSNR: {avg_val_psnr:.2f}, Val SSIM: {avg_val_ssim:.4f}{diag_msg}")

            if epoch % 1 == 0:
                is_color_image = plot_gt.shape[1] == 3

                def show_image(ax, tensor, title):
                    image = tensor[0].detach().cpu().clamp(0, 1)
                    if is_color_image:
                        ax.imshow(image.permute(1, 2, 0).numpy())
                    else:
                        vmax = image[0].max().item()
                        ax.imshow(image[0].numpy(), cmap='bone', vmin=0, vmax=vmax)
                    ax.set_title(title)
                    ax.axis('off')

                plt.figure(figsize=(18, 6))
                plt.subplot(1, 3, 1)
                show_image(plt.gca(), plot_gt, "Ground Truth Image")
                plt.subplot(1, 3, 2)
                init_title = "Noisy Image" if problem == "Denoising" else "FBP Image"
                show_image(plt.gca(), plot_x_init, f"{init_title} - PSNR: {plot_init_psnr:.2f} dB")
                plt.subplot(1, 3, 3)
                show_image(plt.gca(), plot_x_hat_out, f"Reconstructed - PSNR: {plot_recon_psnr:.2f} dB")
                plt.suptitle(f"Epoch {epoch+1} Evaluation")

                # Progressively update the image every epoch so you can watch it learn dynamically
                os.makedirs("logs", exist_ok=True)
                plt.savefig(f"logs/reconstructed_image_{regulariser_type}_{run_problem_tag}_{lower_level_alg}_{sets['ML']}_{epsilon}{filename_suffix}.png",
                            bbox_inches='tight', dpi=300)
                plt.close()

                logs_dict["saved_params"] = opt.model.state_dict()

                # Append formatted parameters to history
                current_params = {k: v.detach().clone().cpu().numpy() for k, v in opt.model.state_dict().items()}
                logs_dict["param_history"].append(current_params)

            if total_iterations >= budget:
                print(f"Reached budget of {budget} iterations, stopping training.")
                break

            # Save logs
            os.makedirs("logs", exist_ok=True)
            torch.save(logs_dict, f"logs/logs_{regulariser_type}_{run_problem_tag}_{lower_level_alg}_{sets['ML']}_{epsilon}{filename_suffix}.pt")


# PDE experiment (separate due to different structure)
def run_pde_experiment(
    I_grid=33,
    K=20,
    gamma=0.0001,
    lambda_true=0.1,
    num_iters=500,
    eta=0.01,
    beta_0=1.0,
    batch_0=1,
    epsilon_0=0.01,
    epsilon=5e-1,
    lower_level_alg="GD",
    use_multilevel=True,
):
    """
    Run the PDE bilevel optimization experiment.

    This experiment has a different structure from the imaging problems
    since it uses vector-valued unknowns and a closed-form lower-level solution.
    """
    from problems.pde_elliptic import setup_pde_experiment

    print(f"{'='*60}")
    print(f"  PDE Experiment | I={I_grid}, K={K}, gamma={gamma}")
    print(f"  Lower-level: {'Exact' if True else lower_level_alg}")
    print(f"{'='*60}")

    pde_model, bilevel_model, config = setup_pde_experiment(
        I_grid=I_grid, K=K, gamma=gamma, lambda_true=lambda_true, device=device
    )

    # Validation set
    validation_size = 50
    x_validation = pde_model.sample_ground_truth(validation_size)

    # Optimizer
    optimizer = torch.optim.SGD(bilevel_model.parameters(), lr=beta_0)

    # Gaussian smoothing matrix
    n_params = sum(p.numel() for p in bilevel_model.parameters())

    # Level schedule
    L, N_l = calculate_levels(epsilon=epsilon, alg=lower_level_alg)

    logs = {
        "loss": [],
        "val_loss": [],
        "lambda_history": [],
        "cost": [],
        "ML": use_multilevel,
        "epsilon": epsilon,
        "problem": "PDE",
    }

    total_cost = 0
    for k in range(1, num_iters + 1):
        # Schedules
        beta_k = beta_0 / np.sqrt(k)
        batch_k = max(1, round(np.sqrt(k) * batch_0))
        epsilon_k = epsilon_0 / np.sqrt(k)

        # Update learning rate
        for pg in optimizer.param_groups:
            pg['lr'] = beta_k

        # Sample training data
        u_batch = pde_model.sample_ground_truth(batch_k)

        # Gaussian smoothing DFO gradient estimation
        grad_est = {n: torch.zeros_like(p) for n, p in bilevel_model.named_parameters()}
        noise_dict = {}
        for name, param in bilevel_model.named_parameters():
            noise_dict[name] = torch.randn_like(param)

        # Perturb + evaluate
        with torch.no_grad():
            for name, param in bilevel_model.named_parameters():
                param.add_(eta * noise_dict[name])

        loss_pos, _ = bilevel_model(u_batch)

        with torch.no_grad():
            for name, param in bilevel_model.named_parameters():
                param.sub_(eta * noise_dict[name])

        loss_base, _ = bilevel_model(u_batch)

        # Gradient estimate
        for name in grad_est:
            grad_est[name] = ((loss_pos - loss_base) / eta) * noise_dict[name]

        # Gradient clipping
        grad_norm = torch.sqrt(sum((g ** 2).sum() for g in grad_est.values()))
        if grad_norm > 10:
            for name in grad_est:
                grad_est[name] *= 10 / grad_norm

        # Update
        optimizer.zero_grad()
        for name, param in bilevel_model.named_parameters():
            param.grad = grad_est[name]
        optimizer.step()

        # Clamp lambda
        with torch.no_grad():
            bilevel_model.log10_lambda.clamp_(-7, 7)

        total_cost += batch_k
        current_params = bilevel_model.get_learned_params()

        # Validation
        if k % 50 == 0 or k == 1:
            with torch.no_grad():
                val_loss, x_hat_val = bilevel_model(x_validation)
            logs["loss"].append(loss_base.item())
            logs["val_loss"].append(val_loss.item())
            logs["lambda_history"].append(current_params["log10_lambda"])
            logs["cost"].append(total_cost)
            print(f"Iter {k}/{num_iters}, Loss: {loss_base.item():.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, "
                  f"lambda: {current_params['lambda']:.6f} (log10: {current_params['log10_lambda']:.3f})")

            # Plotting 2D fields
            n = pde_model.n_interior
            plt.figure(figsize=(15, 5))

            # Ground Truth
            plt.subplot(1, 3, 1)
            u_img = x_validation[0].cpu().numpy().reshape(n, n)
            plt.imshow(u_img, cmap='jet')
            plt.title("Ground Truth u")
            plt.colorbar()

            # Reconstruction
            plt.subplot(1, 3, 2)
            x_hat_img = x_hat_val[0].cpu().numpy().reshape(n, n)
            plt.imshow(x_hat_img, cmap='jet')
            plt.title("Reconstruction x*")
            plt.colorbar()

            # Error + Sensors
            plt.subplot(1, 3, 3)
            error_img = np.abs(u_img - x_hat_img)
            plt.imshow(error_img, cmap='hot')
            plt.title("Abs Error + Sensors")
            plt.colorbar()

            # Plot sensors
            obs_indices = pde_model.get_obs_indices_from_weights(bilevel_model.w.detach())
            for idx in obs_indices:
                sx = idx % n
                sy = idx // n
                plt.plot(sx, sy, 'ro', markersize=4)

            plt.tight_layout()
            os.makedirs("logs", exist_ok=True)
            plt.savefig(f"logs/pde_fields_iter_{k}.png", bbox_inches='tight', dpi=300)
            plt.close()

    # Save results
    os.makedirs("logs", exist_ok=True)
    torch.save(logs, f"logs/logs_PDE_{lower_level_alg}_{use_multilevel}_{epsilon}.pt")
    print(f"\nFinal learned parameters: {bilevel_model.get_learned_params()}")

    return bilevel_model, logs


# CLI entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilevel Bilevel DFO Experiments")
    parser.add_argument("--problem", type=str, default="Denoising",
                        choices=["Denoising", "Inpainting", "CT", "CT_LoDo", "PDE"],
                        help="Problem type")
    parser.add_argument("--regulariser", type=str, default="TV",
                        choices=["TV", "FoE", "CT_TV", "CT_FoE"],
                        help="Regulariser type")
    parser.add_argument("--lower_alg", type=str, default="SGD",
                        choices=["GD", "SGD", "FISTA"],
                        help="Lower-level optimization algorithm")
    parser.add_argument("--lower_step", type=float, default=1e-1,
                        help="Lower-level step size")
    parser.add_argument("--epsilon", type=float, default=5e-1,
                        help="Target accuracy")
    parser.add_argument("--budget", type=float, default=1e7,
                        help="Computational budget")
    parser.add_argument("--upper_lr", type=float, default=1e-3,
                        help="Upper-level learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--rho", type=float, default=0.3,
                        help="Contraction rate for GD level schedule")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of the dataset to securely use (e.g. 0.1 for 10%)")
    parser.add_argument("--img_size", type=int, default=0,
                        help="Target image size to load the data (integer, e.g. 128). Defaults are applied if 0.")
    parser.add_argument("--num_angles", type=int, default=0,
                        help="Number of CT projection angles. Defaults are CT=6 and CT_LoDo=16.")
    parser.add_argument("--noise_level", type=float, default=-1.0,
                        help="Gaussian CT noise standard deviation. Defaults are CT=0.01 and CT_LoDo=0.03.")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Training batch size. If 0, use the minimum needed by the MLMC/SLMC sample schedule.")
    parser.add_argument("--val_batch_size", type=int, default=0,
                        help="Validation batch size. If 0, use min(5, training batch size).")
    parser.add_argument("--val_len", type=int, default=1,
                        help="Number of BSDS validation/test images to keep for denoising and inpainting.")
    parser.add_argument("--train_batches", type=int, default=0,
                        help="Maximum training batches/oracle steps per epoch. If 0, use the full DataLoader.")
    parser.add_argument("--val_batches", type=int, default=0,
                        help="Maximum validation batches per validation pass. If 0, use the full validation loader.")
    parser.add_argument("--upper_directions", type=int, default=1,
                        help="Number of Gaussian smoothing directions averaged per level and upper update.")
    parser.add_argument("--fista_diagnostics", action="store_true",
                        help="Report validation FISTA line-search/restart diagnostics.")
    parser.add_argument("--fista_mode", type=str, default="legacy",
                        choices=["legacy", "restart", "monotone"],
                        help="FISTA reconstruction mode: legacy fixed-step, restart fixed-step, or monotone backtracking.")
    parser.add_argument("--lower_max_iter", type=int, default=0,
                        help="Override the finest lower-level iteration count. If 0, use the automatic MLMC schedule.")
    parser.add_argument("--color", action="store_true",
                        help="Use RGB BSDS500 images for denoising. Intended for color FoE experiments.")
    parser.add_argument("--sgd_step_decay", type=str, default="none",
                        choices=["none", "sqrt", "linear"],
                        help="Step decay for the patch-SGD lower solver.")
    parser.add_argument("--sgd_patch_mode", type=str, default="patch",
                        choices=["patch", "full", "mask"],
                        help="Spatial stochasticity mode for the SGD lower solver.")
    parser.add_argument("--sgd_patch_fraction", type=float, default=0.5,
                        help="Patch side-length fraction for patch mode, or mask density scale for mask mode.")
    parser.add_argument("--freeze_foe_kernels", action="store_true",
                        help="Freeze FoE convolution kernels and learn only weights/smoothing.")
    parser.add_argument("--foe_num_experts", type=int, default=0,
                        help="Override the number of FoE convolution experts. Use 0 for the default.")
    parser.add_argument("--run_tag", type=str, default="",
                        help="Optional suffix appended to log and reconstruction filenames.")

    args = parser.parse_args()

    if args.problem in ["CT", "CT_LoDo"] and args.lower_alg == "SGD":
        print(f"Note: Defaulting to GD for {args.problem} because patch-based SGD is not suitable for Radon transform.")
        args.lower_alg = "GD"

    if args.problem == "PDE":
        run_pde_experiment(
            epsilon=args.epsilon,
            lower_level_alg=args.lower_alg,
            num_iters=args.num_epochs * 20,  # Increase iterations for better convergence
            beta_0=args.upper_lr,            # Pipe CLI upper_lr to PDE learning rate
        )
    else:
        run_experiment(
            problem=args.problem,
            regulariser_type=args.regulariser,
            lower_level_alg=args.lower_alg,
            lower_level_step=args.lower_step,
            epsilon=args.epsilon,
            budget=args.budget,
            upper_lr=args.upper_lr,
            num_epochs=args.num_epochs,
            rho=args.rho,
            subset_fraction=args.subset,
            img_size=args.img_size if args.img_size > 0 else None,
            num_angles=args.num_angles if args.num_angles > 0 else None,
            noise_level=args.noise_level if args.noise_level >= 0 else None,
            batch_size=args.batch_size if args.batch_size > 0 else None,
            val_batch_size=args.val_batch_size if args.val_batch_size > 0 else None,
            val_len=args.val_len,
            train_batches=args.train_batches,
            val_batches=args.val_batches,
            upper_directions=args.upper_directions,
            fista_diagnostics=args.fista_diagnostics,
            fista_mode=args.fista_mode,
            lower_max_iter=args.lower_max_iter if args.lower_max_iter > 0 else None,
            color=args.color,
            sgd_step_decay=args.sgd_step_decay,
            sgd_patch_mode=args.sgd_patch_mode,
            sgd_patch_fraction=args.sgd_patch_fraction,
            freeze_foe_kernels=args.freeze_foe_kernels,
            foe_num_experts=args.foe_num_experts if args.foe_num_experts > 0 else None,
            run_tag=args.run_tag,
        )
# %%
