#%%
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

# set seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# ─── Level schedule calculations ─────────────────────────────────────────
def calculate_levels(epsilon, alg="SGD", rho=0.3):
    """
    Calculate number of levels L and samples per level N_l
    for the multilevel estimator, given a target accuracy epsilon.

    Matches Table 1 in the paper.
    """
    if alg == "SGD":
        L = int(2 * epsilon ** (-1))
        N_l = [int(4 * epsilon ** (-1) * (l + 1) ** (-1)) for l in range(L)]
    elif alg == "GD":
        L = int(np.log(3 * epsilon ** (-1)) / np.log(rho ** (-1)))
        N_l = [int(epsilon ** (-1) * (l + 1) ** 2 * rho ** (l + 1) * np.pi ** 2 / 2) for l in range(L)]
    elif alg == "FISTA":
        # From Theorem 4 / Table 1: L = sqrt(2) * eps^{-1/2}, N_l = 4*eps^{-1}*(l+1)^{-1}
        L = max(1, int(np.sqrt(2) * epsilon ** (-0.5)))
        N_l = [max(1, int(4 * epsilon ** (-1) * (l + 1) ** (-1))) for l in range(L)]
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    # Ensure at least 1 sample per level
    N_l = [max(1, n) for n in N_l]
    L = max(1, L)
    return L, N_l


# ─── Experiment setup ────────────────────────────────────────────────────
def setup_denoising_experiment(regulariser_type, img_size_x=128, img_size_y=128,
                                train_len=300, test_len=1, batch_size=300):
    """Set up the TV denoising experiment (existing experiment)."""
    noise_level = 25 / 255
    physics = Denoising(noise_model=GaussianNoise(sigma=noise_level), device=device).to(device)
    physics.A_dagger = physics.A_adjoint
    data_fidelity = L2(sigma=1.0)

    train_dataset = get_dataset("BSDS500", test=False,
                                transform=Compose([RandomCrop((img_size_x, img_size_y)),
                                                    Grayscale(num_output_channels=1)]))
    val_dataset = get_dataset("BSDS500", test=True,
                              transform=Compose([Resize((128, 128)),
                                                  Grayscale(num_output_channels=1)]))
    train_set = torch.utils.data.Subset(train_dataset, range(train_len))
    val_set = torch.utils.data.Subset(val_dataset, range(96, 97))

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=test_len, shuffle=False, drop_last=True)

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


def setup_inpainting_experiment(regulariser_type, img_size_x=128, img_size_y=128,
                                 train_len=300, test_len=1, batch_size=300):
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
    val_set = torch.utils.data.Subset(val_dataset, range(96, 97))

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=test_len, shuffle=False, drop_last=True)

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
                              noise_level=0.01, train_len=300, test_len=1, batch_size=50):
    """Set up the CT/Radon transform experiment using deepinv.physics.Tomography."""
    from problems.ct_radon import CTSmoothedTV
    from problems.phantoms import PhantomDataset

    # Set up angles — uniformly spaced in [0, 180)
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
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=test_len, shuffle=False, drop_last=True)

    if regulariser_type == "TV":
        regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
    elif regulariser_type == "CT_TV":
        regularizer = CTSmoothedTV(img_size=img_size, device=device).to(device)
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


# ─── Reconstruction dispatch ────────────────────────────────────────────
def reconstruct(y, physics, data_fidelity, regularizer, lmbd, lower_level_alg,
                lower_level_step, max_iter=500, tol=1e-4):
    """Dispatch to the correct lower-level solver."""
    if lower_level_alg == "GD":
        x_hat, iters = reconstruct_gd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, detach_grads=True, verbose=False,
            return_stats=False, BB=False, LS=True,
        )
    elif lower_level_alg == "SGD":
        patch_size = (y.shape[2] // 2, y.shape[3] // 2)
        x_hat, iters = reconstruct_sgd(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, patch_size=patch_size,
            detach_grads=True, verbose=False, return_stats=False,
        )
    elif lower_level_alg == "FISTA":
        x_hat, iters = reconstruct_fista(
            y=y, physics=physics, data_fidelity=data_fidelity,
            regularizer=regularizer, lamda=lmbd, step_size=lower_level_step,
            max_iter=max_iter, tol=tol, detach_grads=True, verbose=False,
            return_stats=False,
        )
    else:
        raise ValueError(f"Unknown lower-level algorithm: {lower_level_alg}")
    return x_hat, iters


# ─── Main training loop ─────────────────────────────────────────────────
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
    print(f"{'='*60}")

    # Setup experiment
    if problem == "Denoising":
        exp = setup_denoising_experiment(regulariser_type)
    elif problem == "Inpainting":
        exp = setup_inpainting_experiment(regulariser_type)
    elif problem == "CT":
        exp = setup_ct_experiment_full(regulariser_type)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    physics = exp["physics"]
    data_fidelity = exp["data_fidelity"]
    lmbd = exp["lmbd"]
    train_dataloader = exp["train_dataloader"]
    val_dataloader = exp["val_dataloader"]

    psnr = PSNR()

    # Print initial quality
    for gt in val_dataloader:
        y = physics(gt.to(device))
        x_init = physics.A_dagger(y)
        print("PSNR before reconstruction: ", psnr(gt.to(device), x_init))

    # Level schedule
    L, N_l = calculate_levels(epsilon=epsilon, alg=lower_level_alg, rho=rho)
    max_lower_iter = [50 for _ in range(len(N_l))]
    N_per_level_upper = [1 for _ in range(len(N_l))]
    print(f"Number of levels: {L}, Number of samples per level: {N_l}")

    # Settings: ML vs SL comparison
    settings = [
        {"level": max_lower_iter, "N_per_level_upper": N_per_level_upper, "N_per_level_lower": N_l, "ML": True},
        {"level": max_lower_iter, "N_per_level_upper": N_per_level_upper,
         "N_per_level_lower": [max(N_l) for _ in range(len(N_l))], "ML": False},
    ]

    for sets in settings:
        # Re-initialize regularizer for fair comparison
        if regulariser_type == "TV":
            regularizer = SmoothedTV(tv_weight=1e-2, smoothing=1e-2, learn_tv_weight=True, learn_smoothing=True, device=device).to(device)
        elif regulariser_type == "FoE":
            regularizer = FieldOfExperts(num_experts=4, device=device).to(device)
        elif regulariser_type == "CT_TV":
            from problems.ct_radon import CTSmoothedTV
            regularizer = CTSmoothedTV(img_size=64, device=device).to(device)
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
        )

        # Compute initial PSNR and loss
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
            "algo": lower_level_alg,
            "upper_algo": "SGD",
            "warm_start": True,
            "reg": regulariser_type,
            "problem": problem,
            "ML": sets["ML"],
            "saved_params": None,
            "epsilon": epsilon,
        }
        if logs_dict['ML']:
            logs_dict["L"] = L
        else:
            logs_dict["L"] = None

        total_iterations = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for gt in train_dataloader:
                gt = gt.to(device)
                levels = sets["level"]
                N_per_level_upper = sets["N_per_level_upper"]
                N_per_level_lower = sets["N_per_level_lower"]
                opt.gt = gt
                loss, psnr_values, iterations = opt.multilevel_gradient_step(
                    levels=levels, N_per_level_upper=N_per_level_upper,
                    N_per_level_lower=N_per_level_lower, sigma=1e-1,
                    warm_start=False, normalize_grad=False, verbose=True
                )
                logs_dict["loss"].append(loss.item() if hasattr(loss, 'item') else float(loss))
                logs_dict["psnr"].append(psnr_values.item() if hasattr(psnr_values, 'item') else float(psnr_values))
                total_iterations += iterations
                logs_dict["cost"].append(total_iterations)
                epoch_loss += loss.item() if hasattr(loss, 'item') else float(loss)

            # Validation
            for gt in val_dataloader:
                gt = gt.to(device)
                y = physics(gt)
                x_hat_out, _ = reconstruct(
                    y=y, physics=physics, data_fidelity=data_fidelity,
                    regularizer=opt.model, lmbd=lmbd,
                    lower_level_alg=lower_level_alg,
                    lower_level_step=lower_level_step,
                    max_iter=500, tol=1e-4,
                )
                psnr_val = psnr(gt, x_hat_out)
                average_loss = epoch_loss / len(train_dataloader)
                logs_dict["val_psnr"].append(psnr_val.mean().item())
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss}, Val PSNR: {psnr_val.mean().item()}")

                if epoch % 1 == 0:
                    plt.figure(figsize=(18, 6))
                    plt.subplot(1, 3, 1)
                    plt.imshow(gt[0, 0].cpu().numpy(), cmap='gray')
                    plt.title("Ground Truth Image")
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(y[0, 0].cpu().numpy(), cmap='gray')
                    plt.title("Noisy Image")
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.imshow(x_hat_out[0, 0].cpu().detach().numpy(), cmap='gray')
                    plt.title("Reconstructed Image")
                    plt.axis('off')
                    plt.suptitle(f"Reconstructed Image - PSNR: {psnr_val.mean().item():.2f} dB")
                    if epoch == num_epochs - 1 or total_iterations >= budget:
                        print(f"Saving reconstructed image for {regulariser_type} {problem} at level {sets['level']}")
                        os.makedirs("logs", exist_ok=True)
                        plt.savefig(f"logs/reconstructed_image_{regulariser_type}_{problem}_{lower_level_alg}_{sets['ML']}_{epsilon}.png",
                                    bbox_inches='tight', dpi=300)
                        plt.close()
                    plt.close()

                logs_dict["saved_params"] = opt.model.state_dict()
            if total_iterations >= budget:
                print(f"Reached budget of {budget} iterations, stopping training.")
                break

            # Save logs
            os.makedirs("logs", exist_ok=True)
            torch.save(logs_dict, f"logs/logs_{regulariser_type}_{problem}_{lower_level_alg}_{sets['ML']}_{epsilon}.pt")


# ─── PDE experiment (separate due to different structure) ────────────────
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
    print(f"  PDE Experiment | I={I_grid}, K={K}, γ={gamma}")
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
                  f"λ: {current_params['lambda']:.6f} (log10: {current_params['log10_lambda']:.3f})")

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


# ─── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilevel Bilevel DFO Experiments")
    parser.add_argument("--problem", type=str, default="Denoising",
                        choices=["Denoising", "Inpainting", "CT", "PDE"],
                        help="Problem type")
    parser.add_argument("--regulariser", type=str, default="TV",
                        choices=["TV", "FoE", "CT_TV"],
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
    parser.add_argument("--upper_lr", type=float, default=2e-3,
                        help="Upper-level learning rate")
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="Number of epochs")
    parser.add_argument("--rho", type=float, default=0.3,
                        help="Contraction rate for GD level schedule")

    args = parser.parse_args()

    if args.problem == "CT" and args.lower_alg == "SGD":
        print("Note: Defaulting to GD for CT problem because patch-based SGD is not suitable for Radon transform.")
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
        )
# %%
