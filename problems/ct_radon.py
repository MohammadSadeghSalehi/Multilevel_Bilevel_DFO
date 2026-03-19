"""
CT/Radon Transform experiment module.

Uses deepinv.physics.Tomography for the forward model.
Learns regularization parameters (lambda, alpha, nu) for Tikhonov + smoothed TV.
"""
import torch
import torch.nn as nn
import numpy as np
from deepinv.physics import Tomography, GaussianNoise
from deepinv.optim import L2
from problems.phantoms import PhantomDataset


def setup_ct_experiment(
    img_size=64,
    num_angles=6,
    noise_level=0.01,
    device="cpu",
):
    """
    Set up the CT/Radon transform experiment.

    Matches the MATLAB code in 2d_OEM_radontransform_Neil.

    Args:
        img_size: Image size (n x n)
        num_angles: Number of illumination angles
        noise_level: Gaussian noise standard deviation
        device: Torch device

    Returns:
        physics: DeepInverse Tomography physics operator
        data_fidelity: L2 data fidelity
        train_dataset: PhantomDataset for training
        val_dataset: PhantomDataset for validation
        config: Dictionary with experiment configuration
    """
    # Set up angles — uniformly spaced in [0, 180)
    angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1]  # Exclude 180 (same as 0)

    # Create Tomography physics operator
    physics = Tomography(
        angles=angles,
        img_width=img_size,
        circle=False,
        normalize=True,
        noise_model=GaussianNoise(sigma=noise_level),
        device=device,
    ).to(device)

    # Data fidelity
    data_fidelity = L2(sigma=1.0)

    # Datasets
    train_dataset = PhantomDataset(n=img_size, length=300, phantom_type="triangle")
    val_dataset = PhantomDataset(n=img_size, length=10, phantom_type="triangle")

    config = {
        "problem": "CT",
        "img_size": img_size,
        "num_angles": num_angles,
        "noise_level": noise_level,
        "lmbd": 1.0,
        "lower_level_step": 1e-1,
    }

    return physics, data_fidelity, train_dataset, val_dataset, config


class CTSmoothedTV(nn.Module):
    """
    Combined Tikhonov + Smoothed TV regularizer for CT.

    Matches the MATLAB formulation:
        R(x) = lambda * ||L*x||^2 + alpha * sum(sqrt(|Dx|^2 + nu^2))

    where L is the Tikhonov matrix (identity), D is the finite difference operator,
    and lambda, alpha, nu are learnable in log-space.
    """

    def __init__(self, img_size=64, device="cpu"):
        super().__init__()
        # Learnable regularization parameters (in log10 space matching MATLAB)
        self.log10_lambda = nn.Parameter(torch.tensor(-5.0))  # Tikhonov weight
        self.log10_alpha = nn.Parameter(torch.tensor(-2.0))   # TV weight
        self.log10_nu = nn.Parameter(torch.tensor(-2.0))      # TV smoothing

        self.img_size = img_size

        # Forward difference kernels for TV
        self.register_buffer(
            "grad_x", torch.tensor([[[[0, -1, 1]]]], dtype=torch.float32)
        )
        self.register_buffer(
            "grad_y", torch.tensor([[[[0], [-1], [1]]]], dtype=torch.float32)
        )

    @property
    def lam(self):
        return 10.0 ** self.log10_lambda

    @property
    def alpha(self):
        return 10.0 ** self.log10_alpha

    @property
    def nu(self):
        return 10.0 ** self.log10_nu

    def compute_gradients(self, x):
        import torch.nn.functional as F
        B, C, H, W = x.shape
        gx = F.conv2d(x, self.grad_x.expand(C, -1, -1, -1), groups=C, padding=(0, 1))
        gy = F.conv2d(x, self.grad_y.expand(C, -1, -1, -1), groups=C, padding=(1, 0))
        return gx, gy

    def g(self, x):
        """Compute regularization energy."""
        # Tikhonov: lambda * ||x||^2
        tik_energy = self.lam * torch.sum(x ** 2, dim=(1, 2, 3))

        # Smoothed TV: alpha * sum(sqrt(|Dx|^2 + nu^2))
        gx, gy = self.compute_gradients(x)
        tv_energy = self.alpha * torch.sqrt(gx ** 2 + gy ** 2 + self.nu ** 2).sum(dim=(1, 2, 3))

        return tik_energy + tv_energy

    def grad(self, x, get_energy=False):
        """Compute gradient of the regularizer."""
        import torch.nn.functional as F
        B, C, H, W = x.shape

        # Tikhonov gradient: 2 * lambda * x
        tik_grad = 2.0 * self.lam * x

        # Smoothed TV gradient
        gx, gy = self.compute_gradients(x)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + self.nu ** 2)

        gx_norm = gx / grad_mag
        gy_norm = gy / grad_mag

        grad_x_kernel = torch.flip(self.grad_x, dims=[2, 3])
        grad_y_kernel = torch.flip(self.grad_y, dims=[2, 3])

        div_x = F.conv2d(gx_norm, grad_x_kernel.expand(C, -1, -1, -1), groups=C, padding=(0, 1))
        div_y = F.conv2d(gy_norm, grad_y_kernel.expand(C, -1, -1, -1), groups=C, padding=(1, 0))

        tv_grad = self.alpha * (div_x + div_y)

        total_grad = tik_grad + tv_grad

        if get_energy:
            energy = self.g(x)
            return energy, total_grad
        return total_grad
