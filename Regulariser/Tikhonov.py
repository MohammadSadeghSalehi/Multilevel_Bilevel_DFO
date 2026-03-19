"""
Tikhonov regularizer compatible with the deepinv Prior interface.
"""
import torch
import torch.nn as nn
from deepinv.optim import Prior


class TikhonovRegularizer(Prior):
    """
    Tikhonov regularizer: R(x) = lambda * ||L x||^2

    Where L is a fixed precision matrix and lambda is learnable (in log space).

    Args:
        n: Dimension of the unknown parameter
        L: Precision matrix (numpy array or torch tensor). If None, uses identity.
        lambda_init: Initial value of lambda
        learn_lambda: Whether to learn lambda
        device: Torch device
    """

    def __init__(
        self,
        n=None,
        L=None,
        lambda_init=1.0,
        learn_lambda=True,
        device="cpu",
    ):
        super().__init__()

        if learn_lambda:
            self.log_lambda = nn.Parameter(torch.log(torch.tensor(lambda_init)))
        else:
            self.register_buffer("log_lambda", torch.log(torch.tensor(lambda_init)))

        if L is not None:
            if not isinstance(L, torch.Tensor):
                L = torch.from_numpy(L).float()
            self.register_buffer("L", L.to(device))
        else:
            self.L = None  # Use identity

    @property
    def lam(self):
        return torch.exp(self.log_lambda)

    def g(self, x):
        """
        Compute Tikhonov energy: lambda * ||L x||^2

        Args:
            x: Input tensor. For images: (B, C, H, W). For vectors: (B, N).

        Returns:
            Energy per sample in batch
        """
        if x.dim() == 4:
            # Image case: flatten spatial dims
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
        else:
            x_flat = x

        if self.L is not None:
            Lx = x_flat @ self.L.T
        else:
            Lx = x_flat

        energy = self.lam * torch.sum(Lx ** 2, dim=1)
        return energy

    def grad(self, x, get_energy=False):
        """
        Compute gradient: 2 * lambda * L' L x

        Args:
            x: Input tensor
            get_energy: If True, also return the energy

        Returns:
            gradient (and optionally energy)
        """
        original_shape = x.shape
        if x.dim() == 4:
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
        else:
            x_flat = x

        if self.L is not None:
            # grad = 2 * lambda * L'L x
            Lx = x_flat @ self.L.T
            gradient = 2.0 * self.lam * (Lx @ self.L)
        else:
            gradient = 2.0 * self.lam * x_flat

        gradient = gradient.reshape(original_shape)

        if get_energy:
            energy = self.g(x)
            return energy, gradient
        return gradient
