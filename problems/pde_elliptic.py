"""
2D Elliptic PDE Inverse Problem module.

Ported from MATLAB code in 2d_linear_PDE_Neil.
Forward model: 2D Laplace equation -Δp = u on [0,1]² with zero BCs.
Inverse problem: recover u from K pointwise observations of p.
Regularization: Tikhonov with covariance prior.
Learnable parameters: [log_lambda, w_1, ..., w_K] (regularization + observation locations).
"""
import torch
import torch.nn as nn
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def create_laplacian_2d(I_grid):
    """
    Create the 2D discrete Laplacian matrix on an (I-1)x(I-1) interior grid.

    Matches MATLAB: A = I^2 * createmat(I-1)

    Args:
        I_grid: Number of grid points per side (including boundaries), so interior = I_grid - 1

    Returns:
        A: Sparse Laplacian matrix of size ((I-1)^2, (I-1)^2)
    """
    n = I_grid - 1  # interior points per dimension
    N = n * n
    h = 1.0 / I_grid  # mesh spacing

    # 1D Laplacian (tridiagonal)
    e = np.ones(n)
    T = sparse.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1], shape=(n, n))

    # 2D Laplacian via Kronecker product
    I_n = sparse.eye(n)
    A = sparse.kron(I_n, T) + sparse.kron(T, I_n)

    # Scale by 1/h^2 = I^2
    A = (I_grid ** 2) * A

    return A


def create_covariance_prior(A, tau=0.1, alpha=1.5, scale=10.0):
    """
    Create the covariance matrix C0 = scale * (tau^2 * I + A)^{-alpha}.

    For alpha=1.5, we compute (tau^2 I + A)^{-1} and then take a matrix power.
    For simplicity and numerical stability, we use alpha=1 (single inverse).

    Args:
        A: Laplacian matrix (sparse)
        tau: Scaling parameter
        alpha: Power parameter (uses 1 for efficiency)
        scale: Overall scale

    Returns:
        C0: Covariance matrix (dense, as numpy array)
        C0_inv: Inverse covariance matrix (dense)
    """
    N = A.shape[0]
    M = tau ** 2 * sparse.eye(N) + A

    # For alpha=1: C0 = scale * M^{-1}
    # For higher alpha, approximate with M^{-alpha} via repeated solve
    M_dense = M.toarray()
    M_inv = np.linalg.inv(M_dense)

    if abs(alpha - 1.0) < 1e-10:
        C0 = scale * M_inv
    else:
        # Use eigendecomposition for fractional powers
        eigvals, eigvecs = np.linalg.eigh(M_dense)
        C0 = scale * (eigvecs * (eigvals ** (-alpha))[None, :]) @ eigvecs.T

    C0_inv = np.linalg.inv(C0)
    return C0, C0_inv


class PDEForwardModel:
    """
    2D Elliptic PDE forward model.

    Solves: -Δp = u on [0,1]² with p = 0 on boundary.
    Forward map: G = A^{-1} (PDE solution operator)
    Observation: O @ G @ u (pointwise measurements)
    """

    def __init__(self, I_grid=33, K=20, gamma=0.0001, lambda_true=0.1, device="cpu"):
        """
        Args:
            I_grid: Grid size (e.g., 2^5 + 1 = 33)
            K: Number of observations
            gamma: Noise level
            lambda_true: True regularization parameter
            device: Torch device
        """
        self.I_grid = I_grid
        self.n_interior = I_grid - 1
        self.N = self.n_interior ** 2
        self.K = K
        self.gamma = gamma
        self.lambda_true = lambda_true
        self.device = device

        # Build discretization
        self.A = create_laplacian_2d(I_grid)
        self.G_solve = sparse.linalg.inv(self.A.tocsc()).toarray()  # PDE solution operator

        # Build covariance prior
        self.C0, self.C0_inv = create_covariance_prior(self.A, tau=0.1, alpha=1.5, scale=10.0)
        self.C0_true = (1.0 / lambda_true) * self.C0

        # All possible sensor locations
        self.number_sensors = self.N
        self.sensors = np.arange(self.N)

        # Initial random sensor weights
        w_init = np.random.randn(self.number_sensors)
        w_softmax = np.exp(-w_init) / np.sum(np.exp(-w_init))
        sorted_indices = np.argsort(-w_softmax)
        self.initial_obs_indices = self.sensors[sorted_indices[:K]]

        # Noise covariance
        self.Gamma = gamma ** 2 * np.eye(K)
        self.Gamma_inv = np.eye(K) / (gamma ** 2)

        # Convert key matrices to torch tensors
        self.G_solve_t = torch.from_numpy(self.G_solve.astype(np.float32)).to(device)
        self.C0_inv_t = torch.from_numpy(self.C0_inv.astype(np.float32)).to(device)
        self.C0_true_t = torch.from_numpy(self.C0_true.astype(np.float32)).to(device)
        self.Gamma_inv_t = torch.from_numpy(self.Gamma_inv.astype(np.float32)).to(device)

    def sample_ground_truth(self, batch_size=1):
        """Sample ground truth from prior N(0, C0_true)."""
        samples = np.random.multivariate_normal(
            np.zeros(self.N), self.C0_true, size=batch_size
        ).astype(np.float32)
        return torch.from_numpy(samples).to(self.device)  # (batch, N)

    def generate_data(self, u, obs_indices=None):
        """
        Generate observations: d = O @ G @ u + noise.

        Args:
            u: Ground truth parameters (batch, N)
            obs_indices: Observation point indices (K,)

        Returns:
            d: Observations (batch, K)
        """
        if obs_indices is None:
            obs_indices = self.initial_obs_indices

        # Forward solve: p = G @ u
        p = u @ self.G_solve_t.T  # (batch, N)

        # Observation: O @ p
        O = self._build_observation_matrix(obs_indices)
        y = p @ O.T  # (batch, K)

        # Add noise
        noise = self.gamma * torch.randn_like(y)
        d = y + noise

        return d

    def _build_observation_matrix(self, obs_indices):
        """Build observation matrix O of shape (K, N)."""
        K = len(obs_indices)
        O = torch.zeros(K, self.N, device=self.device)
        for i, idx in enumerate(obs_indices):
            O[i, idx] = 1.0
        return O

    def solve_lower_level(self, theta_lambda, obs_indices, d):
        """
        Solve the Tikhonov-regularized lower-level problem (exact solution).

        x* = (G'O'Γ⁻¹OG + λC₀⁻¹)⁻¹ G'O'Γ⁻¹d

        Args:
            theta_lambda: log10 of lambda (scalar tensor)
            obs_indices: Observation indices (K,)
            d: Observations (batch, K)

        Returns:
            x_hat: Reconstruction (batch, N)
        """
        lam = 10.0 ** theta_lambda

        O = self._build_observation_matrix(obs_indices)
        G = O @ self.G_solve_t  # (K, N)
        G_t = G.T  # (N, K)

        # System matrix: G'Γ⁻¹G + λC₀⁻¹
        GGammaG = G_t @ self.Gamma_inv_t @ G
        system = GGammaG + lam * self.C0_inv_t

        # RHS: G'Γ⁻¹d
        rhs = (d @ self.Gamma_inv_t @ G).T  # (N, batch)

        # Solve
        x_hat = torch.linalg.solve(system, rhs).T  # (batch, N)

        return x_hat

    def get_obs_indices_from_weights(self, w):
        """
        Convert softmax weights to observation indices.

        Args:
            w: Weight vector (number_sensors,) as tensor

        Returns:
            obs_indices: Top-K indices
        """
        w_softmax = torch.softmax(-w, dim=0)
        _, sorted_indices = torch.sort(w_softmax, descending=True)
        return sorted_indices[:self.K].cpu().numpy()


class PDEBilevelModel(nn.Module):
    """
    Learnable bilevel model for the PDE experiment.

    Parameters to learn:
        - log10_lambda: Tikhonov regularization parameter
        - w: Sensor location weights (softmax policy)
    """

    def __init__(self, pde_model, device="cpu"):
        super().__init__()
        self.pde_model = pde_model

        # Learnable parameters
        self.log10_lambda = nn.Parameter(torch.tensor(0.0))  # log10(lambda)
        self.w = nn.Parameter(torch.randn(pde_model.number_sensors))  # sensor weights

    def forward(self, u_batch, d_batch=None):
        """
        Compute bilevel objective: E[||x*(theta, d) - u||^2]

        Args:
            u_batch: Ground truth (batch, N)
            d_batch: Observations (batch, K) — generated if None

        Returns:
            loss: MSE loss
            x_hat: Reconstruction
        """
        obs_indices = self.pde_model.get_obs_indices_from_weights(self.w.detach())

        if d_batch is None:
            d_batch = self.pde_model.generate_data(u_batch, obs_indices)

        x_hat = self.pde_model.solve_lower_level(
            self.log10_lambda, obs_indices, d_batch
        )

        loss = torch.mean(torch.sum((x_hat - u_batch) ** 2, dim=1))
        return loss, x_hat

    def get_learned_params(self):
        """Return current learned parameters."""
        return {
            "lambda": 10.0 ** self.log10_lambda.item(),
            "log10_lambda": self.log10_lambda.item(),
            "w": self.w.detach().cpu().numpy(),
        }


def setup_pde_experiment(I_grid=33, K=20, gamma=0.0001, lambda_true=0.1, device="cpu"):
    """
    Set up the PDE experiment.

    Args:
        I_grid: Grid size (2^5 + 1 = 33)
        K: Number of observations
        gamma: Noise level
        lambda_true: True regularization parameter
        device: Torch device

    Returns:
        pde_model: PDEForwardModel instance
        bilevel_model: PDEBilevelModel (nn.Module with learnable params)
        config: Experiment configuration dict
    """
    pde_model = PDEForwardModel(
        I_grid=I_grid, K=K, gamma=gamma, lambda_true=lambda_true, device=device
    )
    bilevel_model = PDEBilevelModel(pde_model, device=device).to(device)

    config = {
        "problem": "PDE",
        "I_grid": I_grid,
        "K": K,
        "gamma": gamma,
        "lambda_true": lambda_true,
        "N": pde_model.N,
    }

    return pde_model, bilevel_model, config
