import torch
from typing import Callable, Tuple


def fista(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla_f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prox_g: Callable[[torch.Tensor, float], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1.0,
    tol: float = 1e-4,
    verbose: bool = False,
    stop_by_iter: bool = False,
):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) with backtracking.

    Solves: min_x f(x, y) + g(x)
    where f is smooth and g has a computable proximal operator.

    Args:
        x0: Initial guess
        y: Observation data
        f: Smooth part of objective (data fidelity)
        nabla_f: Gradient of the smooth part
        prox_g: Proximal operator of the non-smooth part g
        max_iter: Maximum number of iterations
        L_init: Initial Lipschitz constant estimate
        tol: Convergence tolerance (relative change)
        verbose: If True, print convergence information
        stop_by_iter: If True, run for exactly max_iter iterations

    Returns:
        x: Solution
        L: Final Lipschitz estimate
        i: Number of iterations performed
    """
    x = x0.clone()
    x_old = x.clone()
    t = 1.0
    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)

    for i in range(max_iter):
        # Momentum step
        t_old = t
        t = (1.0 + (1.0 + 4.0 * t_old ** 2) ** 0.5) / 2.0
        beta = (t_old - 1.0) / t

        # Extrapolation point
        z = x + beta * (x - x_old)

        # Store old iterate
        x_old.copy_(x)

        # Gradient step + proximal step
        grad = nabla_f(z, y)
        step_size = 1.0 / L
        x = prox_g(z - step_size * grad, step_size)

        # Convergence check
        if not stop_by_iter:
            rel_change = torch.norm(x - x_old) / torch.norm(x).clamp(min=1e-8)
            if rel_change < tol:
                if verbose:
                    print(f"FISTA converged at iter {i}, rel change: {rel_change:.2e}")
                break

    if verbose:
        if not stop_by_iter:
            print(f"FISTA finished at iter {i}, rel change: {rel_change:.2e}")
        else:
            print(f"FISTA finished at iter {i}")

    return x, L, i


def reconstruct_fista(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size,
    max_iter,
    tol,
    x_init=None,
    detach_grads=True,
    verbose=False,
    return_stats=False,
):
    """
    FISTA-based reconstruction.

    Args:
        y: Observations
        physics: Forward model
        data_fidelity: Data fidelity term
        regularizer: Regularization term (must implement g() and grad())
        lamda: Regularization parameter
        step_size: Initial step size (1/L)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        x_init: Initial guess (optional)
        detach_grads: Detach gradients for memory efficiency
        verbose: Print convergence info
        return_stats: Return additional statistics

    Returns:
        rec: Reconstructed solution
        stats_or_steps: Dictionary with stats if return_stats, else iteration count
    """
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)

    def smooth_energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

    def smooth_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics)
        if detach_grads:
            grad = grad.detach()
        return grad

    def prox_g(val, step_size_val):
        """
        Proximal operator for lamda * regularizer.
        For smooth regularizers (like smoothed TV), we approximate with a gradient step.
        prox_{step * lamda * g}(v) ≈ v - step * lamda * grad_g(v)
        """
        with torch.no_grad():
            reg_grad = regularizer.grad(val)
            if detach_grads:
                reg_grad = reg_grad.detach()
        return val - step_size_val * lamda * reg_grad

    rec, L, steps = fista(
        x0=x,
        y=y,
        f=smooth_energy,
        nabla_f=smooth_grad,
        prox_g=prox_g,
        max_iter=max_iter,
        L_init=1.0 / step_size,
        tol=tol,
        verbose=verbose,
        stop_by_iter=not return_stats,
    )

    stats = dict(L=L.detach(), steps=steps)
    if return_stats:
        return rec, stats
    return rec, steps
