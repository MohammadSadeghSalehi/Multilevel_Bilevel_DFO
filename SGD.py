import torch
from typing import Callable, Tuple

def gradient_descent_sgd(
    x0: torch.Tensor,
    y: torch.Tensor,
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    max_iter: int = 200,
    patch_size: Tuple[int, int] = (32, 32),
    L_init: float = 1.0,
    tol: float = 1e-4,
    verbose: bool = False,
    stop_by_iter: bool = False,
    detach_grads: bool = True,
):
    """
    SGD with spatial subsampling: operates on random spatial crops of (B, C, W, H).
    """
    x = x0.clone()
    B, C, W, H = x.shape
    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)

    for i in range(max_iter):
        x_patch = x.clone()  # Full x will be modified locally

        # Sample random crop coordinates
        w_patch, h_patch = patch_size
        x0_crop = torch.randint(0, W - w_patch + 1, (1,)).item()
        y0_crop = torch.randint(0, H - h_patch + 1, (1,)).item()

        # Get spatial patch
        patch_slice = (slice(None), slice(None), slice(x0_crop, x0_crop + w_patch), slice(y0_crop, y0_crop + h_patch))
        x_sub = x[patch_slice].clone()
        y_sub = y[patch_slice].clone()

        # Compute local loss and gradient
        energy, grad = f_and_nabla(x_sub, y_sub)
        if detach_grads:
            grad = grad.detach()

        # Take gradient step only on patch
        step = grad / L
        x_patch[patch_slice] = x_sub - step

        # Insert updated patch into full x
        x = x_patch
        rel_change = torch.norm(step) / torch.norm(x_sub).clamp(min=1e-8)

        if i % 50 ==0 and verbose:
            print(f"Iter {i}: energy={energy.mean():.4e}, rel_change={rel_change:.2e}")
        if not stop_by_iter:
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iter {i}, rel change: {rel_change:.2e}")
                break

    return x, L, i

def reconstruct_sgd(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size,
    max_iter,
    tol,
    patch_size=(16, 16),
    x_init=None,
    detach_grads=True,
    verbose=True,
    return_stats=False,
):
    """
    SGD-based reconstruction using spatial patches.
    """

    if x_init is not None:
        x = x_init.clone().detach()
    else:
        x = physics.A_dagger(y)

    def energy_and_grad(val, y_in):
        reg_fun, reg_grad = regularizer.grad(val, get_energy=True)
        data_fun = data_fidelity(val, y_in, physics)
        data_grad = data_fidelity.grad(val, y_in, physics)

        fun = data_fun + lamda * reg_fun
        grad = data_grad + lamda * reg_grad

        if detach_grads:
            fun = fun.detach()
            grad = grad.detach()
        return fun, grad

    rec, L, steps = gradient_descent_sgd(
        x0=x,
        y=y,
        f_and_nabla=energy_and_grad,
        max_iter=max_iter,
        patch_size=patch_size,
        L_init=1 / step_size,
        tol=tol,
        verbose=verbose,
        stop_by_iter=not return_stats,
        detach_grads=detach_grads,
    )

    stats = steps #dict(L=L.detach(), steps=steps)
    return (rec, stats) if return_stats else (rec, steps)
