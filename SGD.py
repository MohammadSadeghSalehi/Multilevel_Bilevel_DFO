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
    step_decay: str = "none",
    patch_mode: str = "patch",
    mask_probability: float = 0.25,
):
    """
    SGD with spatial subsampling: operates on random spatial crops of (B, C, W, H).
    """
    x = x0.clone()
    B, C, W, H = x.shape
    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)
    patch_mode = patch_mode.lower()
    if patch_mode not in {"patch", "full", "mask"}:
        raise ValueError(f"Unknown SGD patch mode: {patch_mode}")

    for i in range(max_iter):
        if patch_mode == "full":
            patch_slice = (slice(None), slice(None), slice(None), slice(None))
            x_sub = x
            y_sub = y
        elif patch_mode == "mask":
            patch_slice = None
            x_sub = x
            y_sub = y
        else:
            # Sample random crop coordinates. Random crops are cheap, but they
            # update boundary pixels less frequently than interior pixels.
            w_patch, h_patch = patch_size
            w_patch = min(max(1, int(w_patch)), W)
            h_patch = min(max(1, int(h_patch)), H)
            x0_crop = torch.randint(0, W - w_patch + 1, (1,)).item()
            y0_crop = torch.randint(0, H - h_patch + 1, (1,)).item()
            patch_slice = (
                slice(None),
                slice(None),
                slice(x0_crop, x0_crop + w_patch),
                slice(y0_crop, y0_crop + h_patch),
            )
            x_sub = x[patch_slice].clone()
            y_sub = y[patch_slice].clone()

        # Compute loss and gradient on the selected stochastic subproblem.
        energy, grad = f_and_nabla(x_sub, y_sub)
        if detach_grads:
            grad = grad.detach()

        # Take a stochastic gradient step only on the sampled patch. A decay is
        # useful for FoE priors, where a fixed patch step can keep denoising
        # past the PSNR optimum and drift into over-regularization.
        if step_decay == "sqrt":
            effective_L = L * torch.sqrt(torch.tensor(float(i + 1), dtype=x.dtype, device=x.device))
        elif step_decay == "linear":
            effective_L = L * float(i + 1)
        elif step_decay == "none":
            effective_L = L
        else:
            raise ValueError(f"Unknown SGD step decay: {step_decay}")
        step = grad / effective_L
        if patch_mode == "mask":
            keep_prob = min(max(float(mask_probability), 1e-4), 1.0)
            mask = (torch.rand((B, 1, W, H), dtype=x.dtype, device=x.device) < keep_prob).to(x.dtype)
            # Coordinate-style stochastic update. We deliberately do not divide
            # by keep_prob: the lower-step parameter controls the stable scale.
            step = step * mask
            x = x - step
        elif patch_mode == "full":
            x = x_sub - step
        else:
            x_patch = x.clone()
            x_patch[patch_slice] = x_sub - step
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
    step_decay="none",
    patch_mode="patch",
    mask_probability=0.25,
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
        step_decay=step_decay,
        patch_mode=patch_mode,
        mask_probability=mask_probability,
    )

    stats = steps #dict(L=L.detach(), steps=steps)
    return (rec, stats) if return_stats else (rec, steps)
