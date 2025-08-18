import torch
from typing import Callable, Tuple, Optional, Union
import inspect
import math

def stochastic_gradient_descent(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    max_iter: int = 200,
    batch_size: Optional[int] = None,
    downsample_ratio: Optional[float] = None,
    BB: bool = False,
    LS: bool = False,
    L_init: float = 1.0,
    tol: float = 1e-4,
    rho: float = 0.9,
    delta: float = 0.1,
    verbose: bool = False,
    shuffle_data: bool = True,
    seed: Optional[int] = None,
):
    """
    Stochastic Gradient Descent with two randomization strategies:
    1. Batch randomization: randomly sample batches from data
    2. Random downsampling: randomly downsample the data at each iteration
    
    Args:
        x0: Initial point
        y: Full dataset/observations
        f: Function to minimize
        nabla: Gradient function
        f_and_nabla: Combined function and gradient computation
        max_iter: Maximum iterations
        batch_size: Size of random batches (strategy 1). If None and downsample_ratio is None, uses full batch
        downsample_ratio: Fraction of data to randomly sample (strategy 2, between 0 and 1)
        BB: Use Barzilai-Borwein step size adaptation
        LS: Use line search
        L_init: Initial inverse step size
        tol: Convergence tolerance
        rho: Line search backtracking factor
        delta: Armijo condition parameter
        verbose: Print progress
        shuffle_data: Whether to shuffle data before batching
        seed: Random seed for reproducibility
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Validate inputs
    if batch_size is not None and downsample_ratio is not None:
        raise ValueError("Cannot use both batch_size and downsample_ratio simultaneously")
    
    if downsample_ratio is not None and (downsample_ratio <= 0 or downsample_ratio > 1):
        raise ValueError("downsample_ratio must be between 0 and 1")
    
    x = x0.clone()
    grad = torch.zeros_like(x)
    grad_old = torch.zeros_like(x)
    x_old = x.clone()
    
    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)
    
    # Determine data size - assuming first dimension is batch/data dimension
    data_size = y.shape[0] if len(y.shape) > 0 else 1
    
    # Setup for batch processing
    if batch_size is not None:
        # Strategy 1: Batch randomization
        n_batches = math.ceil(data_size / batch_size)
        if shuffle_data:
            indices = torch.randperm(data_size, device=y.device)
        else:
            indices = torch.arange(data_size, device=y.device)
    elif downsample_ratio is not None:
        # Strategy 2: Random downsampling
        sample_size = max(1, int(data_size * downsample_ratio))
    else:
        # Full batch (standard GD)
        batch_size = data_size
        n_batches = 1
    
    batch_idx = 0
    epoch = 0
    indices = torch.arange(data_size, device=y.device)
    for i in range(max_iter):
        x_old.copy_(x)
        grad_old.copy_(grad)
        batch_indices = None
        # Select data subset based on strategy
        print("batch size:", batch_size )
        if batch_size is not None:
            # Strategy 1: Batch randomization
            if batch_idx == 0 and shuffle_data and i > 0:
                # Reshuffle at the beginning of each epoch
                indices = torch.randperm(data_size, device=y.device)
            
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, data_size)
            batch_indices = indices[start_idx:end_idx]
            y_batch = y[batch_indices]
            
            # Update batch index
            batch_idx = (batch_idx + 1) % n_batches
            if batch_idx == 0:
                epoch += 1
                
        elif downsample_ratio is not None:
            # Strategy 2: Random downsampling
            sample_indices = torch.randperm(data_size)[:sample_size]
            y_batch = y[sample_indices]
        else:
            # Full batch
            y_batch = y
        
        # Compute energy and gradient on selected data
        energy, grad[batch_indices] = f_and_nabla(x[batch_indices], y_batch)

        # BB1 update for step size adaptation
        if BB and i > 0:
            dx = x[batch_indices] - x_old[batch_indices]
            dg = grad[batch_indices] - grad_old[batch_indices]
            s_dot_y = (dx * dg).sum()
            s_dot_s = (dx * dx).sum()
            if s_dot_s > 0:
                L = torch.clamp(s_dot_y / s_dot_s, min=1e-6, max=1e6)
        
        # Line search (optional for SGD, but can help with convergence)
        step_accepted = False
        x_new = x.clone()
        for ls_iter in range(50 if LS else 1):
            step = grad / L
            x_new[batch_indices] = x[batch_indices] - step

            if LS:
                energy_new = f(x_new[batch_indices], y_batch)
                # Armijo condition
                armijo = energy_new <= energy - delta * (step * step).view(step.shape[0], -1).sum(dim=1)
                if torch.all(armijo):
                    x.copy_(x_new)
                    step_accepted = True
                    break
                else:
                    L /= rho
            else:
                x.copy_(x_new)
                step_accepted = True
                break
        
        if not step_accepted:
            if verbose:
                print(f"Line search failed at iter {i}")
            break
        
        # Convergence check (based on parameter change)
        rel_change = torch.norm(x - x_old) / torch.norm(x).clamp(min=1e-8)
        if rel_change < tol:
            if verbose:
                print(f"Converged at iter {i}, rel change: {rel_change:.2e}, epoch: {epoch}")
            break
        
        if verbose and i % 10 == 0:
            print(f"Iter {i}: energy={energy.item():.4e}, rel_change={rel_change:.2e}, epoch: {epoch}")
            strategy_info = ""
            if batch_size is not None and batch_size < data_size:
                strategy_info = f", batch: {len(y_batch)}/{data_size}, epoch: {epoch}"
            elif downsample_ratio is not None:
                strategy_info = f", sampled: {len(y_batch)}/{data_size} ({downsample_ratio:.1%})"
            
            # Handle energy display - ensure scalar for logging
            energy_val = energy.item() if energy.numel() == 1 else energy.mean().item()
            print(f"Iter {i}: energy={energy_val:.4e}, rel_change={rel_change:.2e}{strategy_info}")
    
    # if verbose:
    #     print(f"Finished at iter {i}, rel change: {rel_change:.2e}")
    
    return x, L, i


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
# def reconstruct_sgd(
#     y,
#     physics,
#     data_fidelity,
#     regularizer,
#     lamda,
#     step_size,
#     max_iter,
#     tol,
#     x_init=None,
#     batch_size=None,
#     downsample_ratio=None,
#     detach_grads=True,
#     verbose=False,
#     return_stats=False,
#     BB=False,
#     LS=False,
#     shuffle_data=True,
#     seed=None,
# ):
#     """
#     Wrapper for stochastic gradient descent reconstruction.
    
#     Args:
#         y: Observations (batch dimension first)
#         physics: Forward model
#         data_fidelity: Data fidelity term
#         regularizer: Regularization term
#         lamda: Regularization parameter
#         step_size: Initial step size
#         max_iter: Maximum iterations
#         tol: Convergence tolerance
#         x_init: Initial guess (optional)
#         batch_size: Size of random batches (strategy 1)
#         downsample_ratio: Fraction of data to randomly sample (strategy 2)
#         detach_grads: Detach gradients for memory efficiency
#         verbose: Print convergence info
#         return_stats: Return additional statistics
#         BB: Use Barzilai-Borwein step size
#         LS: Use line search
#         shuffle_data: Whether to shuffle data before batching
#         seed: Random seed for reproducibility
    
#     Returns:
#         rec: Reconstructed solution
#         stats (optional): Dictionary with statistics
#     """
    
#     if x_init is not None:
#         x = torch.clone(x_init).detach()
#     else:
#         # Always initialize with full data to get correct dimensions
#         x = physics.A_dagger(y)
    
    
#     def energy(val, y_batch):
#         with torch.no_grad():
#             fun = data_fidelity(val, y_batch, physics) + lamda * regularizer.g(val)
#         if detach_grads:
#             fun = fun.detach()
#         # Ensure scalar output for optimization
#         if fun.numel() > 1:
#             return fun.mean()
#         return fun
    
#     def energy_grad(val, y_batch):
#         grad = data_fidelity.grad(val, y_batch, physics) + lamda * regularizer.grad(val)
#         if detach_grads:
#             grad = grad.detach()
#         # Ensure grad has same shape as val
#         if grad.shape != val.shape:
#             if verbose:
#                 print(f"Warning: grad shape {grad.shape} != val shape {val.shape}")
#         return grad
    
#     # Check if regularizer supports combined energy and gradient computation
#     signature = inspect.signature(regularizer.grad)
#     argument_names = [param.name for param in signature.parameters.values()]
    
#     if "get_energy" in argument_names:
#         def energy_and_grad(val, y_batch):
#             reg_fun, reg_grad = regularizer.grad(val, get_energy=True)
#             data_fun = data_fidelity(val, y_batch, physics)
#             data_grad = data_fidelity.grad(val, y_batch, physics)
            
#             fun = data_fun + lamda * reg_fun
#             grad = data_grad + lamda * reg_grad
            
#             if detach_grads:
#                 fun = fun.detach()
#                 grad = grad.detach()
#             return fun, grad
#     else:
#         energy_and_grad = lambda val, y_batch: (energy(val, y_batch), energy_grad(val, y_batch))
    
#     # Run stochastic gradient descent

    
#     rec, L, steps = stochastic_gradient_descent(
#         x0=x,
#         y=y,
#         max_iter=max_iter,
#         f=energy,
#         nabla=energy_grad,
#         f_and_nabla=energy_and_grad,
#         batch_size=batch_size,
#         downsample_ratio=downsample_ratio,
#         L_init=1 / step_size,
#         tol=tol,
#         verbose=verbose,
#         BB=BB,
#         LS=LS,
#         shuffle_data=shuffle_data,
#         seed=seed,
#     )
    
    
#     stats = dict(L=L.detach(), steps=steps)
#     if return_stats:
#         return rec, stats
#     return rec, steps

def reconstruct_sgd(
    y,
    physics,
    data_fidelity,
    regularizer,
    lamda,
    step_size,
    max_iter,
    tol,
    patch_size=(32, 32),
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

# Example usage functions
# def example_batch_sgd():
#     """Example of using SGD with batch randomization"""
#     # Assuming you have y, physics, data_fidelity, regularizer defined
    
#     rec, stats = reconstruct_sgd(
#         y=y,  # Your observation data
#         physics=physics,
#         data_fidelity=data_fidelity,
#         regularizer=regularizer,
#         lamda=0.01,
#         step_size=0.1,
#         max_iter=1000,
#         tol=1e-4,
#         batch_size=32,  # Use batches of size 32
#         verbose=True,
#         return_stats=True,
#         shuffle_data=True,
#         seed=42
#     )
#     return rec, stats


# def example_downsample_sgd():
#     """Example of using SGD with random downsampling"""
    
#     rec, stats = reconstruct_sgd(
#         y=y,  # Your observation data
#         physics=physics,
#         data_fidelity=data_fidelity,
#         regularizer=regularizer,
#         lamda=0.01,
#         step_size=0.1,
#         max_iter=1000,
#         tol=1e-4,
#         downsample_ratio=0.1,  # Use 10% of data at each iteration
#         verbose=True,
#         return_stats=True,
#         seed=42
#     )
#     return rec, stats