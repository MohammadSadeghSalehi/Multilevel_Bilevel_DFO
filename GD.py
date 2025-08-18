import torch
from typing import Callable, Tuple
import inspect

def gradient_descent(
    x0: torch.Tensor,
    y: torch.Tensor,
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    nabla: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_and_nabla: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    max_iter: int = 200,
    BB: bool = False,
    LS: bool = True,
    L_init: float = 1.0,
    tol: float = 1e-4,
    rho: float = 0.5,
    delta: float = 0.0001,
    verbose: bool = False,
    stop_by_iter: bool = False
):
    """
    Gradient Descent with optional Barzilai-Borwein step and Armijo Line Search.
    """
    x = x0.clone()
    grad = torch.zeros_like(x)
    grad_old = torch.zeros_like(x)
    x_old = x.clone()

    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)

    for i in range(max_iter):
        x_old.copy_(x)
        grad_old.copy_(grad)

        energy, grad = f_and_nabla(x, y)

        # BB1 update
        if BB and i > 0:
            dx = x - x_old
            dg = grad - grad_old
            s_dot_y = (dx * dg).sum()
            s_dot_s = (dx * dx).sum()
            if s_dot_s > 0:
                L = torch.clamp(s_dot_y / s_dot_s, min=1e-6, max=1e6)
        # Line search
        step_accepted = False
        for _ in range(50 if LS else 1):
            step = grad / L
            x_new = x - step
            energy_new = f(x_new, y)

            # Armijo condition (assuming scalar energies)
            armijo = energy_new <= energy - delta * step.norm()**2

            if not LS or armijo.all():
                x.copy_(x_new)
                step_accepted = True
                break
            else:
                L /= rho

        if not step_accepted:
            if verbose:
                print(f"Line search failed at iter {i}, falling back to fixed step size.")
            # Fallback: take a fixed small step in the negative gradient direction
            step = grad / 1e7
            x.sub_(step)
            # Optionally recompute energy if needed, but continue optimization

        # Convergence check
        if not stop_by_iter:
            rel_change = torch.norm(x - x_old) / torch.norm(x).clamp(min=1e-8)
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iter {i}, rel change: {rel_change:.2e}")
                break

    if verbose:
        if not stop_by_iter:
            print(f"Finished at iter {i}, rel change: {rel_change:.2e}")
        else:
            print(f"Finished at iter {i}")

    return x, L, i


def reconstruct_gd(
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
    BB=False,
    LS=True,
):
    """
    Wrapper for gradient descent reconstruction.
    
    Args:
        y: Observations
        physics: Forward model
        data_fidelity: Data fidelity term
        regularizer: Regularization term
        lamda: Regularization parameter
        step_size: Initial step size
        max_iter: Maximum iterations
        tol: Convergence tolerance
        x_init: Initial guess (optional)
        detach_grads: Detach gradients for memory efficiency
        verbose: Print convergence info
        return_stats: Return additional statistics
        BB: Use Barzilai-Borwein step size
        LS: Use line search
    
    Returns:
        rec: Reconstructed solution
        stats (optional): Dictionary with statistics
    """
    
    if x_init is not None:
        x = torch.clone(x_init).detach()
    else:
        x = physics.A_dagger(y)
    
    def energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)
    
    def energy_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val)
        if detach_grads:
            grad = grad.detach()
        return grad
    
    # Check if regularizer supports combined energy and gradient computation
    signature = inspect.signature(regularizer.grad)
    argument_names = [param.name for param in signature.parameters.values()]
    
    if "get_energy" in argument_names:
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
    else:
        energy_and_grad = lambda val, y_in: (energy(val, y_in), energy_grad(val, y_in))
    
    # Run gradient descent
    rec, L, steps = gradient_descent(
        x0=x,
        y=y,
        max_iter=max_iter,
        f=energy,
        nabla=energy_grad,
        f_and_nabla=energy_and_grad,
        L_init=1 / step_size,
        tol=tol,
        verbose=verbose,
        BB=BB,
        LS=LS,
    )
    
    stats = dict(L=L.detach(), steps=steps)
    if return_stats:
        return rec, stats
    return rec, steps
