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
    Basic Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

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


def fista_with_restart(
    x0: torch.Tensor,
    y: torch.Tensor,
    nabla_f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    prox_g: Callable[[torch.Tensor, float], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1.0,
    tol: float = 1e-4,
    verbose: bool = False,
    stop_by_iter: bool = False,
):
    """
    Fixed-step FISTA with adaptive restart.

    This keeps the old reconstruction dynamics, but resets acceleration when
    the momentum direction starts fighting the accepted update.
    """
    x = x0.clone()
    x_old = x.clone()
    t = 1.0
    L = torch.tensor(L_init, dtype=x.dtype, device=x.device)
    rel_change = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    restarts = 0

    for i in range(max_iter):
        t_old = t
        t = (1.0 + (1.0 + 4.0 * t_old ** 2) ** 0.5) / 2.0
        beta = (t_old - 1.0) / t
        z = x + beta * (x - x_old)

        grad = nabla_f(z, y)
        step_size = 1.0 / L
        x_new = prox_g(z - step_size * grad, step_size)

        restart_score = ((z - x_new).reshape(x.shape[0], -1) * (x_new - x).reshape(x.shape[0], -1)).sum()
        if restart_score > 0:
            restarts += 1
            t = 1.0
            z = x
            grad = nabla_f(z, y)
            x_new = prox_g(z - step_size * grad, step_size)

        rel_change = torch.norm(x_new - x) / torch.norm(x_new).clamp(min=1e-8)
        x_old = x
        x = x_new

        if not stop_by_iter and rel_change < tol:
            if verbose:
                print(f"FISTA converged at iter {i}, rel change: {rel_change:.2e}, restarts: {restarts}")
            break

    if verbose:
        if not stop_by_iter:
            print(f"FISTA finished at iter {i}, rel change: {rel_change:.2e}, restarts: {restarts}")
        else:
            print(f"FISTA finished at iter {i}, restarts: {restarts}")

    stats = {
        "L": L.detach(),
        "steps": i,
        "rel_change": rel_change.detach(),
        "restarts": restarts,
        "backtracks": 0,
        "converged": (not stop_by_iter and rel_change < tol),
    }
    return x, L, i, stats


def _batch_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).sum(dim=1)


def accelerated_gradient_backtracking(
    x0: torch.Tensor,
    y: torch.Tensor,
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gradient: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    max_iter: int = 200,
    L_init: float = 1.0,
    tol: float = 1e-4,
    rho: float = 0.5,
    max_backtracking: int = 25,
    restart: bool = True,
    monotone: bool = True,
    verbose: bool = False,
    stop_by_iter: bool = False,
):
    """
    Monotone accelerated gradient with Armijo-style backtracking and restart.

    This is the safer lower-level solver for the CT regularizers in this repo:
    the TV and FoE priors are smooth, but their exact proximal maps are not used.
    Treating the full lower objective as smooth avoids the pseudo-proximal
    FISTA instability that can otherwise create overly blurred validation images.
    """
    x = x0.clone()
    z = x.clone()
    t = 1.0
    L = torch.tensor(max(float(L_init), 1e-12), dtype=x.dtype, device=x.device)
    energy_x = objective(x, y)

    rel_change = torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    total_backtracks = 0
    total_restarts = 0
    converged = False

    def propose(point, L_start):
        grad_point = gradient(point, y)
        energy_point = objective(point, y)
        L_trial = L_start.clone()
        x_trial = point
        energy_trial = energy_point
        local_backtracks = 0

        for bt in range(max_backtracking + 1):
            step = 1.0 / L_trial
            x_trial = point - step * grad_point
            energy_trial = objective(x_trial, y)

            diff = x_trial - point
            model = (
                energy_point
                + _batch_inner(grad_point, diff)
                + 0.5 * L_trial * _batch_inner(diff, diff)
            )
            slack = 1e-7 * (1.0 + energy_point.abs())
            if (energy_trial <= model + slack).all():
                local_backtracks = bt
                break

            L_trial = L_trial / rho
            local_backtracks = bt + 1

        return x_trial, energy_trial, L_trial, local_backtracks

    for i in range(max_iter):
        x_prev = x.clone()
        z_prev = z.clone()

        x_candidate, energy_candidate, L, n_backtracks = propose(z, L)
        total_backtracks += n_backtracks

        if monotone:
            slack = 1e-7 * (1.0 + energy_x.abs())
            if (energy_candidate > energy_x + slack).any():
                total_restarts += 1
                t = 1.0
                z_prev = x.clone()
                x_candidate, energy_candidate, L, n_backtracks = propose(x, L)
                total_backtracks += n_backtracks

        rel_change = torch.norm(x_candidate - x_prev) / torch.norm(x_candidate).clamp(min=1e-8)
        x = x_candidate
        energy_x = energy_candidate

        restart_momentum = False
        if restart:
            # Gradient restart: positive alignment means the momentum is fighting
            # the accepted descent direction.
            restart_score = _batch_inner(z_prev - x, x - x_prev).sum()
            restart_momentum = restart_score > 0

        if restart_momentum:
            total_restarts += 1
            t = 1.0
            z = x.clone()
        else:
            t_old = t
            t = (1.0 + (1.0 + 4.0 * t_old ** 2) ** 0.5) / 2.0
            beta = (t_old - 1.0) / t
            z = x + beta * (x - x_prev)

        if not stop_by_iter and rel_change < tol:
            converged = True
            if verbose:
                print(f"FISTA converged at iter {i}, rel change: {rel_change:.2e}, L: {L.item():.2e}")
            break

    if verbose:
        status = "converged" if converged else "finished"
        print(
            f"FISTA {status} at iter {i}, rel change: {rel_change:.2e}, "
            f"L: {L.item():.2e}, restarts: {total_restarts}, backtracks: {total_backtracks}"
        )

    stats = {
        "L": L.detach(),
        "steps": i,
        "rel_change": rel_change.detach(),
        "final_energy": energy_x.detach(),
        "restarts": total_restarts,
        "backtracks": total_backtracks,
        "converged": converged,
    }
    return x, L, i, stats


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
    backtracking=True,
    restart=True,
    monotone=True,
    mode="legacy",
    stop_by_iter=True,
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

    def full_energy(val, y_in):
        with torch.no_grad():
            fun = data_fidelity(val, y_in, physics) + lamda * regularizer.g(val)
        if detach_grads:
            fun = fun.detach()
        return fun.reshape(-1)

    def full_grad(val, y_in):
        grad = data_fidelity.grad(val, y_in, physics) + lamda * regularizer.grad(val)
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

    if mode == "monotone":
        rec, L, steps, solver_stats = accelerated_gradient_backtracking(
            x0=x,
            y=y,
            objective=full_energy,
            gradient=full_grad,
            max_iter=max_iter,
            L_init=1.0 / step_size,
            tol=tol,
            verbose=verbose,
            stop_by_iter=stop_by_iter,
            restart=restart,
            monotone=monotone,
        )
    elif mode == "restart":
        rec, L, steps, solver_stats = fista_with_restart(
            x0=x,
            y=y,
            nabla_f=lambda val, y_in: data_fidelity.grad(val, y_in, physics),
            prox_g=prox_g,
            max_iter=max_iter,
            L_init=1.0 / step_size,
            tol=tol,
            verbose=verbose,
            stop_by_iter=stop_by_iter,
        )
    else:
        rec, L, steps = fista(
            x0=x,
            y=y,
            f=lambda val, y_in: data_fidelity(val, y_in, physics).reshape(-1),
            nabla_f=lambda val, y_in: data_fidelity.grad(val, y_in, physics),
            prox_g=prox_g,
            max_iter=max_iter,
            L_init=1.0 / step_size,
            tol=tol,
            verbose=verbose,
            stop_by_iter=stop_by_iter,
        )
        solver_stats = {"L": L.detach(), "steps": steps}

    stats = dict(solver_stats)
    stats["L"] = L.detach()
    stats["steps"] = steps
    if return_stats:
        return rec, stats
    return rec, steps
