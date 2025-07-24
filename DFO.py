import torch
from GD import reconstruct_gd
import numpy as np
import torch.nn as nn
from deepinv.loss.metric import PSNR

def sample_noise_like(model):
    return {name: torch.randn_like(param).detach() for name, param in model.named_parameters()}

def apply_perturbation(model, noise, sigma=1e-2):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(sigma * noise[name])
def apply_negative_perturbation(model, noise, sigma=1e-2):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.sub_(sigma * noise[name])
                       
def set_parameters(model, reference_params):
    with torch.no_grad():
        for (name, param), (_, ref_param) in zip(model.named_parameters(), reference_params.items()):
            param.copy_(ref_param)
@torch.no_grad()
def eval_f(model, physics, data_fidelity, gt, lmbd, x_init=None, max_iter=10):
    """
    Evaluate f(θ) = ||x̂ - x*|| where x̂ = argmax_x φ(x, θ)
    """
    # Generate observed data y from ground truth
    y = physics(gt)

    # Solve the variational problem (i.e., reconstruct image x̂)
    if x_init is None:
        x_init = y.clone()
    x_hat, iterations = reconstruct_gd(
        y=y,
        x_init=x_init,
        physics=physics,
        data_fidelity=data_fidelity,
        regularizer=model,  # this is θ
        lamda=lmbd,
        step_size=1.0,
        max_iter=max_iter,
        tol=1e-2,
        detach_grads=True,
        verbose=False,
        return_stats=False,
        BB=True,
        LS=True,
    )
    # Compute distance to ground truth
    return torch.mean(
        torch.norm(x_hat - gt, p=2, dim=(1, 2, 3))**2
    ), x_hat , iterations
def gaussian_smoothing_step(
    model: nn.Module,
    physics,
    data_fidelity,
    gt,
    lmbd,
    optim_algo="SGD",
    normalize_gradients=True,
    warm_start=True,
    sigma=1e-1,
    N=20,
    lr=1e-1,
    verbose=True,
    device="cpu",
):
    if optim_algo == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_algo == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    psnr = PSNR()
    # Store original parameters
    original_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Initialize empty gradient dict
    grad_est = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    
    x_hat = None # To store the last x_hat for negative perturbation
    x_hat_pos = None # To store the last x_hat for positive perturbation
    loss = []
    psnr_values = []
    iterations = 0
    for _ in range(N):
        noise = sample_noise_like(model)
        # 1. θ + σu
        apply_perturbation(model, noise, sigma)
        f_pos, out_pos , iterations_pos = eval_f(model=model, physics=physics, data_fidelity=data_fidelity, gt=gt, lmbd=lmbd, x_init=x_hat_pos) # x_init can be reset for each perturbation or carried over

        # Reset to θ
        set_parameters(model, original_params)

        # 2. θ
        # Evaluate f(θ)
        f, out, iterations_ = eval_f(model=model, physics=physics, data_fidelity=data_fidelity, gt=gt, lmbd=lmbd, x_init=x_hat) # x_init can be reset for each perturbation or carried over
        iterations += iterations_ + iterations_pos
        if warm_start:
            # Use the output from the positive perturbation as the new x_init for the next iteration
            x_hat_pos = out_pos
            x_hat = out
        # Reset to θ
        set_parameters(model, original_params)

        # Accumulate gradients: (f(θ + σu) - f(θ)) * u
        for name in grad_est:
            grad_est[name] += (f_pos - f) * noise[name]
        loss.append(f.item())
        psnr_values.append(psnr(gt.to(device), out).mean().item())

    if verbose:
        print(f"Loss: {np.mean(loss)}, PSNR: {np.mean(psnr_values)}, Cost: {iterations}")
    optimizer.zero_grad()  # Clear previous gradients
    # Final gradient estimate
    for name in grad_est:
        grad_est[name] /= (N * sigma) # Divide by N*sigma due to the perturbation scaling

    # Gradient descent step: θ ← θ - lr * grad
    with torch.no_grad():
        # normalize gradients to prevent large updates
        norm_grad = torch.sqrt(sum((g**2).sum() for g in grad_est.values()))
        for name, param in model.named_parameters():
            param.grad = grad_est[name]  # Set the gradient for the upper-level optimizer
            if normalize_gradients:
                param.grad /= (norm_grad + 1e-8)  # Normalize the gradient
    optimizer.step()  # Perform the optimization step
    return np.mean(loss), np.mean(psnr_values), iterations