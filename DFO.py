import torch
import numpy as np
from deepinv.loss.metric import PSNR
from GD import reconstruct_gd
from SGD import reconstruct_sgd
from FISTA import reconstruct_fista
from typing import List
import matplotlib.pyplot as plt
np.random.seed(0)
class MultilevelSmoothingOptimizer:
    def __init__(
        self,
        model: torch.nn.Module,
        physics,
        data_fidelity,
        ground_truth,
        lmbd,
        lr=1e-2,
        optimizer_type="SGD",
        device="cpu",
        lower_alg="GD",
        lower_lr=1e-2,
        lower_tol=1e-1
    ):
        self.model = model.to(device)
        self.physics = physics
        self.data_fidelity = data_fidelity
        self.gt = ground_truth.to(device)
        self.gt_slice = ground_truth.to(device)
        self.slice = None  # Placeholder for lower-level indices given each level
        self.lmbd = lmbd
        self.device = device
        self.psnr_metric = PSNR()
        self.optimizer = self._build_optimizer(optimizer_type, lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200,   # number of iterations/epochs for one cosine cycle
            eta_min=0.0  # minimum learning rate (default 0)
        )
        self.alg = lower_alg
        self.lower_lr = lower_lr
        self.eps = lower_tol

    def _build_optimizer(self, opt_type, lr):
        if opt_type == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def sample_noise(self, normalize=True):
        param_shapes = [p.shape for p in self.model.parameters()]
        total_size = sum(p.numel() for p in self.model.parameters())
        noise_vec = torch.randn(total_size, device=self.device)
        if normalize:
            noise_vec /= (torch.norm(noise_vec) + 1e-8)

        noise_dict = {}
        idx = 0
        for (name, param), shape in zip(self.model.named_parameters(), param_shapes):
            size = param.numel()
            noise_dict[name] = noise_vec[idx:idx+size].reshape(shape)
            idx += size
        return noise_dict

    def apply_perturbation(self, noise, sigma, sign=1):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.add_(sign * sigma * noise[name])

    def set_parameters(self, param_dict):
        with torch.no_grad():
            for (name, param), (_, ref_param) in zip(self.model.named_parameters(), param_dict.items()):
                param.copy_(ref_param)

    @torch.no_grad()
    def evaluate_objective(self, x_init=None, N_l= None, alg = "GD", max_iter=1000, lower_lr = 1e-2):
        y = self.physics(self.gt)
        random_indices = None
        x_hat = torch.zeros_like(self.physics.A_dagger(y))
        if N_l is None:
            N_l = y.shape[0]
            self.gt_slice = self.gt
        else:
            random_indices = self.slice
            y = y[random_indices]
            self.gt_slice = self.gt[random_indices]
        
        # Ensure x_init is on the correct device if provided
        if x_init is not None:
             x_init = x_init.to(self.device)
        
        if alg == "GD":
            x_hat, iters = reconstruct_gd(
                y=y,
                x_init=x_init,
                physics=self.physics,
                data_fidelity=self.data_fidelity,
                regularizer=self.model,
                lamda=self.lmbd,
                step_size=lower_lr,
                max_iter=max_iter,
                tol=self.eps,
                detach_grads=True,
                verbose=False,
                return_stats=False,
                BB=False,
                LS=True,
            )
        elif alg == "SGD":
            patch_size = (y.shape[2]//2, y.shape[3]//2)
            x_hat, iters = reconstruct_sgd(
                y=y,
                x_init=x_init,
                physics=self.physics,
                data_fidelity=self.data_fidelity,
                regularizer=self.model,
                lamda=self.lmbd,
                step_size= lower_lr,
                patch_size=patch_size,
                max_iter=max_iter,
                tol=self.eps,
                detach_grads=True,
                verbose=False,
                return_stats=False,
            )
        elif alg == "FISTA":
            x_hat, iters = reconstruct_fista(
                y=y,
                x_init=x_init.to(self.device) if x_init is not None else None,
                physics=self.physics,
                data_fidelity=self.data_fidelity,
                regularizer=self.model,
                lamda=self.lmbd,
                step_size=lower_lr,
                max_iter=max_iter,
                tol=self.eps,
                detach_grads=True,
                verbose=False,
                return_stats=False,
            )
        mse = torch.mean(torch.norm(x_hat - self.gt_slice, p=2, dim=(1, 2, 3)) ** 2)
        return mse, x_hat, iters

    def estimate_gradient_single_level(
        self, sigma=1e-1, N=20, max_iter=100, warm_start=False, normalize_grad=True, N_l=None
    ):
        original_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        grad_est = {name: torch.zeros_like(p) for name, p in self.model.named_parameters()}
        loss_list, psnr_list, total_iters = [], [], 0
        x_hat = x_hat_pos = None
        replace = False if N_l <= self.gt.shape[0] else True
        self.slice = np.random.choice(self.gt.shape[0], N_l, replace=replace) if N_l else np.arange(self.gt.shape[0])
        for _ in range(N):
            noise = self.sample_noise()

            # θ + σu
            self.apply_perturbation(noise, sigma)
            f_pos, x_hat_pos_out, iters_pos = self.evaluate_objective(x_init=x_hat_pos, max_iter=max_iter, N_l=N_l, alg= self.alg, lower_lr= self.lower_lr)

            self.set_parameters(original_params)

            # θ
            f, x_hat_out, iters = self.evaluate_objective(x_init=x_hat, max_iter=max_iter, N_l=N_l, alg=self.alg, lower_lr= self.lower_lr)
            total_iters += (iters+1) * N_l + (iters_pos + 1) * N_l

            if warm_start:
                x_hat = x_hat_out
                x_hat_pos = x_hat_pos_out
            for name in grad_est:
                grad_est[name] += (f_pos - f) * noise[name]

            loss_list.append(f.item())
            psnr_list.append(self.psnr_metric(self.gt_slice, x_hat_out).mean().item())

        # Normalize gradients
        for name in grad_est:
            grad_est[name] /= (N * sigma)

        if normalize_grad:
            norm = torch.sqrt(sum((g**2).sum() for g in grad_est.values()))
            for g in grad_est.values():
                g /= (norm + 1e-8)
        self.gt_slice = self.gt.clone()  # Reset ground truth slice for next iteration
        # plt.imshow(x_hat_out[0,0].cpu().numpy(), cmap='gray')
        # plt.title("Reconstructed Image")
        # plt.axis('off')
        # plt.show()
        return grad_est, np.mean(loss_list), np.mean(psnr_list), total_iters

    def multilevel_gradient_step(
        self,
        levels: List[int],
        N_per_level_upper: List[int],
        N_per_level_lower: List[int],
        sigma=1e-1,
        warm_start=True,
        normalize_grad=True,
        verbose=True,        
    ):
        assert len(levels) == len(N_per_level_upper) == len(N_per_level_lower)

        grad_total = {name: torch.zeros_like(p) for name, p in self.model.named_parameters()}
        total_loss, total_psnr, total_iters = [], [], 0

        for i, (lvl, N, N_l) in enumerate(zip(levels, N_per_level_upper, N_per_level_lower)):
            grad_high, loss_high, psnr_high, iters_high = self.estimate_gradient_single_level(
                sigma=sigma, N=N, max_iter=lvl, warm_start=warm_start, normalize_grad=False, N_l =N_l
            )

            if i > 0:
                grad_low, *_ = self.estimate_gradient_single_level(
                    sigma=sigma, N=N, max_iter=levels[i-1], warm_start=warm_start, normalize_grad=False, N_l=N_l
                )
            else:
                grad_low = {name: torch.zeros_like(p) for name, p in self.model.named_parameters()}

            for name in grad_total:
                grad_total[name] += grad_high[name] - grad_low[name]

            total_loss.append(loss_high)
            total_psnr.append(psnr_high)
            total_iters += iters_high

        # Normalize gradients if needed
        if normalize_grad:
            norm = torch.sqrt(sum((g**2).sum() for g in grad_total.values()))
            for g in grad_total.values():
                g /= (norm + 1e-8)

        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            param.grad = grad_total[name]
        self.optimizer.step()
        self.scheduler.step( )
        print(self.optimizer.param_groups[0]['lr'])
        if verbose:
            print(f"[MLMC] Loss: {np.mean(total_loss):.4f}, PSNR: {np.mean(total_psnr):.2f}, Cost: {total_iters}")

        return np.mean(total_loss), np.mean(total_psnr), total_iters