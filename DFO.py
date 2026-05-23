import torch
import numpy as np
from deepinv.loss.metric import PSNR, SSIM
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
        lower_tol=1e-1,
        fista_mode="legacy",
        sgd_step_decay="none",
        sgd_patch_mode="patch",
        sgd_patch_fraction=0.5,
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
        self.ssim_metric = SSIM()
        self.optimizer = self._build_optimizer(optimizer_type, lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=200,   # number of iterations/epochs for one cosine cycle
            eta_min=0.0  # minimum learning rate (default 0)
        )
        self.alg = lower_alg
        self.lower_lr = lower_lr
        self.eps = lower_tol
        self.fista_mode = fista_mode
        self.sgd_step_decay = sgd_step_decay
        self.sgd_patch_mode = sgd_patch_mode
        self.sgd_patch_fraction = sgd_patch_fraction

    def trainable_named_parameters(self):
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def _build_optimizer(self, opt_type, lr):
        param_groups = []
        for name, param in self.trainable_named_parameters():
            # Filters (Raw Linear Kernels): Need 10x slower learning since a massive 0.05 absolute update destroys 5x5 structural masks
            if "conv" in name or param.dim() >= 2:
                param_groups.append({'params': param, 'lr': lr * 0.1})
            # Smoothing (Log Bounds): 10x slower learning prevents boundary destabilizations
            elif "nu" in name or "smoothing" in name:
                param_groups.append({'params': param, 'lr': lr * 0.1})
            # Coefficients (Log Multipliers): Perfectly scaled for 1.0x learning rate (~10% functional update)
            else:
                param_groups.append({'params': param, 'lr': lr})


        if opt_type == "Adam":
            return torch.optim.Adam(param_groups)
        return torch.optim.SGD(param_groups)

    def sample_noise(self, normalize=True):
        named_params = self.trainable_named_parameters()
        param_shapes = [p.shape for _, p in named_params]
        total_size = sum(p.numel() for _, p in named_params)
        noise_vec = torch.randn(total_size, device=self.device)
        if normalize:
            noise_vec /= (torch.norm(noise_vec) + 1e-8)

        noise_dict = {}
        idx = 0
        for (name, param), shape in zip(named_params, param_shapes):
            size = param.numel()
            noise_dict[name] = noise_vec[idx:idx+size].reshape(shape)
            idx += size
        return noise_dict

    def apply_perturbation(self, noise, sigma, sign=1):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in noise:
                    param.add_(sign * sigma * noise[name])

    def set_parameters(self, param_dict):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in param_dict:
                    param.copy_(param_dict[name])

    def sample_batch_indices(self, N_l=None):
        if N_l is None:
            N_l = self.gt.shape[0]
        replace = N_l > self.gt.shape[0]
        self.slice = np.random.choice(self.gt.shape[0], N_l, replace=replace)
        return N_l

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
            patch_fraction = min(max(float(self.sgd_patch_fraction), 1e-4), 1.0)
            patch_size = (
                max(1, int(round(y.shape[2] * patch_fraction))),
                max(1, int(round(y.shape[3] * patch_fraction))),
            )
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
                step_decay=self.sgd_step_decay,
                patch_mode=self.sgd_patch_mode,
                mask_probability=patch_fraction * patch_fraction,
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
                mode=self.fista_mode,
            )
        mse = torch.mean(torch.norm(x_hat - self.gt_slice, p=2, dim=(1, 2, 3)) ** 2)
        return mse, x_hat, iters

    def estimate_gradient_single_level(
        self, sigma=1e-1, N=20, max_iter=100, warm_start=False, normalize_grad=True, N_l=None
    ):
        original_params = {n: p.detach().clone() for n, p in self.trainable_named_parameters()}
        grad_est = {name: torch.zeros_like(p) for name, p in self.trainable_named_parameters()}
        loss_list, psnr_list, ssim_list, total_iters = [], [], [], 0
        x_hat = x_hat_pos = None
        N_l = self.sample_batch_indices(N_l)
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
            ssim_list.append(self.ssim_metric(self.gt_slice, x_hat_out).mean().item())

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
        return grad_est, np.mean(loss_list), np.mean(psnr_list), np.mean(ssim_list), total_iters

    def estimate_gradient_level_difference(
        self,
        sigma=1e-1,
        N=20,
        high_iter=100,
        low_iter=50,
        warm_start=False,
        normalize_grad=True,
        N_l=None,
    ):
        original_params = {n: p.detach().clone() for n, p in self.trainable_named_parameters()}
        grad_est = {name: torch.zeros_like(p) for name, p in self.trainable_named_parameters()}
        loss_list, psnr_list, ssim_list, total_iters = [], [], [], 0
        x_high = x_high_pos = None
        x_low = x_low_pos = None
        N_l = self.sample_batch_indices(N_l)

        for _ in range(N):
            noise = self.sample_noise()

            self.apply_perturbation(noise, sigma)
            f_high_pos, x_high_pos_out, iters_high_pos = self.evaluate_objective(
                x_init=x_high_pos, max_iter=high_iter, N_l=N_l, alg=self.alg, lower_lr=self.lower_lr
            )
            f_low_pos, x_low_pos_out, iters_low_pos = self.evaluate_objective(
                x_init=x_low_pos, max_iter=low_iter, N_l=N_l, alg=self.alg, lower_lr=self.lower_lr
            )

            self.set_parameters(original_params)

            f_high, x_high_out, iters_high = self.evaluate_objective(
                x_init=x_high, max_iter=high_iter, N_l=N_l, alg=self.alg, lower_lr=self.lower_lr
            )
            f_low, x_low_out, iters_low = self.evaluate_objective(
                x_init=x_low, max_iter=low_iter, N_l=N_l, alg=self.alg, lower_lr=self.lower_lr
            )

            total_iters += (
                (iters_high_pos + 1) * N_l
                + (iters_low_pos + 1) * N_l
                + (iters_high + 1) * N_l
                + (iters_low + 1) * N_l
            )

            if warm_start:
                x_high = x_high_out
                x_high_pos = x_high_pos_out
                x_low = x_low_out
                x_low_pos = x_low_pos_out

            diff_pos = f_high_pos - f_low_pos
            diff_base = f_high - f_low
            for name in grad_est:
                grad_est[name] += (diff_pos - diff_base) * noise[name]

            loss_list.append(f_high.item())
            psnr_list.append(self.psnr_metric(self.gt_slice, x_high_out).mean().item())
            ssim_list.append(self.ssim_metric(self.gt_slice, x_high_out).mean().item())

        for name in grad_est:
            grad_est[name] /= (N * sigma)

        if normalize_grad:
            norm = torch.sqrt(sum((g**2).sum() for g in grad_est.values()))
            for g in grad_est.values():
                g /= (norm + 1e-8)

        self.gt_slice = self.gt.clone()
        return grad_est, np.mean(loss_list), np.mean(psnr_list), np.mean(ssim_list), total_iters

    def apply_gradient_step(self, grad_est, normalize_grad=True):
        if normalize_grad:
            norm = torch.sqrt(sum((g**2).sum() for g in grad_est.values()))
            for g in grad_est.values():
                g /= (norm + 1e-8)

        self.optimizer.zero_grad()
        for name, param in self.trainable_named_parameters():
            param.grad = grad_est[name]
        self.optimizer.step()
        self.scheduler.step()

    def single_level_gradient_step(
        self,
        level: int,
        N_upper: int,
        N_lower: int,
        sigma=1e-1,
        warm_start=True,
        normalize_grad=True,
        verbose=True,
    ):
        grad_est, loss, psnr, ssim, total_iters = self.estimate_gradient_single_level(
            sigma=sigma,
            N=N_upper,
            max_iter=level,
            warm_start=warm_start,
            normalize_grad=False,
            N_l=N_lower,
        )

        self.apply_gradient_step(grad_est, normalize_grad=normalize_grad)
        if verbose:
            print(f"[SLMC] Loss: {loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Cost: {total_iters}")

        return loss, psnr, ssim, total_iters

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

        grad_total = {name: torch.zeros_like(p) for name, p in self.trainable_named_parameters()}
        total_loss, total_psnr, total_ssim, total_iters = [], [], [], 0

        for i, (lvl, N, N_l) in enumerate(zip(levels, N_per_level_upper, N_per_level_lower)):
            if i == 0:
                grad_delta, loss_high, psnr_high, ssim_high, iters_delta = self.estimate_gradient_single_level(
                    sigma=sigma, N=N, max_iter=lvl, warm_start=warm_start, normalize_grad=False, N_l=N_l
                )
            else:
                grad_delta, loss_high, psnr_high, ssim_high, iters_delta = self.estimate_gradient_level_difference(
                    sigma=sigma, N=N, high_iter=lvl, low_iter=levels[i-1],
                    warm_start=warm_start, normalize_grad=False, N_l=N_l
                )

            for name in grad_total:
                grad_total[name] += grad_delta[name]

            total_loss.append(loss_high)
            total_psnr.append(psnr_high)
            total_ssim.append(ssim_high)
            total_iters += iters_delta

        self.apply_gradient_step(grad_total, normalize_grad=normalize_grad)

        # Report the finest-level reconstruction metric. Averaging the metrics
        # over levels is useful diagnostically, but it unfairly mixes the coarse
        # 50-iteration reconstructions with the finest level used for comparison
        # against SLMC and validation.
        finest_loss = float(total_loss[-1])
        finest_psnr = float(total_psnr[-1])
        finest_ssim = float(total_ssim[-1])
        level_mean_loss = float(np.mean(total_loss))

        if verbose:
            print(
                f"[MLMC] Loss: {finest_loss:.4f}, PSNR: {finest_psnr:.2f}, "
                f"SSIM: {finest_ssim:.4f}, LevelMeanLoss: {level_mean_loss:.4f}, "
                f"Cost: {total_iters}"
            )

        return finest_loss, finest_psnr, finest_ssim, total_iters
