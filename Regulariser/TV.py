import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.optim import Prior


class SmoothedTV(Prior):
    """
    Smoothed Total Variation regularizer using finite differences (not Sobel).
    
    Computes: TV_smooth(x) = sum(sqrt(|∇x|^2 + |∇y|^2 + ε^2))
    """

    def __init__(
        self,
        tv_weight=0.1,
        smoothing=1e-3,
        learn_tv_weight=True,
        learn_smoothing=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
    ):
        super(SmoothedTV, self).__init__()

        if learn_tv_weight:
            self.log_tv_weight = nn.Parameter(torch.log(torch.tensor(tv_weight)))
        else:
            self.register_buffer("log_tv_weight", torch.log(torch.tensor(tv_weight)))

        if learn_smoothing:
            self.log_smoothing = nn.Parameter(torch.log(torch.tensor(smoothing)))
        else:
            self.register_buffer("log_smoothing", torch.log(torch.tensor(smoothing)))

        # Simple forward difference kernels
        self.register_buffer("grad_x", torch.tensor([[[[0, -1, 1]]]], dtype=torch.float32))
        self.register_buffer("grad_y", torch.tensor([[[[0], [-1], [1]]]], dtype=torch.float32))

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    @property
    def tv_weight(self):
        return torch.exp(self.log_tv_weight)

    @property
    def smoothing(self):
        return torch.exp(self.log_smoothing)

    def compute_gradients(self, x):
        B, C, H, W = x.shape
        grad_x = F.conv2d(x, self.grad_x.expand(C, -1, -1, -1), groups=C, padding=(0, 1))
        grad_y = F.conv2d(x, self.grad_y.expand(C, -1, -1, -1), groups=C, padding=(1, 0))
        return grad_x, grad_y

    def g(self, x):
        grad_x, grad_y = self.compute_gradients(x)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.smoothing**2)
        tv_energy = self.tv_weight * grad_magnitude.sum(dim=(1, 2, 3))
        return tv_energy

    def grad(self, x, get_energy=False):
        B, C, H, W = x.shape
        grad_x, grad_y = self.compute_gradients(x)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.smoothing**2)

        if get_energy:
            energy = self.tv_weight * grad_magnitude.sum(dim=(1, 2, 3))

        grad_x_norm = grad_x / grad_magnitude
        grad_y_norm = grad_y / grad_magnitude

        grad_x_kernel = torch.flip(self.grad_x, dims=[2, 3])
        grad_y_kernel = torch.flip(self.grad_y, dims=[2, 3])

        div_x = F.conv2d(grad_x_norm, grad_x_kernel.expand(C, -1, -1, -1), groups=C, padding=(0, 1))
        div_y = F.conv2d(grad_y_norm, grad_y_kernel.expand(C, -1, -1, -1), groups=C, padding=(1, 0))

        total_grad = self.tv_weight * (div_x + div_y)

        if get_energy:
            return energy, total_grad
        return total_grad
