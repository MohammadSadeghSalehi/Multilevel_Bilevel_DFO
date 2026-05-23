#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from deepinv.optim import Prior


class ZeroMean(nn.Module):
    """Enforcing zero mean on the filters improves performances"""

    def forward(self, x):
        return x - torch.mean(x, dim=(2, 3), keepdim=True)


class FieldOfExperts(Prior):
    """
    Field of Experts regularizer with learnable convolutional kernels and smooth potentials.

    The regularizer computes:
    FoE(x) = Σ_k λ_k * Σ_i ψ(K_k * x_i, ε_k)

    where:
    - K_k are learnable convolution kernels
    - ψ is a smooth potential function (smoothed L1-norm)
    - λ_k are learnable weights for each expert
    - ε_k are learnable smoothing parameters for each expert
    """

    def __init__(
        self,
        num_experts=8,
        kernel_size=5,
        num_channels=1,
        colored=False,
        expert_weights=None,
        smoothing=1e-3,
        tanh_potential=False,
        zero_mean=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained=None,
    ):
        super(FieldOfExperts, self).__init__()

        self.num_experts = num_experts
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.colored = colored
        self.tanh_potential = tanh_potential
        self.zero_mean = zero_mean
        self.total_weight = nn.Parameter(torch.tensor(-1.0), requires_grad=True)
        # Determine input/output channels based on color mode
        if colored:
            # For colored images, we can either:
            # 1. Apply same kernels to all channels (groups=1)
            # 2. Apply different kernels per channel (groups=num_channels)
            # Here we use option 1 for simplicity
            conv_in_channels = num_channels
            conv_out_channels = num_experts
        else:
            # For grayscale
            conv_in_channels = num_channels
            conv_out_channels = num_experts

        # Learnable convolution kernels (experts)
        self.conv_experts = nn.Conv2d(
            conv_in_channels,
            conv_out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
            padding_mode='zeros'
        )

        # Apply zero-mean constraint if requested
        if zero_mean:
            P.register_parametrization(self.conv_experts, "weight", ZeroMean())

        # Initialize kernels
        self._initialize_kernels()

        # Learnable expert weights (in log space for positivity)
        if expert_weights is None:
            expert_weights = torch.ones(num_experts)
        else:
            expert_weights = torch.tensor(expert_weights)
        self.log_expert_weights = nn.Parameter(torch.log(expert_weights))

        # Learnable smoothing parameters for each expert (in log space)
        if isinstance(smoothing, (int, float)):
            smoothing = torch.full((num_experts,), smoothing)
        else:
            smoothing = torch.tensor(smoothing)
        self.log_smoothing = nn.Parameter(torch.log(smoothing))

        # For computing transpose convolution
        self._setup_transpose_conv()

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

    def _initialize_kernels(self):
        """Initialize convolution kernels with common edge detection patterns"""
        with torch.no_grad():
            # Initialize with Xavier normal
            nn.init.xavier_normal_(self.conv_experts.weight)

            # Optionally initialize some kernels with common patterns
            if self.num_experts >= 4 and self.kernel_size >= 3:
                center = self.kernel_size // 2

                # Horizontal edge detector (central row)
                self.conv_experts.weight[0].fill_(0)
                if self.kernel_size == 3:
                    self.conv_experts.weight[0, :, center, :] = torch.tensor([-1., 0., 1.])
                elif self.kernel_size == 5:
                    self.conv_experts.weight[0, :, center, :] = torch.tensor([-2., -1., 0., 1., 2.])
                else:
                    # General case: simple difference kernel
                    self.conv_experts.weight[0, :, center, 0] = -1.
                    self.conv_experts.weight[0, :, center, -1] = 1.

                # Vertical edge detector (central column)
                self.conv_experts.weight[1].fill_(0)
                if self.kernel_size == 3:
                    self.conv_experts.weight[1, :, :, center] = torch.tensor([-1., 0., 1.])
                elif self.kernel_size == 5:
                    self.conv_experts.weight[1, :, :, center] = torch.tensor([-2., -1., 0., 1., 2.])
                else:
                    # General case: simple difference kernel
                    self.conv_experts.weight[1, :, 0, center] = -1.
                    self.conv_experts.weight[1, :, -1, center] = 1.

                # Diagonal edge detectors
                self.conv_experts.weight[2].fill_(0)
                self.conv_experts.weight[2, :, 0, 0] = -1
                self.conv_experts.weight[2, :, center, center] = 0
                self.conv_experts.weight[2, :, -1, -1] = 1

                self.conv_experts.weight[3].fill_(0)
                self.conv_experts.weight[3, :, 0, -1] = -1
                self.conv_experts.weight[3, :, center, center] = 0
                self.conv_experts.weight[3, :, -1, 0] = 1

            # Apply zero mean constraint manually during initialization
            if self.zero_mean:
                for i in range(self.num_experts):
                    kernel_mean = self.conv_experts.weight[i].mean(dim=(1, 2), keepdim=True)
                    self.conv_experts.weight[i] = self.conv_experts.weight[i] - kernel_mean

    def _setup_transpose_conv(self):
        """Setup for transpose convolution operations"""
        # Create transpose convolution layer with same parameters
        self.conv_transpose = nn.ConvTranspose2d(
            self.num_experts,
            self.num_channels,
            self.kernel_size,
            padding=self.kernel_size // 2,
            bias=False
        )
        self.conv_transpose.weight.requires_grad = False

        # The transpose conv will use the same weights as forward conv (but transposed)
        # This is handled in the forward pass

    @property
    def expert_weights(self):
        """Get current expert weights (positive)"""
        return torch.exp(self.log_expert_weights)

    @property
    def smoothing_params(self):
        """Get current smoothing parameters (positive)"""
        return torch.exp(self.log_smoothing)

    def smooth_l1_potential(self, x, eps):
        """
        Smooth L1 potential function.

        Args:
            x: Input tensor
            eps: Smoothing parameter

        Returns:
            Smoothed L1 values
        """
        if self.tanh_potential:
            # Smooth approximation using log-cosh
            x_abs = torch.abs(x)
            return torch.log((torch.exp(x/eps - x_abs/eps) + torch.exp(-x/eps - x_abs/eps)) / 2) * eps + x_abs
        else:
            # Huber-like smooth L1
            abs_x = torch.abs(x)
            return torch.where(abs_x <= eps,
                             x**2 / (2 * eps),
                             abs_x - eps/2)

    def grad_smooth_l1_potential(self, x, eps):
        """
        Gradient of smooth L1 potential function.

        Args:
            x: Input tensor
            eps: Smoothing parameter

        Returns:
            Gradient of smoothed L1
        """
        if self.tanh_potential:
            return torch.tanh(x / eps)
        else:
            # Huber-like gradient
            abs_x = torch.abs(x)
            return torch.where(abs_x <= eps,
                             x / eps,
                             torch.sign(x))

    def apply_experts(self, x):
        """Apply all expert kernels to input"""
        return self.conv_experts(x)

    def apply_experts_transpose(self, x):
        """Apply transpose of expert kernels"""
        # Use the weights from conv_experts but in transpose mode
        # We need to manually set the weights for transpose conv
        self.conv_transpose.weight.data = self.conv_experts.weight.data
        return self.conv_transpose(x)

    def g(self, x):
        """
        Compute the Field of Experts regularization energy.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            reg: Regularization energy for each sample in batch
        """
        # Apply expert convolutions
        expert_responses = self.apply_experts(x)  # Shape: (B, num_experts, H, W)

        total_energy = torch.zeros(x.size(0), device=x.device)

        # Compute potential for each expert
        for k in range(self.num_experts):
            responses_k = expert_responses[:, k:k+1]  # (B, 1, H, W)
            eps_k = self.smoothing_params[k]
            weight_k = self.expert_weights[k]

            # Apply smooth potential
            potential_values = self.smooth_l1_potential(responses_k, eps_k)

            # Sum over spatial dimensions and weight by expert weight
            expert_energy = weight_k * potential_values.sum(dim=(1, 2, 3))
            total_energy = total_energy + expert_energy

        return total_energy * self.total_weight.exp()  # Scale by total weight

    def grad(self, x, get_energy=False):
        """
        Compute the gradient of the Field of Experts regularizer.

        Args:
            x: Input tensor of shape (B, C, H, W)
            get_energy: If True, also return the energy

        Returns:
            grad: Gradient tensor
            energy (optional): Regularization energy
        """
        # Apply expert convolutions
        expert_responses = self.apply_experts(x)  # Shape: (B, num_experts, H, W)

        if get_energy:
            total_energy = torch.zeros(x.size(0), device=x.device)

        # Initialize gradient accumulator
        total_grad = torch.zeros_like(expert_responses)

        # Compute gradient for each expert
        for k in range(self.num_experts):
            responses_k = expert_responses[:, k:k+1]  # (B, 1, H, W)
            eps_k = self.smoothing_params[k]
            weight_k = self.expert_weights[k]

            if get_energy:
                # Compute energy for this expert
                potential_values = self.smooth_l1_potential(responses_k, eps_k)
                expert_energy = weight_k * potential_values.sum(dim=(1, 2, 3))
                total_energy = total_energy + expert_energy

            # Compute gradient of potential
            grad_potential = self.grad_smooth_l1_potential(responses_k, eps_k)

            # Weight by expert weight
            total_grad[:, k:k+1] = weight_k * grad_potential

        # Apply transpose of expert convolutions
        final_grad = self.apply_experts_transpose(total_grad)
        scale = self.total_weight.exp()

        if get_energy:
            return total_energy * scale, final_grad * scale
        return final_grad * scale

    def get_expert_kernels(self):
        """Return the current expert kernels for visualization"""
        return self.conv_experts.weight.data.clone()

    def set_expert_kernel(self, expert_idx, kernel):
        """Set a specific expert kernel"""
        with torch.no_grad():
            self.conv_experts.weight[expert_idx] = kernel
            if self.zero_mean:
                # Apply zero mean constraint
                kernel_mean = self.conv_experts.weight[expert_idx].mean(dim=(1, 2), keepdim=True)
                self.conv_experts.weight[expert_idx] = self.conv_experts.weight[expert_idx] - kernel_mean


# Example usage and testing
# if __name__ == "__main__":
#     # Test with grayscale images
#     print("Testing with grayscale images:")
#     foe_gray = FieldOfExperts(num_experts=4, kernel_size=5, num_channels=1, colored=False)

#     x_gray = torch.randn(2, 1, 32, 32, requires_grad=True)
#     energy_gray = foe_gray.g(x_gray)
#     grad_gray = foe_gray.grad(x_gray)

#     print(f"Grayscale - Energy shape: {energy_gray.shape}, values: {energy_gray}")
#     print(f"Grayscale - Gradient shape: {grad_gray.shape}")
#     print(f"Expert weights: {foe_gray.expert_weights}")
#     print(f"Smoothing params: {foe_gray.smoothing_params}")

#     # Test with colored images
#     print("\nTesting with colored images:")
#     foe_color = FieldOfExperts(num_experts=6, kernel_size=3, num_channels=3, colored=True)

#     x_color = torch.randn(2, 3, 32, 32, requires_grad=True)
#     energy_color = foe_color.g(x_color)
#     grad_color = foe_color.grad(x_color)

#     print(f"Color - Energy shape: {energy_color.shape}, values: {energy_color}")
#     print(f"Color - Gradient shape: {grad_color.shape}")

#     # Test energy and gradient consistency
#     energy2, grad2 = foe_gray.grad(x_gray, get_energy=True)
#     print(f"\nConsistency check - Energy matches: {torch.allclose(energy_gray, energy2)}")
#     print(f"Consistency check - Gradient matches: {torch.allclose(grad_gray, grad2)}")

#     # Display some kernel information
#     kernels = foe_gray.get_expert_kernels()
#     print(f"\nKernel shapes: {kernels.shape}")
#     print(f"First kernel mean (should be ~0 if zero_mean=True): {kernels[0].mean().item():.6f}")
