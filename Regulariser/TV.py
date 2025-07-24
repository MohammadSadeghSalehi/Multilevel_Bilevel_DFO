#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinv.optim import Prior


class SmoothedTV(Prior):
    """
    Smoothed Total Variation regularizer with learnable parameters.
    
    The regularizer computes:
    TV_smooth(x) = sum(sqrt(|∇_x|² + |∇_y|² + ε²))
    
    where ε is the smoothing parameter and the sum is weighted by a learnable coefficient.
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
        
        # Learnable TV coefficient (in log space for positivity)
        if learn_tv_weight:
            self.log_tv_weight = nn.Parameter(torch.log(torch.tensor(tv_weight)))
        else:
            self.register_buffer('log_tv_weight', torch.log(torch.tensor(tv_weight)))
            
        # Learnable smoothing parameter (in log space for positivity)
        if learn_smoothing:
            self.log_smoothing = nn.Parameter(torch.log(torch.tensor(smoothing)))
        else:
            self.register_buffer('log_smoothing', torch.log(torch.tensor(smoothing)))
        
        # Gradient operators (Sobel-like filters)
        self.register_buffer('grad_x', torch.tensor([[[[-1., 0., 1.],
                                                       [-2., 0., 2.],
                                                       [-1., 0., 1.]]]]) / 8.0)
        
        self.register_buffer('grad_y', torch.tensor([[[[-1., -2., -1.],
                                                       [ 0.,  0.,  0.],
                                                       [ 1.,  2.,  1.]]]]) / 8.0)
        
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))
    
    @property
    def tv_weight(self):
        """Get the current TV weight (positive)"""
        return torch.exp(self.log_tv_weight)
    
    @property
    def smoothing(self):
        """Get the current smoothing parameter (positive)"""
        return torch.exp(self.log_smoothing)
    
    def compute_gradients(self, x):
        """
        Compute spatial gradients using convolution.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            grad_x, grad_y: Gradient tensors
        """
        # Pad input to handle boundaries
        x_padded = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        
        # Compute gradients for each channel
        grad_x = F.conv2d(x_padded, self.grad_x.expand(x.size(1), -1, -1, -1), 
                         groups=x.size(1), padding=0)
        grad_y = F.conv2d(x_padded, self.grad_y.expand(x.size(1), -1, -1, -1), 
                         groups=x.size(1), padding=0)
        
        return grad_x, grad_y
    
    def g(self, x):
        """
        Compute the smoothed TV regularization energy.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            reg: Regularization energy for each sample in batch
        """
        grad_x, grad_y = self.compute_gradients(x)
        
        # Compute smoothed TV: sqrt(|∇_x|² + |∇_y|² + ε²)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.smoothing**2)
        
        # Sum over spatial dimensions and channels, weight by learnable coefficient
        tv_energy = self.tv_weight * grad_magnitude.sum(dim=(1, 2, 3))
        
        return tv_energy
    
    def grad(self, x, get_energy=False):
        """
        Compute the gradient of the smoothed TV regularizer.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            get_energy: If True, also return the energy
            
        Returns:
            grad: Gradient tensor
            energy (optional): Regularization energy
        """
        grad_x, grad_y = self.compute_gradients(x)
        
        # Compute smoothed gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.smoothing**2)
        
        # Compute energy if requested
        if get_energy:
            energy = self.tv_weight * grad_magnitude.sum(dim=(1, 2, 3))
        
        # Compute gradient of smoothed TV
        # ∂/∂x [sqrt(|∇_x|² + |∇_y|² + ε²)] = ∇_x / sqrt(|∇_x|² + |∇_y|² + ε²)
        grad_x_norm = grad_x / grad_magnitude
        grad_y_norm = grad_y / grad_magnitude
        
        # Apply transpose of gradient operators (divergence)
        # Transpose convolution with flipped kernels
        grad_x_kernel = torch.flip(self.grad_x, dims=[2, 3])
        grad_y_kernel = torch.flip(self.grad_y, dims=[2, 3])
        
        # Pad normalized gradients
        grad_x_norm_padded = F.pad(grad_x_norm, (1, 1, 1, 1), mode='constant', value=0)
        grad_y_norm_padded = F.pad(grad_y_norm, (1, 1, 1, 1), mode='constant', value=0)
        
        # Compute divergence (negative gradient of TV)
        div_x = F.conv2d(grad_x_norm_padded, grad_x_kernel.expand(x.size(1), -1, -1, -1),
                        groups=x.size(1), padding=0)
        div_y = F.conv2d(grad_y_norm_padded, grad_y_kernel.expand(x.size(1), -1, -1, -1), 
                        groups=x.size(1), padding=0)
        
        # Total gradient (with learnable weight)
        total_grad = self.tv_weight * (div_x + div_y)
        
        if get_energy:
            return energy, total_grad
        return total_grad


# Example usage and testing
if __name__ == "__main__":
    # Create a smoothed TV regularizer
    reg = SmoothedTV(tv_weight=0.1, smoothing=1e-3)
    
    # Test with a sample image
    x = torch.randn(2, 3, 64, 64, requires_grad=True)
    
    # Compute energy
    energy = reg.g(x)
    print(f"TV energy shape: {energy.shape}")
    print(f"TV energy values: {energy}")
    
    # Compute gradient
    grad = reg.grad(x)
    print(f"Gradient shape: {grad.shape}")
    
    # Test energy and gradient together
    energy2, grad2 = reg.grad(x, get_energy=True)
    print(f"Energy matches: {torch.allclose(energy, energy2)}")
    print(f"Gradient matches: {torch.allclose(grad, grad2)}")
    
    # Check learnable parameters
    print(f"Current TV weight: {reg.tv_weight.item():.6f}")
    print(f"Current smoothing: {reg.smoothing.item():.6f}")