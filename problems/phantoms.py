"""
Random phantom generators for CT experiments.
Ported from MATLAB random_triangle.m / random_ellipse.m / random_phantom.m.
"""
import torch
import numpy as np


def random_triangle(n, num_triangles=1, device="cpu"):
    """
    Generate a random triangle phantom image.

    Ported from MATLAB random_triangle.m.

    Args:
        n: Image size (n x n)
        num_triangles: Number of triangles to generate
        device: Torch device

    Returns:
        Tensor of shape (1, 1, n, n)
    """
    P = np.zeros((n, n), dtype=np.float32)

    for _ in range(num_triangles):
        PP = np.zeros((n, n), dtype=np.float32)
        height = 0.5 + 0.5 * np.random.rand()
        size = max(8, 1 + np.random.randint(n // 4 - 1))

        # Force the triangle to stay away from boundary
        pos1 = max(size + 8, min(np.random.randint(n - 2 * size - 8) + size + 8, n - size - 8))
        pos2 = max(size + 8, min(np.random.randint(n - size - 16) + size - 1 + 8, n - 8))

        # Clip to valid range
        pos1 = np.clip(pos1, size + 1, n - size - 2)
        pos2 = np.clip(pos2, size + 1, n - 2)

        PP[pos1, min(pos2 + 1, n - 1)] = height
        for s in range(1, size + 1):
            row_start = max(0, pos1 - s)
            row_end = min(n, pos1 + s + 1)
            col = max(0, pos2 - s + 1)
            if col < n:
                PP[row_start:row_end, col] = height

        P = PP + P

    P_tensor = torch.from_numpy(P).unsqueeze(0).unsqueeze(0).to(device)
    return P_tensor


def random_ellipse_phantom(n, num_ellipses=3, device="cpu"):
    """
    Generate a random ellipse phantom image.

    Args:
        n: Image size (n x n)
        num_ellipses: Number of ellipses
        device: Torch device

    Returns:
        Tensor of shape (1, 1, n, n)
    """
    P = np.zeros((n, n), dtype=np.float32)
    y_coords, x_coords = np.mgrid[0:n, 0:n]
    x_coords = x_coords.astype(np.float32) / n
    y_coords = y_coords.astype(np.float32) / n

    for _ in range(num_ellipses):
        cx = 0.2 + 0.6 * np.random.rand()
        cy = 0.2 + 0.6 * np.random.rand()
        rx = 0.05 + 0.15 * np.random.rand()
        ry = 0.05 + 0.15 * np.random.rand()
        intensity = 0.3 + 0.7 * np.random.rand()

        mask = ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2 <= 1.0
        P[mask] += intensity

    P = np.clip(P, 0, 1)
    return torch.from_numpy(P).unsqueeze(0).unsqueeze(0).to(device)


class PhantomDataset(torch.utils.data.Dataset):
    """
    Dataset that generates random phantom images on-the-fly.

    Args:
        n: Image size
        length: Number of samples
        phantom_type: 'triangle' or 'ellipse'
    """

    def __init__(self, n=64, length=100, phantom_type="triangle"):
        self.n = n
        self.length = length
        self.phantom_type = phantom_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.phantom_type == "triangle":
            img = random_triangle(self.n, num_triangles=1)
        else:
            img = random_ellipse_phantom(self.n, num_ellipses=3)
        return img.squeeze(0)  # Return (1, n, n)
