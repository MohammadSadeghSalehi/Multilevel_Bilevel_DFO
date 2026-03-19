# Multilevel Bilevel Derivative-Free Optimization (DFO) Pipeline

This repository contains a PyTorch-based pipeline for solving bilevel optimization problems using Multilevel Monte Carlo (MLMC) gradient estimation with Gaussian smoothing. It supports various inverse problems, including image restoration (denoising, inpainting), Computed Tomography (CT), and Parameter Estimation in Partial Differential Equations (PDEs).

## Dependencies

The project relies on [DeepInverse](https://deepinv.github.io/) for physics operators and metrics, along with standard scientific computing libraries.

## Installation Guide

It is recommended to use a Conda environment for dependency management.

1. **Create and activate a new Conda environment:**
   ```bash
   conda create -n BiMo python=3.10
   conda activate BiMo
   ```

2. **Install PyTorch:**
   Please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to install the correct version of PyTorch corresponding to your system and CUDA setup. A standard installation might look like:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

3. **Install the remaining requirements:**
   Navigate to the project directory and install the dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

The main entry point for running experiments is `train.py`. It provides a configurable CLI interface.

### General Command Structure

```bash
python train.py --problem <PROBLEM_TYPE> --regulariser <REGULARISER_TYPE> [OPTIONS]
```

### Available Problem Types (`--problem`)
- `Denoising`: Image denoising with Gaussian noise.
- `Inpainting`: Image inpainting with random missing pixels.
- `CT`: Computed Tomography reconstruction.
- `PDE`: 2D Elliptic PDE parameter estimation and optimal sensor placement.

### Available Regularizers (`--regulariser`)
- `TV`: Smoothed Total Variation.
- `FoE`: Field of Experts.
- `CT_TV`: Specialized combined Tikhonov + Smoothed TV for the CT problem.

### Key Command Line Arguments
- `--lower_alg`: Lower-level solver (`GD`, `SGD`, `FISTA`). Note: `SGD` defaults back to `GD` for CT due to its global nature.
- `--epsilon`: Target accuracy for the bilevel optimization (e.g., `5e-1`). Controls the MLMC level schedule.
- `--upper_lr`: Learning rate for the upper-level optimization (defaults to `2e-3`, adjust for PDEs).
- `--num_epochs`: Number of epochs (or outer iterations for PDE).
- `--lower_step`: Step size for the lower-level solver.

### Example Runs

**1. Image Denoising with Total Variation (TV)**
```bash
python train.py --problem Denoising --regulariser TV --lower_alg SGD --epsilon 0.5 --num_epochs 10
```

**2. CT Reconstruction with CT_TV Regularizer**
Use Gradient Descent (`GD`) or `FISTA` for CT, as patch-based SGD is not applicable to the Radon transform.
```bash
python train.py --problem CT --regulariser CT_TV --lower_alg FISTA --epsilon 0.5 --num_epochs 10
```

**3. 2D Elliptic PDE Evaluation**
This experiment learns both the regularization parameter ($\lambda$) and optimal observation locations via a softmax policy. Adjust the upper learning rate to ensure stable convergence.
```bash
python train.py --problem PDE --lower_alg GD --upper_lr 0.1 --num_epochs 25
```

## Outputs and Logging

During execution, standard output provides iteration/epoch progress alongside metrics like upper level Loss, Validation PSNR (or Validation Loss for PDE), and current learned parameters.

- **Image Reconstructions**: Periodically, visual plots showing the Ground Truth, Noisy Observation, and Current Reconstruction are saved directly to the `logs/` directory.
- **PDE Overlays**: For the PDE problem, a 3-panel plot containing the 2D field, reconstruction, and current sensor locations over the absolute error map is saved to `logs/pde_fields_iter_k.png`.
- **Checkpoints**: The model parameters, loss histories, and level configurations are saved as `.pt` files within the `logs/` directory for downstream use.

## Inference and Plotting

After training, you can evaluate the models and generate plots.

**Run Inference:**
```bash
python inference.py --problem Denoising --regulariser TV --lower_alg SGD
```

**Generate Metric Plots:**
`log_plot.py` will automatically read the `.pt` files in the `logs/` folder and generate comparison charts (Cost vs. Loss/PSNR).
```bash
python log_plot.py
```
