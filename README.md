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
- `CT`: Computed Tomography reconstruction with deepinverse phantoms.
- `CT_LoDo`: Higher-resolution Computed Tomography reconstruction using the LoDoPaB-CT dataset.
- `PDE`: 2D Elliptic PDE parameter estimation and optimal sensor placement.

### Available Regularizers (`--regulariser`)
- `TV`: Smoothed Total Variation.
- `FoE`: Field of Experts.
- `CT_TV`: Specialized combined Tikhonov + Smoothed TV for the CT problem.
- `CT_FoE`: Specialized combined Tikhonov + Field of Experts for the CT problem.

### Key Command Line Arguments
- `--lower_alg`: Lower-level solver (`GD`, `SGD`, `FISTA`). Note: `SGD` defaults back to `GD` for CT due to its global nature.
- `--epsilon`: Target accuracy for the bilevel optimization (e.g., `5e-1`). Controls the MLMC level schedule.
- `--upper_lr`: Learning rate for the upper-level optimization (defaults to `2e-3`, adjust for PDEs).
- `--num_epochs`: Number of epochs (or outer iterations for PDE).
- `--lower_step`: Step size for the lower-level solver.
- `--subset`: Fraction of the dataset to use (e.g., `0.1`), specifically for large datasets like `CT_LoDo`.
- `--img_size`: Image size for CT and LoDoPaB-CT experiments.
- `--num_angles`: Number of sparse-view CT projection angles.
- `--noise_level`: Standard deviation of synthetic Gaussian measurement noise.
- `--batch_size`: Training batch size. Use `0` to automatically use the minimum sample pool required by the MLMC/SLMC schedule.
- `--val_batch_size`: Validation batch size. Use `0` for a small automatic validation batch.
- `--val_len`: Number of BSDS validation/test images to keep for denoising and inpainting.
- `--train_batches`: Maximum number of training batches/oracle calls per epoch. Use `0` for the full training loader.
- `--val_batches`: Maximum number of validation batches. Use `0` for the full validation loader.
- `--upper_directions`: Number of Gaussian smoothing directions averaged per level and upper update. Increasing this reduces derivative-free gradient variance at proportional cost.
- `--fista_diagnostics`: Report validation FISTA line-search and restart diagnostics.
- `--fista_mode`: FISTA reconstruction mode. Use `legacy` for the original fixed-step solver, `restart` for fixed-step FISTA with adaptive restart, and `monotone` for full-objective backtracking diagnostics.
- `--lower_max_iter`: Override the finest lower-level iteration count used by MLMC/SLMC training. This is useful for high-resolution CT, where the automatic schedule may be too coarse.
- `--color`: Keep BSDS500 denoising images in RGB instead of converting them to grayscale. Use this with `--regulariser FoE`.
- `--sgd_step_decay`: Step decay for the patch-SGD lower solver (`none`, `sqrt`, or `linear`). For RGB FoE, `sqrt` is usually more stable than a fixed patch step.
- `--sgd_patch_mode`: Spatial mode for the SGD lower solver (`patch`, `full`, or `mask`). Use `full` if random patch updates create border artifacts.
- `--sgd_patch_fraction`: Patch side-length fraction for `patch` mode, or mask density scale for `mask` mode.
- `--freeze_foe_kernels`: Freeze FoE convolution kernels and learn only weights/smoothing. This is recommended for derivative-free RGB FoE runs unless many Gaussian directions are used.
- `--foe_num_experts`: Override the number of FoE convolution experts. For unfrozen RGB FoE, `4` is a useful lower-variance alternative to the default `8`.
- `--run_tag`: Optional suffix appended to log and reconstruction filenames, useful for diagnostics.

### Sample-Aware CT Runs

For CT experiments the MLMC schedule controls how many samples are needed per gradient estimate, while the DataLoader controls how many such gradient estimates are evaluated per epoch.

- `--batch_size 0` keeps each batch only as large as required by the MLMC/SLMC schedule.
- `--train_batches` controls the number of upper-level oracle calls per epoch.
- `--val_batches` controls validation cost and the number of initial PSNR/SSIM lines printed.
- `--upper_directions` controls how many Gaussian perturbation directions are averaged inside each upper-level gradient estimate.

For example, with `--epsilon 10.0`, the FISTA MLMC schedule may require only one sample on each of three levels, so the automatic training batch size is `3`. If `--train_batches` and `--val_batches` are left at `0`, the code still loops over the full loaders.

Quick sparse-view CT smoke test:
```bash
python train.py --problem CT_LoDo --regulariser CT_TV --lower_alg FISTA \
  --epsilon 10.0 --num_epochs 5 --img_size 128 --num_angles 16 --noise_level 0.015 \
  --batch_size 0 --train_batches 1 --val_batches 1
```

More meaningful CT_TV run:
```bash
python train.py --problem CT_LoDo --regulariser CT_TV --lower_alg FISTA \
  --epsilon 1.0 --num_epochs 50 --img_size 128 --num_angles 32 --noise_level 0.02 \
  --batch_size 0 --train_batches 20 --val_batches 3
```

To inspect lower-level FISTA behavior during validation, add:
```bash
--fista_diagnostics
```
This reports the averaged line-search Lipschitz estimate, restart count, backtracking count, and final relative change for the validation reconstructions.

The default `--fista_mode legacy` reproduces the original fixed-step reconstruction behavior. Use `--fista_mode restart` as the first stability check if acceleration appears to create blur or oscillations. The `monotone` mode is stricter and useful diagnostically, but can be overly conservative for the current CT/FoE scaling.

For high-resolution CT, compare the training-level reconstruction quality against validation. If training uses a very coarse finest level, e.g. `[50, 124, 303]`, while validation uses 2000 FISTA iterations, the upper-level gradient can be strongly biased. Use `--lower_max_iter 1000` or `--lower_max_iter 1500` to test whether this lower-level bias is the dominant issue.

### Example Runs

**1. Image Denoising with Total Variation (TV)**
```bash
python train.py --problem Denoising --regulariser TV --lower_alg SGD --epsilon 0.5 --num_epochs 10
```

**2. RGB Image Denoising with Field of Experts (FoE)**
Use `--color` to keep BSDS500 in RGB and initialize FoE with three input channels. GD is useful for the deterministic lower-level rate, while patch-SGD gives the stochastic lower-level comparison.
```bash
python train.py --problem Denoising --regulariser FoE --lower_alg GD \
  --epsilon 0.5 --num_epochs 20 --img_size 64 --batch_size 0 \
  --train_batches 10 --val_len 12 --val_batches 3 --lower_step 0.1 --upper_lr 1e-3 --color

python train.py --problem Denoising --regulariser FoE --lower_alg SGD \
  --epsilon 0.5 --num_epochs 20 --img_size 64 --batch_size 0 \
  --train_batches 10 --val_len 12 --val_batches 3 --lower_step 0.02 --upper_lr 1e-3 \
  --upper_directions 4 --lower_max_iter 1000 --sgd_step_decay sqrt \
  --sgd_patch_mode full --freeze_foe_kernels --color

python train.py --problem Denoising --regulariser FoE --lower_alg SGD \
  --epsilon 1.0 --num_epochs 20 --img_size 128 --batch_size 0 \
  --train_batches 10 --val_len 12 --val_batches 3 --lower_step 0.01 --lower_max_iter 500 \
  --upper_lr 2e-3 --upper_directions 4 --sgd_step_decay sqrt \
  --sgd_patch_mode full --foe_num_experts 4 --color --run_tag full_sgd_e1_k4
```

**3. CT Reconstruction with CT_TV Regularizer**
Use Gradient Descent (`GD`) or `FISTA` for CT, as patch-based SGD is not applicable to the Radon transform.
```bash
python train.py --problem CT --regulariser CT_TV --lower_alg FISTA --epsilon 0.5 --num_epochs 10
```

**4. CT Reconstruction on LoDoPaB Dataset**
Use `CT_LoDo` to leverage the custom LoDoPaB implementation. You can specify what fraction of the dataset to use via the `--subset` flag for faster testing.
```bash
python train.py --problem CT_LoDo --regulariser CT_TV --lower_alg FISTA --epsilon 0.5 --num_epochs 10 --subset 0.1
```

**5. CT Reconstruction with CT_FoE**
The CT Field-of-Experts regularizer is heavier and can be more sensitive than CT_TV, so start with a short smoke test and a smaller upper-level learning rate.
```bash
python train.py --problem CT_LoDo --regulariser CT_FoE --lower_alg FISTA \
  --epsilon 10.0 --num_epochs 2 --img_size 128 --num_angles 16 --noise_level 0.015 \
  --upper_lr 5e-4 --batch_size 0 --train_batches 1 --val_batches 1
```

If the smoke test is stable, use a longer sparse-view run:
```bash
python train.py --problem CT_LoDo --regulariser CT_FoE --lower_alg FISTA \
  --epsilon 1.0 --num_epochs 20 --img_size 128 --num_angles 32 --noise_level 0.02 \
  --upper_lr 5e-4 --batch_size 0 --train_batches 10 --val_batches 2 --upper_directions 2
```

**6. 2D Elliptic PDE Evaluation**
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

For the saved LoDoPaB-CT runs:
```bash
python inference.py --problem CT_LoDo --regulariser CT_TV --lower_alg FISTA --img_size 96 --num_angles 16 --noise_level 0.03
```

Training logs store the final learned regularizer parameters in the `saved_params` field. To visualize a specific validation image from a specific CT/FoE run, pass the exact log file and image index:
```bash
python inference.py --problem CT_LoDo --regulariser CT_FoE --lower_alg FISTA \
  --img_size 256 --num_angles 60 --noise_level 0.05 --max_iter 2000 \
  --image_idx 0 --max_images 1 \
  --log_file logs/logs_CT_FoE_CT_LoDo_FISTA_True_1.0.pt
```

For FISTA runs, inference uses the `fista_mode` saved in the training log. You can override it with `--fista_mode restart` if needed.

**Generate Metric Plots:**
`log_plot.py` will automatically read the `.pt` files in the `logs/` folder and generate comparison charts (Cost vs. Loss/PSNR).
```bash
python log_plot.py
```
