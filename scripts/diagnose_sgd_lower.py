import argparse
import csv
import sys
from pathlib import Path

import torch
from deepinv.loss.metric import PSNR, SSIM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SGD import reconstruct_sgd
from train import device, limited_batches, setup_denoising_experiment


def parse_csv_floats(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_csv_ints(value):
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Sweep patch-SGD lower-solver settings for FoE denoising."
    )
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--color", action="store_true")
    parser.add_argument("--steps", type=str, default="0.03,0.01,0.003")
    parser.add_argument("--iters", type=str, default="200,500,1000")
    parser.add_argument("--decays", type=str, default="sqrt")
    parser.add_argument("--val_batches", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--val_len", type=int, default=8)
    parser.add_argument("--patch_fraction", type=float, default=0.5)
    parser.add_argument(
        "--patch_mode",
        type=str,
        default="patch",
        choices=["patch", "full", "mask"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="figs/TV/color_foe_sgd_lower_sweep.csv",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    exp = setup_denoising_experiment(
        "FoE",
        img_size_x=args.img_size,
        img_size_y=args.img_size,
        batch_size=1,
        val_batch_size=args.val_batch_size,
        test_len=args.val_len,
        color=args.color,
    )

    physics = exp["physics"]
    data_fidelity = exp["data_fidelity"]
    regularizer = exp["regularizer"]
    lmbd = exp["lmbd"]
    psnr = PSNR()
    ssim = SSIM()

    batches = [gt.to(device) for _, gt in limited_batches(exp["val_dataloader"], args.val_batches)]
    steps = parse_csv_floats(args.steps)
    iters = parse_csv_ints(args.iters)
    decays = [v.strip() for v in args.decays.split(",") if v.strip()]

    rows = []
    for decay in decays:
        for step_size in steps:
            for max_iter in iters:
                psnrs = []
                ssims = []
                for batch_idx, gt in enumerate(batches):
                    torch.manual_seed(args.seed + 1000 * batch_idx + max_iter)
                    y = physics(gt)
                    patch_h = max(1, int(round(y.shape[2] * args.patch_fraction)))
                    patch_w = max(1, int(round(y.shape[3] * args.patch_fraction)))
                    x_hat, used_iters = reconstruct_sgd(
                        y=y,
                        physics=physics,
                        data_fidelity=data_fidelity,
                        regularizer=regularizer,
                        lamda=lmbd,
                        step_size=step_size,
                        max_iter=max_iter,
                        tol=0.0,
                        patch_size=(patch_h, patch_w),
                        verbose=False,
                        return_stats=False,
                        step_decay=decay,
                        patch_mode=args.patch_mode,
                        mask_probability=args.patch_fraction * args.patch_fraction,
                    )
                    psnrs.append(psnr(gt, x_hat).mean().item())
                    ssims.append(ssim(gt, x_hat).mean().item())

                row = {
                    "decay": decay,
                    "lower_step": step_size,
                    "max_iter": max_iter,
                    "mean_psnr": sum(psnrs) / len(psnrs),
                    "mean_ssim": sum(ssims) / len(ssims),
                    "val_batches": len(batches),
                    "val_len": args.val_len,
                    "patch_fraction": args.patch_fraction,
                    "patch_mode": args.patch_mode,
                }
                rows.append(row)
                print(
                    f"{decay:>6} step={step_size:g} iters={max_iter:4d} "
                    f"PSNR={row['mean_psnr']:.3f} SSIM={row['mean_ssim']:.4f}"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved sweep to {out_path}")


if __name__ == "__main__":
    main()
