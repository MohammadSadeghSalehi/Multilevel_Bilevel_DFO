import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "figs" / "TV"

EPSILONS = [1.0, 0.5, 0.1]
FIG_STEM = "ct_tv_eps_0p1_0p5_1"
ESTIMATORS = [
    (True, "MLMC", "-"),
    (False, "SLMC", "--"),
]


def load_log(epsilon, is_ml):
    path = LOG_DIR / f"logs_CT_TV_CT_LoDo_FISTA_{is_ml}_{epsilon}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    log = torch.load(path, map_location="cpu", weights_only=False)
    log["path"] = path
    return log


def aligned_x(log, metric):
    values = log.get(metric, [])
    costs = log.get("cost", [])
    if not values or not costs:
        return np.array([]), np.array([])

    values = np.asarray(values, dtype=float)
    costs = np.asarray(costs, dtype=float)

    if metric.startswith("val_") and len(values) != len(costs) and len(values) > 1:
        stride = (len(costs) - 1) / (len(values) - 1)
        idx = [min(int(round(i * stride)), len(costs) - 1) for i in range(len(values))]
        x = costs[idx]
    else:
        n = min(len(values), len(costs))
        values = values[:n]
        x = costs[:n]

    mask = x > 0
    return x[mask], values[mask]


def moving_average(values, window=5):
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def moving_average_xy(x, y, window=5):
    if window <= 1 or len(y) < window:
        return x, y
    return x[window - 1 :], moving_average(y, window)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "figure.dpi": 160,
            "savefig.dpi": 400,
        }
    )


def save_figure(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def format_cost_axis(ax, xscale):
    if xscale == "log":
        tick_values = np.array([0.1, 0.2, 0.5, 1, 2, 5, 10]) * 1e6
        ax.set_xscale("log")
        ax.set_xlim(5e4, 1.1e7)
        ax.set_xticks(tick_values)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:g}"))
        ax.annotate(
            r"$\times 10^6$",
            xy=(1.0, 0.0),
            xycoords="axes fraction",
            xytext=(4, -16),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=10,
        )
        return

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((6, 6))
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.xaxis.get_offset_text().set_fontsize(7)
    ax.set_xlim(left=0)


def make_metric_figure(logs, xscale="linear"):
    colors = {1.0: "#1f77b4", 0.5: "#009e73", 0.1: "#d55e00"}
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), constrained_layout=True)

    metric_specs = [
        ("loss", "Training objective", 7),
        ("val_psnr", "Validation PSNR (dB)", 1),
        ("val_ssim", "Validation SSIM", 1),
    ]

    for ax, (metric, ylabel, window) in zip(axes, metric_specs):
        for eps in EPSILONS:
            for is_ml, est_name, linestyle in ESTIMATORS:
                log = logs[(eps, is_ml)]
                x, y = aligned_x(log, metric)
                if len(x) == 0:
                    continue
                x, y = moving_average_xy(x, y, window=window)
                label = rf"{est_name}, $\varepsilon={eps:g}$"
                marker = "o" if metric.startswith("val_") else None
                ax.plot(
                    x,
                    y,
                    linestyle=linestyle,
                    color=colors[eps],
                    label=label,
                    marker=marker,
                    markersize=2.6,
                    markevery=max(1, len(x) // 6),
                )

        format_cost_axis(ax, xscale)
        ax.set_xlabel("Cost")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", alpha=0.25)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16))
    suffix = "" if xscale == "linear" else f"_{xscale}x"
    save_figure(fig, f"{FIG_STEM}_metrics{suffix}")


def crop_panel(image_path, panel_index):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    panel_w = w / 3.0
    # The training figure is a 3-panel plot. These margins isolate the image
    # body while dropping subplot titles and most whitespace.
    x0 = int(panel_index * panel_w + 0.075 * panel_w)
    x1 = int((panel_index + 1) * panel_w - 0.075 * panel_w)
    y0 = int(0.19 * h)
    y1 = int(0.90 * h)
    crop = image.crop((x0, y0, x1, y1))
    side = min(crop.size)
    cx, cy = crop.size[0] // 2, crop.size[1] // 2
    half = side // 2
    return crop.crop((cx - half, cy - half, cx + half, cy + half))


def final_metric(log, key):
    values = log.get(key, [])
    return float(values[-1]) if values else float("nan")


def make_reconstruction_figure(logs):
    source = LOG_DIR / "reconstructed_image_CT_TV_CT_LoDo_FISTA_True_1.0.png"
    gt_crop = crop_panel(source, 0)
    fbp_crop = crop_panel(source, 1)

    def show_panel(ax, title, image):
        ax.imshow(image)
        ax.set_title(title, fontsize=9, pad=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig = plt.figure(figsize=(7.3, 3.35), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.12, 0.02, 1.12, 1.12, 1.12],
        left=0.006,
        right=0.998,
        top=0.895,
        bottom=0.012,
        hspace=0.22,
        wspace=0.03,
    )

    show_panel(fig.add_subplot(grid[0, 0]), "Ground truth", gt_crop)
    show_panel(fig.add_subplot(grid[1, 0]), "FBP", fbp_crop)

    sep_ax = fig.add_subplot(grid[:, 1])
    sep_ax.axis("off")
    sep_ax.plot([0.5, 0.5], [0.06, 0.94], color="0.78", lw=0.8, transform=sep_ax.transAxes)

    for col, eps in enumerate(EPSILONS):
        for row, (is_ml, est_name, _) in enumerate([(False, "SLMC", "--"), (True, "MLMC", "-")]):
            img_path = LOG_DIR / f"reconstructed_image_CT_TV_CT_LoDo_FISTA_{is_ml}_{eps}.png"
            crop = crop_panel(img_path, 2)
            log = logs[(eps, is_ml)]
            psnr = final_metric(log, "val_psnr")
            ssim = final_metric(log, "val_ssim")
            title = rf"{est_name}, $\varepsilon={eps:g}$" + f"\n{psnr:.2f} dB / {ssim:.3f}"
            show_panel(fig.add_subplot(grid[row, col + 2]), title, crop)

    save_figure(fig, f"{FIG_STEM}_reconstructions")


def make_summary_table(logs):
    lines = ["epsilon,estimator,final_cost,final_val_psnr,final_val_ssim,best_val_psnr,best_val_ssim"]
    for eps in EPSILONS:
        for is_ml, est_name, _ in ESTIMATORS:
            log = logs[(eps, is_ml)]
            val_psnr = np.asarray(log.get("val_psnr", []), dtype=float)
            val_ssim = np.asarray(log.get("val_ssim", []), dtype=float)
            cost = log.get("cost", [float("nan")])[-1]
            lines.append(
                f"{eps},{est_name},{cost},{val_psnr[-1]:.6f},{val_ssim[-1]:.6f},"
                f"{np.max(val_psnr):.6f},{np.max(val_ssim):.6f}"
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{FIG_STEM}_summary.csv").write_text("\n".join(lines) + "\n")


def main():
    setup_style()
    logs = {(eps, is_ml): load_log(eps, is_ml) for eps in EPSILONS for is_ml, _, _ in ESTIMATORS}
    make_metric_figure(logs)
    make_metric_figure(logs, xscale="log")
    make_reconstruction_figure(logs)
    make_summary_table(logs)
    print(f"Saved paper figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
