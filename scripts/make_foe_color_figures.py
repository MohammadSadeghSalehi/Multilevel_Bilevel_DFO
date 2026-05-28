from __future__ import annotations

import argparse
import math
import pickle
import re
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from PIL import Image


class _FakeStorageType:
    pass


class _FakeStorage:
    def __init__(self, persistent_id):
        self.persistent_id = persistent_id


class _FakeTensor:
    def __init__(self, storage, storage_offset, size, stride, requires_grad, backward_hooks):
        self.storage = storage
        self.storage_offset = storage_offset
        self.size = tuple(size)
        self.stride = tuple(stride)
        self.requires_grad = requires_grad
        self.backward_hooks = backward_hooks

    def item(self):
        return float("nan")


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return _FakeTensor(storage, storage_offset, size, stride, requires_grad, backward_hooks)


class TorchlessUnpickler(pickle.Unpickler):
    """Load metric dictionaries from torch-saved zip files without importing torch."""

    def find_class(self, module, name):
        if module == "torch" and name.endswith("Storage"):
            return _FakeStorageType
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        return super().find_class(module, name)

    def persistent_load(self, persistent_id):
        return _FakeStorage(persistent_id)


def load_torch_zip(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as zf:
        data_name = next(name for name in zf.namelist() if name.endswith("/data.pkl"))
        with zf.open(data_name, "r") as f:
            return TorchlessUnpickler(f).load()


def parse_log_name(path: Path) -> tuple[bool, float]:
    match = re.search(r"SGD_(True|False)_([0-9.]+)_eps1_full_sgd_k4_iter1000", path.name)
    if not match:
        raise ValueError(f"Cannot parse estimator/epsilon from {path.name}")
    return match.group(1) == "True", float(match.group(2))


def aligned_x(log: dict, metric: str) -> np.ndarray:
    y_len = len(log.get(metric, []))
    costs = np.asarray(log.get("cost", []), dtype=float)
    if y_len == 0:
        return np.asarray([])
    if costs.size == 0:
        return np.arange(y_len, dtype=float)
    if len(costs) == y_len:
        return costs
    if y_len == 1:
        return np.asarray([costs[-1]], dtype=float)
    idx = np.rint(np.linspace(0, len(costs) - 1, y_len)).astype(int)
    return costs[idx]


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_runs(log_dir: Path) -> list[dict]:
    runs = []
    pattern = "logs_FoE_ColorDenoising_SGD_*_eps1_full_sgd_k4_iter1000.pt"
    for path in sorted(log_dir.glob(pattern)):
        is_ml, epsilon = parse_log_name(path)
        log = load_torch_zip(path)
        log["path"] = path
        log["is_ml"] = is_ml
        log["epsilon"] = epsilon
        runs.append(log)
    return sorted(runs, key=lambda r: (r["epsilon"], not r["is_ml"]))


def make_metrics_figure(runs: list[dict], out_base: Path) -> None:
    metric_style = {
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

    def format_cost_axis(ax):
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

    with plt.rc_context(metric_style):
        colors = {1.0: "#1f77b4", 0.5: "#009e73", 0.1: "#d55e00"}
        labels = {True: "MLMC", False: "SLMC"}
        styles = {True: "-", False: "--"}
        metrics = [
            ("loss", "Training objective", 7),
            ("val_psnr", "Validation PSNR (dB)", 1),
            ("val_ssim", "Validation SSIM", 1),
        ]

        eps_order = [1.0, 0.5, 0.1]
        run_order = []
        for eps in eps_order:
            run_order.extend(
                sorted(
                    [run for run in runs if run["epsilon"] == eps],
                    key=lambda run: not run["is_ml"],
                )
            )

        fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), constrained_layout=True)
        for ax, (metric, ylabel, window) in zip(axes, metrics):
            for run in run_order:
                y = np.asarray(run.get(metric, []), dtype=float)
                if y.size == 0:
                    continue
                x = aligned_x(run, metric)
                min_len = min(len(x), len(y))
                x = x[:min_len]
                y = y[:min_len]
                if window > 1 and len(y) >= window:
                    y = moving_average(y, window)
                    x = x[window - 1 :]

                mask = x > 0
                x = x[mask]
                y = y[mask]
                if len(x) == 0:
                    continue

                eps = run["epsilon"]
                label = rf"{labels[run['is_ml']]}, $\varepsilon={eps:g}$"
                marker = "o" if metric.startswith("val_") else None
                ax.plot(
                    x,
                    y,
                    linestyle=styles[run["is_ml"]],
                    color=colors.get(eps, "black"),
                    label=label,
                    marker=marker,
                    markersize=2.6,
                    markevery=max(1, len(x) // 6),
                )

            format_cost_axis(ax)
            ax.set_xlabel("Cost")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", alpha=0.25)

        handles, labels_seen = axes[1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels_seen,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 1.16),
        )
        fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
        plt.close(fig)


def crop_panel(path: Path, crop_index: int, crop_count: int = 3) -> Image.Image:
    image = Image.open(path).convert("RGB")
    w, h = image.size
    panel_w = w // crop_count
    left = crop_index * panel_w
    right = w if crop_index == crop_count - 1 else (crop_index + 1) * panel_w
    panel = image.crop((left, 0, right, h))
    # Remove matplotlib whitespace and titles; the image rectangle is the
    # largest region with dense non-white pixels.
    arr = np.asarray(panel)
    mask = np.any(arr < 245, axis=2)
    row_density = mask.sum(axis=1)
    col_density = mask.sum(axis=0)
    dense_rows = np.where(row_density > 0.25 * panel.size[0])[0]
    dense_cols = np.where(col_density > 0.25 * panel.size[1])[0]
    if dense_rows.size and dense_cols.size:
        def longest_run(values: np.ndarray) -> np.ndarray:
            breaks = np.where(np.diff(values) > 1)[0] + 1
            runs = np.split(values, breaks)
            return max(runs, key=len)

        dense_rows = longest_run(dense_rows)
        dense_cols = longest_run(dense_cols)
        pad = 8
        left = max(dense_cols.min() - pad, 0)
        right = min(dense_cols.max() + pad + 1, panel.size[0])
        top = max(dense_rows.min() - pad, 0)
        bottom = min(dense_rows.max() + pad + 1, panel.size[1])
        panel = panel.crop((left, top, right, bottom))
    return panel


def make_reconstruction_figure(log_dir: Path, out_base: Path) -> None:
    epsilons = [1.0, 0.5, 0.1]
    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.8), constrained_layout=True)

    first_path = log_dir / "reconstructed_image_FoE_ColorDenoising_SGD_True_1.0_eps1_full_sgd_k4_iter1000.png"
    clean = crop_panel(first_path, 0)
    noisy = crop_panel(first_path, 1)

    for row in range(2):
        for col in range(4):
            axes[row, col].axis("off")

    axes[0, 0].imshow(clean)
    axes[0, 0].set_title("Clean", fontsize=10)
    axes[1, 0].imshow(noisy)
    axes[1, 0].set_title("Noisy", fontsize=10)

    for col, eps in enumerate(epsilons, start=1):
        for row, is_ml in enumerate([False, True]):
            path = log_dir / (
                f"reconstructed_image_FoE_ColorDenoising_SGD_{is_ml}_{eps:.1f}"
                "_eps1_full_sgd_k4_iter1000.png"
            )
            recon = crop_panel(path, 2)
            axes[row, col].imshow(recon)
            axes[row, col].set_title(
                rf"$\varepsilon={eps:g}$ {'MLMC' if is_ml else 'SLMC'}",
                fontsize=10,
            )

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_summary(runs: list[dict], out_path: Path) -> None:
    rows = []
    for run in runs:
        cost = np.asarray(run.get("cost", []), dtype=float)
        val_psnr = np.asarray(run.get("val_psnr", []), dtype=float)
        val_ssim = np.asarray(run.get("val_ssim", []), dtype=float)
        rows.append(
            {
                "epsilon": run["epsilon"],
                "estimator": "MLMC" if run["is_ml"] else "SLMC",
                "final_cost": float(cost[-1]) if cost.size else math.nan,
                "final_psnr": float(val_psnr[-1]) if val_psnr.size else math.nan,
                "final_ssim": float(val_ssim[-1]) if val_ssim.size else math.nan,
                "best_psnr": float(val_psnr.max()) if val_psnr.size else math.nan,
                "best_ssim": float(val_ssim.max()) if val_ssim.size else math.nan,
                "epochs": len(val_psnr),
                "train_points": len(run.get("loss", [])),
                "img_size": run.get("img_size", ""),
                "num_channels": run.get("num_channels", ""),
                "foe_num_experts": run.get("foe_num_experts", ""),
                "freeze_foe_kernels": run.get("freeze_foe_kernels", ""),
                "upper_directions": run.get("upper_directions", ""),
                "lower_max_iter": run.get("lower_max_iter", ""),
                "max_lower_iter": ";".join(str(v) for v in run.get("max_lower_iter", [])),
                "N_l": ";".join(str(v) for v in run.get("N_l", [])),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(
            "epsilon,estimator,final_cost,final_psnr,final_ssim,best_psnr,best_ssim,"
            "epochs,train_points,img_size,num_channels,foe_num_experts,freeze_foe_kernels,"
            "upper_directions,lower_max_iter,max_lower_iter,N_l\n"
        )
        for row in rows:
            f.write(
                "{epsilon:g},{estimator},{final_cost:.0f},{final_psnr:.6f},"
                "{final_ssim:.6f},{best_psnr:.6f},{best_ssim:.6f},{epochs},{train_points},"
                "{img_size},{num_channels},{foe_num_experts},{freeze_foe_kernels},"
                "{upper_directions},{lower_max_iter},{max_lower_iter},{N_l}\n".format(
                    **row
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out-dir", default="figs/TV")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(log_dir)
    if len(runs) != 6:
        raise RuntimeError(f"Expected 6 FoE color runs, found {len(runs)}")

    make_metrics_figure(runs, out_dir / "color_foe_eps_0p1_0p5_1_metrics_logx")
    make_reconstruction_figure(log_dir, out_dir / "color_foe_eps_0p1_0p5_1_reconstructions")
    write_summary(runs, out_dir / "color_foe_eps_0p1_0p5_1_summary.csv")

    print("Saved:")
    print(out_dir / "color_foe_eps_0p1_0p5_1_metrics_logx.pdf")
    print(out_dir / "color_foe_eps_0p1_0p5_1_reconstructions.pdf")
    print(out_dir / "color_foe_eps_0p1_0p5_1_summary.csv")


if __name__ == "__main__":
    main()
