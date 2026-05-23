import os, glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-ready aesthetic parameters
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams.update({'lines.linewidth': 2.5})

# -- Load the logs
logs_files = glob.glob("logs/*.pt")
all_logs = []

def extract_sort_key(filename):
    base = os.path.basename(filename)
    key_str = base.split("_")[-1].replace(".pt", "")
    try:
        return float(key_str)
    except ValueError:
        return key_str

logs_files = sorted(logs_files, key=extract_sort_key)

for file in logs_files:
    logs = torch.load(file, map_location="cpu", weights_only=False)
    logs["file"] = os.path.basename(file)
    all_logs.append(logs)


def make_label(logs, show_upper_alg=False, show_lr=False):
    """Build a clean legend label from log metadata."""
    parts = []
    if logs.get("problem"):
        parts.append(f"{logs['problem']}")
    if logs.get("reg"):
        parts.append(f"{logs['reg']}")
    if logs.get("algo") and show_upper_alg:
        parts.append(f"{logs['algo']}")
    if logs.get("epsilon") is not None:
        parts.append(rf"$\epsilon={logs['epsilon']}$")
    if "ML" in logs:
        parts.append("MLMC" if logs['ML'] else "SLMC")
    if logs.get("lr") is not None and show_lr:
        parts.append(f"lr={logs['lr']}")
    if logs.get("L") is not None:
        parts.append(f"L={logs['L']}")
    return ", ".join(parts)


def rolling_average(data, window_size):
    """Computes a simple moving average over the given window size."""
    if len(data) < window_size:
        return data
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def metric_x_data(logs, metric, use_epochs):
    """Return an x-axis aligned with the cadence at which a metric is logged."""
    metric_len = len(logs.get(metric, []))
    if use_epochs:
        return np.arange(metric_len)

    costs = logs.get("cost", [])
    if not costs:
        return None

    # Validation metrics and parameter histories are logged once per epoch, while
    # training metrics are logged once per batch. Align validation points with
    # the cost reached at the corresponding epoch boundary.
    if metric.startswith("val_") and len(costs) != metric_len and metric_len > 1:
        stride = (len(costs) - 1) / (metric_len - 1)
        indices = [min(int(round(i * stride)), len(costs) - 1) for i in range(metric_len)]
        return [costs[i] for i in indices]

    return costs


def plot_metric(all_logs, metric, xlabel, ylabel, title, filename,
                yscale="linear", xlim=None, ylim=None, filter_problem=None, use_epochs=False, window_size=1):
    """Generic plotting function with optional problem filtering."""
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10")

    ML_counter = 0
    SL_counter = 0
    plotted = False

    for logs in all_logs:
        if filter_problem and logs.get("problem") != filter_problem:
            continue
        if metric not in logs or len(logs[metric]) == 0:
            continue

        y_data = logs[metric]
        x_data = metric_x_data(logs, metric, use_epochs)
        if x_data is None:
            continue

        # Align lengths
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]

        if window_size > 1 and len(y_data) >= window_size:
            y_data = rolling_average(y_data, window_size)
            x_data = x_data[window_size - 1:] # align x-axis with the smoothed data

        label = make_label(logs)
        is_ml = logs.get("ML", False)
        color = palette[ML_counter % len(palette)] if is_ml else palette[(SL_counter+5) % len(palette)]
        linestyle = "-" if is_ml else "--"

        plt.plot(x_data, y_data, label=label, color=color, linestyle=linestyle, alpha=0.9)

        if is_ml:
            ML_counter += 1
        else:
            SL_counter += 1
        plotted = True

    if not plotted:
        plt.close()
        return

    # Log scaling
    if not use_epochs:
        plt.xscale("symlog")

    if yscale != "linear":
        plt.yscale(yscale)

    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(loc='best', frameon=True, shadow=True)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.tight_layout()
    os.makedirs("plot_figures", exist_ok=True)

    # Save as PNG and PDF (Vector graphic for LaTeX)
    plt.savefig(os.path.join("plot_figures", filename), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join("plot_figures", filename.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close()


def plot_parameters(all_logs, filter_problem):
    """Plots the evolution of hyperparameters."""
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab20")
    color_idx = 0
    plotted = False

    for logs in all_logs:
        if filter_problem and logs.get("problem") != filter_problem:
            continue

        label_base = make_label(logs)

        if "param_history" in logs and len(logs["param_history"]) > 0:
            # Imaging problems (CT, Denoising)
            history = logs["param_history"]
            keys = history[0].keys()

            if logs.get("cost") and len(logs["cost"]) != len(history) and len(history) > 1:
                stride = (len(logs["cost"]) - 1) / (len(history) - 1)
                indices = [min(int(round(i * stride)), len(logs["cost"]) - 1) for i in range(len(history))]
                x_data = [logs["cost"][i] for i in indices]
            else:
                x_data = logs.get("cost", np.arange(len(history)))
            # Align
            min_len = min(len(x_data), len(history))
            x_data = x_data[:min_len]
            history = history[:min_len]

            for k in keys:
                # Extract scalar parameters (e.g. log_tv_weight, log_smoothing)
                v_example = history[0][k]
                if np.size(v_example) == 1:
                    y_data = [h[k].item() if hasattr(h[k], 'item') else float(h[k]) for h in history]
                    plt.plot(x_data, y_data, label=f"{label_base}: {k}", color=palette[color_idx % len(palette)])
                    color_idx += 1
                    plotted = True

        elif "lambda_history" in logs and len(logs["lambda_history"]) > 0:
            # PDE problems
            history = logs["lambda_history"]
            x_data = logs.get("cost", np.arange(len(history)))
            min_len = min(len(x_data), len(history))
            x_data = x_data[:min_len]
            y_data = history[:min_len]

            plt.plot(x_data, y_data, label=f"{label_base}: log10(lambda)", color=palette[color_idx % len(palette)], marker="o", markersize=4)
            color_idx += 1
            plotted = True

    if not plotted:
        plt.close()
        return

    plt.xscale("symlog")
    plt.xlabel("Total Computational Cost", fontweight='bold')
    plt.ylabel("Parameter Value", fontweight='bold')
    plt.title(f"Hyperparameter Evolution ({filter_problem})", fontweight='bold', pad=15)

    # Put legend outside nicely
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, shadow=True)

    plt.tight_layout()
    os.makedirs("plot_figures", exist_ok=True)

    filename = f"params_evolution_{filter_problem}.png"
    plt.savefig(os.path.join("plot_figures", filename), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join("plot_figures", filename.replace(".png", ".pdf")), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # -- Determine unique problem types
    problem_types = {logs.get("problem") for logs in all_logs if "problem" in logs}
    if not problem_types:
        problem_types = {None}  # fallback: plot all

    metrics_map = {
        "loss": ("Training Loss", "log", None),
        "psnr": ("Training PSNR", "linear", None),
        "ssim": ("Training SSIM", "linear", None),
        "val_psnr": ("Validation PSNR", "linear", None),
        "val_ssim": ("Validation SSIM", "linear", None),
        "val_loss": ("Validation Loss", "log", None)
    }

    for problem in problem_types:
        suffix = f"_{problem}" if problem else ""

        for metric_key, (ylab, yscl, ylim) in metrics_map.items():
            window = 5 if problem == "PDE" or metric_key in ["loss", "psnr", "ssim"] else 1
            # Plot vs Cost
            plot_metric(all_logs, metric_key, "Total Computational Cost", ylab,
                        f"{ylab} vs Cost ({problem or ''})",
                        f"{metric_key}_vs_cost{suffix}.png",
                        yscale=yscl, ylim=ylim, filter_problem=problem, use_epochs=False, window_size=window)

            # Plot vs Epochs (for convergence comparisons)
            plot_metric(all_logs, metric_key, "Epoch / Iteration Step", ylab,
                        f"{ylab} vs Epochs ({problem or ''})",
                        f"{metric_key}_vs_epoch{suffix}.png",
                        yscale=yscl, ylim=ylim, filter_problem=problem, use_epochs=True, window_size=window)

        # Plot Parameter evolution
        plot_parameters(all_logs, filter_problem=problem)

    print("Plotting complete. Vector PDFs and PNGs saved to 'plot_figures/' directory.")
