#%%
import os, glob
import torch
import matplotlib.pyplot as plt

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
    logs = torch.load(file, map_location="cpu")
    logs["file"] = file
    all_logs.append(logs)


def make_label(logs, show_upper_alg=False, show_lr=False):
    """Build a clean legend label from log metadata."""
    parts = []
    if logs.get("problem"):
        parts.append(f"{logs['problem']}")
    if logs.get("algo") and show_upper_alg:
        parts.append(f"{logs['algo']}")
    if logs.get("epsilon") is not None:
        parts.append(rf"$\epsilon={logs['epsilon']}$")
    if "ML" in logs:
        if logs['ML']:
            parts.append(f"ML")
        else:
            parts.append(f"SL")
    if logs.get("lr") is not None and show_lr:
        parts.append(f"lr={logs['lr']}")
    if logs.get("L") is not None:
        parts.append(f"L={logs['L']}")
    return ", ".join(parts)


def plot_metric(all_logs, metric, xlabel, ylabel, title, filename,
                yscale="linear", xlim=None, ylim=None, filter_problem=None):
    """Generic plotting function with optional problem filtering."""
    plt.rcParams.update({"font.size": 20})
    plt.rcParams.update({'lines.linewidth': 1.5})
    color_palette_ML = plt.get_cmap("tab10")
    color_palette_SL = plt.get_cmap("tab10")
    line_styles = ["-", "dotted"]
    ML_counter = 0
    SL_counter = 0
    plt.figure(figsize=(12, 8))

    for logs in all_logs:
        # Filter by problem type if specified
        if filter_problem and logs.get("problem") != filter_problem:
            continue
        # Skip logs that don't have the requested metric
        if metric not in logs or "cost" not in logs:
            continue

        label = make_label(logs)
        x_data = logs["cost"]
        y_data = logs[metric]

        # Align lengths
        min_len = min(len(x_data), len(y_data))
        x_data = x_data[:min_len]
        y_data = y_data[:min_len]

        plt.plot(x_data, y_data, label=label,
                 color=color_palette_ML(ML_counter) if logs.get("ML") else color_palette_SL(SL_counter),
                 linestyle=line_styles[0] if logs.get("ML") else line_styles[1])
        if logs.get("ML"):
            ML_counter += 1
        else:
            SL_counter += 1

    plt.xscale("symlog")
    if yscale != "linear":
        plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(fontsize=16, loc='best')
    plt.grid(True, alpha=0.9)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    plt.savefig(os.path.join("logs", filename), bbox_inches="tight", dpi=300)
    plt.show()


# -- Determine unique problem types
problem_types = set()
for logs in all_logs:
    if "problem" in logs:
        problem_types.add(logs["problem"])

if not problem_types:
    problem_types = {None}  # fallback: plot all

for problem in problem_types:
    suffix = f"_{problem}" if problem else ""

    plot_metric(all_logs, "loss", "Total computational cost", "Training Loss",
                f"Training Loss {problem or ''}",
                f"loss_bilevel_monte_carlo{suffix}.png",
                yscale="log", xlim=(1e2, 1e8), filter_problem=problem)

    plot_metric(all_logs, "psnr", "Total computational cost", "Training PSNR",
                f"Training PSNR {problem or ''}",
                f"psnr_bilevel_monte_carlo{suffix}.png",
                xlim=(1e2, 1e8), filter_problem=problem)

    plot_metric(all_logs, "val_psnr", "Total computational cost", "Validation PSNR",
                f"Validation PSNR {problem or ''}",
                f"val_psnr_bilevel_monte_carlo{suffix}.png",
                xlim=(1e2, 1e8), filter_problem=problem)

    # PDE-specific: validation loss
    if problem == "PDE":
        plot_metric(all_logs, "val_loss", "Total computational cost", "Validation Loss",
                    "Validation Loss (PDE)",
                    f"val_loss_bilevel_monte_carlo_PDE.png",
                    yscale="log", filter_problem=problem)
#%%