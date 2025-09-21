#%%
import os, glob
import torch
import matplotlib.pyplot as plt

# -- Load the logs
logs_files = glob.glob("logs/*.pt")
all_logs = []
for file in logs_files:
    logs = torch.load(file, map_location="cpu")
    logs["file"] = file
    all_logs.append(logs)

def make_label(logs):
    """Build a clean legend label from log metadata."""
    parts = []
    if logs.get("algo"):
        parts.append(f"{logs['algo']}")
    # if logs.get("reg"):
    #     parts.append(f"reg={logs['reg']}")
    # if logs.get("problem"):
    #     parts.append(f"problem={logs['problem']}")
    if "ML" in logs:
        parts.append(f"ML={logs['ML']}")
    if logs.get("lr") is not None:
        parts.append(f"lr={logs['lr']}")
    if logs.get("epsilon") is not None:
        parts.append(f"eps={logs['epsilon']}")
    if logs.get("L") is not None:
        parts.append(f"L={logs['L']}")
    return ", ".join(parts)

def plot_metric(all_logs, metric, ylabel, title, filename, yscale="linear", xlim=None, ylim = None):
    """Generic plotting function."""
    # set font size
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(10, 8))
    for logs in all_logs:
        label = make_label(logs)
        plt.plot(logs["cost"], logs[metric], label=label)
    plt.xscale("symlog")
    if yscale != "linear":
        plt.yscale(yscale)
    plt.xlabel("Total Cost (Iterations)")
    plt.ylabel(ylabel)  
    plt.title(title)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.5)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

# -- Generate plots
plot_metric(all_logs, "loss", "Loss", 
            "Training Loss", 
            "loss_bilevel_monte_carlo.png", yscale="log", xlim=(1e2, None))

plot_metric(all_logs, "psnr", "PSNR", 
            "Training PSNR ", 
            "psnr_bilevel_monte_carlo.png", xlim=(1e2, None))

plot_metric(all_logs, "val_psnr", "Validation PSNR", 
            "Validation PSNR", 
            "val_psnr_bilevel_monte_carlo.png", xlim=(1e2, None))
#%%