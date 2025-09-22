#%%
import os, glob
import torch
import matplotlib.pyplot as plt

# -- Load the logs
logs_files = glob.glob("logs/*.pt")
all_logs = []
# Sort log files based on the value before .pt (e.g., 0.5 in logs_TV_Denoising_SGD_False_0.5.pt)
def extract_sort_key(filename):
    base = os.path.basename(filename)
    key_str = base.split("_")[-1].replace(".pt", "")
    try:
        return float(key_str)
    except ValueError:
        return key_str  # fallback if not a float

logs_files = sorted(logs_files, key=extract_sort_key)

for file in logs_files:
    logs = torch.load(file, map_location="cpu")
    logs["file"] = file
    all_logs.append(logs)

def make_label(logs, show_upper_alg=False, show_lr=False):
    """Build a clean legend label from log metadata."""
    parts = []
    if logs.get("algo") and show_upper_alg:
        parts.append(f"{logs['algo']}")
    # if logs.get("reg"):
    #     parts.append(f"reg={logs['reg']}")
    # if logs.get("problem"):
    #     parts.append(f"problem={logs['problem']}")
    if logs.get("epsilon") is not None:
        parts.append(rf"$\epsilon={logs['epsilon']}$")
    if "ML" in logs:
        if logs['ML']:
            parts.append(f"ML")
    if logs.get("lr") is not None and show_lr:
        parts.append(f"lr={logs['lr']}")
    if logs.get("L") is not None:
        parts.append(f"L={logs['L']}")
    return ", ".join(parts)

def plot_metric(all_logs, metric, xlabel, ylabel, title, filename, yscale="linear", xlim=None, ylim = None):
    """Generic plotting function."""
    # set font size
    plt.rcParams.update({"font.size": 20})
    plt.rcParams.update({'lines.linewidth': 1.5})
    color_palette_ML = plt.get_cmap("tab10")
    color_palette_SL = plt.get_cmap("tab10")
    line_styles = ["-", "dotted"]
    ML_counter = 0
    SL_counter = 0
    plt.figure(figsize=(12, 8))
    for logs in all_logs:
        label = make_label(logs)
        plt.plot(logs["cost"], logs[metric], label=label, 
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
    # plt.title(title)
    plt.legend(fontsize=20, loc='best')
    plt.grid(True, alpha=0.9)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

# -- Generate plots
plot_metric(all_logs, "loss", "Total computational cost", "Training Loss", 
            "Training Loss", 
            "loss_bilevel_monte_carlo.png", yscale="log", xlim=(1e2, 1e8))

plot_metric(all_logs, "psnr", "Total computational cost", "Training PSNR", 
            "Training PSNR", 
            "psnr_bilevel_monte_carlo.png", xlim=(1e2, 1e8))

plot_metric(all_logs, "val_psnr", "Total computational cost", "Validation PSNR", 
            "Validation PSNR", 
            "val_psnr_bilevel_monte_carlo.png", xlim=(1e2, 1e8))
#%%