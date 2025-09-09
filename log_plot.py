
 #%%
# from matplotlib import pyplot as plt
# import torch
# import os, glob
# # -- Load the logs and plot the result
# logs_files = glob.glob("logs/*.pt")
# all_logs = []
# for file in logs_files:
#     logs = torch.load(file, map_location="cpu")
#     logs["file"] = file
#     all_logs.append(logs)

# plt.figure(figsize=(10, 5))
# for logs in all_logs:
#     label = logs.get("reg", "") + "_" + logs.get("problem", "") + "_" + str(logs.get("ML", "")) + "_" 
#     plt.plot(logs["cost"], logs["loss"], label=f"{os.path.basename(logs['file'])[5:-3]} Loss")
#     plt.xlim(1e2,None)
# plt.xscale("symlog")
# plt.yscale("log")
# plt.xlabel("Total Cost (Iterations)")
# plt.ylabel("Loss")
# plt.title("Loss during Bilevel Monte Carlo Optimization (All Settings)")
# plt.legend()
# plt.grid()
# plt.savefig("loss_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
# plt.show()

# plt.figure(figsize=(10, 5))
# for logs in all_logs:
#     plt.plot(logs["cost"], logs["psnr"], label=f"{os.path.basename(logs['file'])[5:-3]}")
# plt.xscale("symlog")
# plt.xlabel("Total Cost (Iterations)")
# plt.ylabel("PSNR")
# plt.title("PSNR during Bilevel Monte Carlo Optimization (All Settings)")
# plt.legend()
# plt.grid()
# plt.savefig("psnr_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
# plt.show()

# plt.figure(figsize=(10, 5))
# for logs in all_logs:
#     plt.plot(logs["cost"], logs["val_psnr"], label=f"{os.path.basename(logs['file'])[5:-3]}")
# plt.xscale("symlog")
# plt.xlabel("Total Cost (Iterations)")
# plt.ylabel("Validation PSNR")
# plt.title("Validation PSNR during Bilevel Monte Carlo Optimization (All Settings)")
# plt.legend()
# plt.grid()
# plt.savefig("val_psnr_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
# plt.show()

# # plotting the results
# # plt.figure(figsize=(10, 5))
# # plt.plot(logs_dict["cost"], logs_dict["loss"], label="Loss")
# # plt.xscale("linear")
# # plt.yscale("log")
# # plt.xlabel("Total Iterations")
# # plt.ylabel("Loss")
# # plt.title("Loss during Bilevel Monte Carlo Optimization")
# # plt.legend()
# # plt.grid()
# # plt.show()

# # plt.figure(figsize=(10, 5))
# # plt.plot(logs_dict["cost"], logs_dict["psnr"], label="PSNR", color="orange")
# # plt.xscale("linear")
# # plt.xlabel("Total Iterations")
# # plt.ylabel("PSNR")
# # plt.title("PSNR during Bilevel Monte Carlo Optimization")
# # plt.legend()
# # plt.grid()
# # plt.show()

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

def plot_metric(all_logs, metric, ylabel, title, filename, yscale="linear"):
    """Generic plotting function."""
    # set font size
    plt.rcParams.update({"font.size": 14})
    plt.figure(figsize=(8, 6))
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
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.show()

# -- Generate plots
plot_metric(all_logs, "loss", "Loss", 
            "Loss during Bilevel Monte Carlo Optimization", 
            "loss_bilevel_monte_carlo.png", yscale="log")

plot_metric(all_logs, "psnr", "PSNR", 
            "PSNR during Bilevel Monte Carlo Optimization", 
            "psnr_bilevel_monte_carlo.png")

plot_metric(all_logs, "val_psnr", "Validation PSNR", 
            "Validation PSNR", 
            "val_psnr_bilevel_monte_carlo.png")
#%%