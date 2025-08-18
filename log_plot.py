
#%%
from matplotlib import pyplot as plt
import torch
import os, glob
# -- Load the logs and plot the result
logs_files = glob.glob("logs/*.pt")
all_logs = []
for file in logs_files:
    logs = torch.load(file, map_location="cpu")
    logs["file"] = file
    all_logs.append(logs)

plt.figure(figsize=(10, 5))
for logs in all_logs:
    label = logs.get("reg", "") + "_" + logs.get("problem", "") + "_" + str(logs.get("ML", "")) + "_" 
    plt.plot(logs["cost"], logs["loss"], label=f"{os.path.basename(logs['file'])[5:-3]} Loss")
plt.xscale("symlog")
plt.yscale("log")
plt.xlabel("Total Cost (Iterations)")
plt.ylabel("Loss")
plt.title("Loss during Bilevel Monte Carlo Optimization (All Settings)")
plt.legend()
plt.grid()
plt.savefig("loss_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for logs in all_logs:
    plt.plot(logs["cost"], logs["psnr"], label=f"{os.path.basename(logs['file'])[5:-3]}")
plt.xscale("symlog")
plt.xlabel("Total Cost (Iterations)")
plt.ylabel("PSNR")
plt.title("PSNR during Bilevel Monte Carlo Optimization (All Settings)")
plt.legend()
plt.grid()
plt.savefig("psnr_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(10, 5))
for logs in all_logs:
    plt.plot(logs["cost"], logs["val_psnr"], label=f"{os.path.basename(logs['file'])[5:-3]}")
plt.xscale("symlog")
plt.xlabel("Total Cost (Iterations)")
plt.ylabel("Validation PSNR")
plt.title("Validation PSNR during Bilevel Monte Carlo Optimization (All Settings)")
plt.legend()
plt.grid()
plt.savefig("val_psnr_bilevel_monte_carlo.png", bbox_inches='tight', dpi=300)
plt.show()

# plotting the results
# plt.figure(figsize=(10, 5))
# plt.plot(logs_dict["cost"], logs_dict["loss"], label="Loss")
# plt.xscale("linear")
# plt.yscale("log")
# plt.xlabel("Total Iterations")
# plt.ylabel("Loss")
# plt.title("Loss during Bilevel Monte Carlo Optimization")
# plt.legend()
# plt.grid()
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(logs_dict["cost"], logs_dict["psnr"], label="PSNR", color="orange")
# plt.xscale("linear")
# plt.xlabel("Total Iterations")
# plt.ylabel("PSNR")
# plt.title("PSNR during Bilevel Monte Carlo Optimization")
# plt.legend()
# plt.grid()
# plt.show()