# python
import os
from glob import glob
import subprocess

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

ieee_colors_bright = ["#FFA300", "#FFD100", "#78BE20", "#00843D", "#BA0C2F", "#981D97", "#009CA6", "#00629B", "#00B5E2"]
plt.style.use(["science", "ieee"])

results_path = "figures"
if not os.path.exists(results_path):
  os.makedirs(results_path)

PLOT_FILETYPE="pdf"

def introduction_figure():
  ignore_cache = False
  # caching
  cache_files = [
    "introduction_heaviside_crisp_xs.npy",
    "introduction_heaviside_crisp_ys.npy",
    "introduction_heaviside_ipa_dys.npy",
    "introduction_heaviside_smooth_dys.npy"
  ]
  xs = np.linspace(-1,1,200)
  if ignore_cache or not np.all([os.path.exists(c) for c in cache_files]):
    print("No cached files or chose to ignore them, this may take a while...")
    def H(x): # heaviside function
      if x < 0:
        return 0
      return 1
    def dHxdx(x): # heaviside pathwise derivative
      return 0
    crisp_xs = xs.copy()
    # crisp plot with correct visualization of jumping discontinuity
    crisp_ys = np.array([H(x) for x in crisp_xs], dtype=float)
    jump_pos = np.where(np.abs(np.diff(crisp_ys)) >= 0.5)[0]+1
    crisp_xs = np.insert(crisp_xs, jump_pos, np.nan)
    crisp_ys = np.insert(crisp_ys, jump_pos, np.nan)
    # IPA derivative
    #ipa_dys = np.array([np.mean(np.vectorize(dHxdx)(x+np.random.normal(0, 1, 5000000))) for x in xs])
    ipa_dys = np.array([np.mean(np.vectorize(dHxdx)(x+np.random.normal(0, 1, 1000))) for x in xs])
    # Smooth derivative
    mu = 0.52
    #smooth_dys = np.array([np.mean([((H(x + mu*u)-H(x)) / mu) * u for u in np.random.normal(0, 1, 5000000)]) for x in xs])
    smooth_dys = np.array([np.mean([((H(x + mu*u)-H(x)) / mu) * u for u in np.random.normal(0, 1, 1000)]) for x in xs])
    np.save(cache_files[0], crisp_xs)  
    np.save(cache_files[1], crisp_ys)
    np.save(cache_files[2], ipa_dys)
    np.save(cache_files[3], smooth_dys)
  else:
    crisp_xs = np.load(cache_files[0])
    crisp_ys = np.load(cache_files[1])
    ipa_dys = np.load(cache_files[2])
    smooth_dys = np.load(cache_files[3])
  # plotting
  fig, ax = plt.subplots(figsize=(3.3, 1.0))
  ax.plot(crisp_xs, crisp_ys, label=r"$H(x)$", color=ieee_colors_bright[5])
  ax.plot(xs, smooth_dys, label=r"$dH/dx$ (smoothed)", color=ieee_colors_bright[6])
  ax.plot(xs, ipa_dys, label=r"$dH/dx$ (IPA)", color=ieee_colors_bright[7])
  #ax.set_title("The Heaviside Step Function and its Derivative")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.4, 0.0, -0.05), ncols=3, handlelength=1.1, handletextpad=0.5)
  outpath = os.path.join(results_path, f"introductory_heaviside.{PLOT_FILETYPE}")
  fig.savefig(outpath)
  print("Resulting figure written to", outpath, ".")

def synthetic_example():
  #subprocess.run("./run.py experiments/synthetic_example/", cwd="experiments/gradient_evaluation", shell=True)
  results_dir = "experiments/gradient_evaluation/results/*synthetic*"
  convolution = glob(results_dir + "stochastic*")[0]
  si_32 = glob(results_dir + "dgsi*32*use_dea=0-*")[0]
  si_32_up = glob(results_dir + "dgsi*32*use_dea=0.33-*")[0]
  si_4_up = glob(results_dir + "dgsi*num_paths=4-*")[0]
  fig, ax = plt.subplots()
  #ax.hlines(0.0, -4, 4, color="black", alpha=0.2, lw=0.5)
  ax.grid()
  for file, label, color, alpha, linestyle, linewidth in zip(
      [convolution, si_32, si_32_up, si_4_up],
      ["Convolution", "SI+AD/32 Paths", "SI+AD/32 P./UP", "SI+AD/4 P./UP"],
      #["black", "#BA0C2F", "#D72447", "#F03F61"],
      ["black", "#BA0C2F", "#BA0C2F", "#BA0C2F"],
      [1.0, 1.0, 0.75, 0.65],
      #["-", "--", "-.", ":"],
      ["-", "-.", "--", "-"],
      [0.75, 1.5, 1.5 , 1.5]
  ):
    data = pd.read_csv(file)
    ax.plot(data["x0"], data["dydx0"], label=label, color=color, alpha=alpha, ls=linestyle, lw=linewidth)
    #ax.plot(data["x0"], data["y"], label=label, color=color, alpha=alpha, ls=linestyle)
  ax.legend(handlelength=1.8, frameon=True, framealpha=1.0)
  #ax.set_yscale("symlog")
  ax.set_xlabel("input")
  ax.set_ylabel("derivative")
  fig.tight_layout()
  outpath = os.path.join(results_path, f"synthetic_example.{PLOT_FILETYPE}")
  fig.savefig(outpath)
  print("Resulting figure written to", outpath, ".")

def everything():
  print("Warning: reproducing every result could take up to multiple weeks on a high-end machine. There is no mechanism to pause and resume.") 
  proceed = input("Proceed anyway? (y/N): ")
  if proceed != "y" and proceed != "Y":
    exit(0)

def todo():
  pass
