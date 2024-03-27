#!/usr/bin/python3

from run import get_fname
import sys
import os
import re
import argparse
import importlib
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
#from matplotlib.ticker import LogLocator
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats

plt.rcParams.update({"font.size": 12})

experiment_labels = {
  "traffic_grid_populations_2x2": "Traffic 2x2",
  "traffic_grid_populations_5x5": "Traffic 5x5",
  "epidemics": "Epidemics",
  "ac": "AC",
  "hotel": "Hotel",
}

parser = argparse.ArgumentParser(prog="visual_mse.py")
parser.add_argument("experiments_path", type=str, help="The directory containing the experiment definitions.")
parser.add_argument("results_path", type=str, help="The directory containing the experiment results.")
#parser.add_argument("ndims", type=int, help="The number of dimensions of the model.")
parser.add_argument("nreps", type=int, help="The number of macroreplications.")
parser.add_argument("--max", "-m", type=float, default=None)
parser.add_argument("--ignore-cache", "-i", action="store_true", help="Force redoing all calculations.")
args = parser.parse_args()

experiments = ["ac", "traffic_grid_populations_2x2", "traffic_grid_populations_5x5", "hotel", "epidemics"]
results = ["ac", "traffic_grid_populations_2x2", "traffic_grid_populations_5x5", "hotel", "epidemics"]
ndims = [25, 4, 25, 25, 25]
def include_files(file):
  return ("dgo" in file and ("samples=100-" in file or "samples=1000-" in file or "samples=10000-" in file)) or \
         ("dgsi" in file and ("paths=4-" in file or "paths=8-" in file or "paths=16-" in file )) or \
         ("pgo" in file and ("samples=100-" in file or "samples=1000-" in file or "samples=10000-" in file)) or \
         ("reinforce" in file and ("samples=100-" in file or "samples=1000-" in file or "samples=10000-" in file))

         #("dgsi" in file and ("paths=4-" in file or "paths=8-" in file or ("paths=16-" in file and "mode=0" in file))) or \
def extract_estimator(filename: str) -> str:
  """Given a filename, return the name of the estimator used."""
  names = "pgo dgo dgsi reinforce ipa".split(" ")
  pos = -1
  for name in names:
    pos = filename.find(name)
    if pos != -1:
      break
  endpos = filename[pos:].find("dim") + pos - 1
  return filename[pos:endpos]

def fname_to_label(fname):
  """Given an estimator string from extract_estimator, return a short label for plotting."""
  estimator_to_label = {"dgsi": "DGSI", "dgo": "DGO", "pgo": "PGO", "reinforce": "RF", "crisp__SA": "SA", "crisp__GA": "GA", "crisp_enable_ad=True": "IPA"}
  #restrict_mode_to_label = {0: "Di", 3: "Ch", 4: "IW", 5: " WO"}
  m = re.search(r"([a-z]+(_[a-z]*)?)_num", fname)
  estimator_label = estimator_to_label[m.group(1)]
  num_paths = 1
  m = re.search('num_paths=(\d+)', fname)
  if m:
    num_paths = int(m.group(1))
  num_samples = 1
  m = re.search('num_samples=(\d+)', fname)
  if m:
    num_samples = int(m.group(1))
  restrict_mode = None
  m = re.search('restrict_mode=(\w+)', fname)
  if m:
    #restrict_mode = restrict_mode_to_label[int(m.group(1))]
    restrict_mode = m.group(1)
  paths_samples_str = ""
  if not "GA" in estimator_label and not "SA" in estimator_label:
    #paths_samples_str = f", {num_paths if num_paths > 1 else num_samples} {'paths' if num_paths > 1 else 'samples'}"
    paths_samples_str = f"/{num_paths if num_paths > 1 else num_samples}"
  #label = f"{estimator_label}{(', ' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"
  label = f"{estimator_label}{('/' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"
  return label

def format_paths(num: int):
  if np.log2(num) == int(np.log2(num)):
    return str(num)
  if np.log10(num) == int(np.log10(num)) and np.log10(num) > 1:
    return f"$10^{int(np.log10(num))}$"
  return str(num)

mse_means = []
mse_vars = []

for experiment, result, ndim in zip(experiments, results, ndims):
  experiment_path = os.path.join(args.experiments_path, experiment)
  result_path = os.path.join(args.results_path, result)

  sys.path.append(experiment_path)
  print(experiment_path)
  import experiment
  importlib.reload(experiment)
  sys.path.remove(experiment_path)
  programs = experiment.programs
  estimators = experiment.estimators

  def load_ref_dfs():
    # reference data for each dimension
    ref_dfs = defaultdict(lambda: [])
    for dim in range(ndim):
      estim_param_str = '-'.join([f"{k}={v}" for k, v in estimators[-1]["params"].items()])
      for rep in range(args.nreps):
        file = get_fname(
          result_path,
          programs[0]["name"].replace("/", "_"),
          programs[0]["stddevs"],
          programs[0]["seed"][0],
          programs[0]["nreps"][0],
          estimators[-1],
          estim_param_str,
          dim,
          rep
        )
        print(f"adding {file} to reference dfs")
        try:
          df = pd.read_csv(file)
        except pd.errors.EmptyDataError as ex:
          print("reference is empty!")
          exit()
        if len(df) == 0:
          print("reference is empty!")
          exit()
        ref_dfs[dim].append(df)
    return ref_dfs

  def load_dfs():
    # estimator data for each dimension
    dfs = defaultdict(lambda: defaultdict(lambda: []))
    for file in sorted(os.listdir(result_path)):
      if not include_files(file):
        continue
      print(f"adding {file}")
      dimpos = file.find("dim_") + 4
      dim = int(file[dimpos:dimpos+3])
      try:
        df = pd.read_csv(os.path.join(result_path, file))
      except FileNotFoundError as ex:
        print(f"Warning: {ex}. Padding with 0s.")
        exit() # for submission, we don't want accidental zeros...
        df = pd.DataFrame(ref_dfs[dim][0])
      except pd.errors.EmptyDataError as ex:
        print("Empty file!")
        exit() # for submission, this needs to work
        continue

      if len(df) != len(ref_dfs[dim][0]):
        print("Length does not fit (yet)!")
        continue
      dfs[extract_estimator(file)][dim].append(df)
    return dfs

  def load_filenames():
    dfs = defaultdict()
    for file in sorted(os.listdir(result_path)):
      if not include_files(file):
        continue
      dfs[extract_estimator(file)] = None
    return dfs

  def calculate_mse_stats(ref_dfs, dfs):
    mse_mean = defaultdict(defaultdict)
    mse_var = defaultdict(defaultdict)
    maxmse = float("-1")
    for file in dfs:
      for dim in range(ndim):
        _mse_mean = None
        _mse_var = None
        if dim in dfs[file] and dim in ref_dfs:
          print(file, dim, len(dfs[file][dim]))
          # for each makroreplication, calculate mean squared error for estimator in file, in the dimension dim
          #_mse = [np.mean((dfs[file][dim][rep][ys[dim]] - ref_dfs[dim][rep][ys[dim]]) ** 2) for rep in range(len(dfs[file][dim]))]
          #_mse = [np.mean((dfs[file][dim][rep][ys[dim]] - ref_dfs[dim][rep][ys[dim]]) ** 2) for rep in range(args.nreps)]
          _mse = [np.mean(np.abs(dfs[file][dim][rep][ys[dim]] - ref_dfs[dim][rep][ys[dim]])) for rep in range(args.nreps)]
          # calculate mean and variance of the mse over makroreplications
          _mse_mean = np.mean(_mse)
          _mse_var = np.var(_mse)
        mse_mean[file][dim] = _mse_mean
        mse_var[file][dim] = _mse_var
        #print("mse of", file, "in", dim, "is", _mse_mean)
        if _mse_mean is not None and _mse_mean > maxmse:
          maxmse = _mse_mean
    if args.max:
      maxmse = args.max
    return mse_mean, mse_var, maxmse

  def get_sorted_keys(dfs):
    def est_id(key):
      parts = fname_to_label(key).split("/")
      if parts[0] == "DGSI":
        if parts[1] == "Di":
          return "C" + "{a:03d}".format(a=int(parts[2]))
        else:
          return "D" + "{a:03d}".format(a=int(parts[2]))
      elif parts[0] == "RF":
        return "A/"+parts[1]
      elif parts[0] == "PGO":
        return "B/"+parts[1]
      elif parts[0] == "DGO":
        return "Z/"+parts[1]
      return key
    ret = sorted(dfs, key=est_id)
    return ret

  def mse_dict_to_matrix(mse_mean, mse_var):
    mse_mean_matrix = np.zeros((len(mse_mean.keys()), ndim), dtype=float)
    mse_var_matrix = np.zeros((len(mse_var.keys()), ndim), dtype=float)
    for idx, file in enumerate(get_sorted_keys(mse_mean)):
      for dim in range(ndim):
        mse_mean_matrix[idx][dim] = mse_mean[file][dim]
        mse_var_matrix[idx][dim] = mse_var[file][dim]
    return mse_mean_matrix, mse_var_matrix

  cache_fname_mean = f"fidelity_matrix_cache_{experiment_path.replace('/','_')}_{args.nreps}_mean.npy"
  cache_fname_var = f"fidelity_matrix_cache_{experiment_path.replace('/','_')}_{args.nreps}_var.npy"
  if args.ignore_cache or not os.path.exists(cache_fname_mean):
    # list of column labels for the dimensions' gradients
    ys = [f"dydx{i}" for i in range(ndim)]
    ref_dfs = load_ref_dfs()
    dfs = load_dfs()
    sorted_keys = get_sorted_keys(dfs)
    mse_mean, mse_var, maxmse = calculate_mse_stats(ref_dfs, dfs)
    mse_mean_matrix, mse_var_matrix = mse_dict_to_matrix(mse_mean, mse_var)
    # cache results
    np.save(cache_fname_mean, mse_mean_matrix)
    np.save(cache_fname_var, mse_mean_matrix)
  else:
    print("using cached results.")
    dfs = load_filenames()
    sorted_keys = get_sorted_keys(dfs)
    # load cached results
    mse_mean_matrix = np.load(cache_fname_mean)
    mse_var_matrix = np.load(cache_fname_var)
  mse_means.append(mse_mean_matrix)
  mse_vars.append(mse_var_matrix)

print(sorted_keys)

# calculate mean over dimensions for each model
mse_means = list(map(lambda x: np.mean(x, axis=1)[:,None], mse_means))
#mse_vars = list(map(lambda x: np.mean(x, axis=1)[:,None], mse_vars))
# we now have one column vector for each model; concatenate them
mean_matrix = np.concatenate(mse_means, axis=1)
#var_matrix = np.concatenate(mse_vars, axis=1)
#im = plt.imshow(mean_matrix, norm="log", cmap="Reds")
#plt.colorbar(im)
#plt.show()
#print(mean_matrix)
mean_matrix = mean_matrix.transpose()
#print(mean_matrix)
group_nums = [3, 3, 3, 3, 3] # number of estimators in each group
ranges = np.cumsum([0] + group_nums + [1]) # end indices in the mean_matrix
files = list(map(fname_to_label, sorted_keys))

# ========
# PLOTTING
#========= # note: this is now condensed into a readable form, but required 5 hrs. of work to get just right.
#norm = LogNorm(vmin=np.min(mean_matrix), vmax=np.max(mean_matrix))
norm = LogNorm(vmin=1e-2, vmax=1e2)
# create gridspec with one row for the colorbar and one for the subplots
fig = plt.figure(figsize=(6.05, 3.0))
ncols = len(group_nums)
gs = plt.GridSpec(
  nrows=2,
  ncols=ncols,
  figure=fig,
  width_ratios=group_nums, # scale according to number of estimators in group
  height_ratios=[1, ncols], # make colorbar half a cell high
  left=0.21,
  bottom=0.16,
  #right=0.99,
  right=0.965,
  top=0.75,
  #wspace=0.08,
  wspace=0.1,
  hspace=0.15
)
axs = [plt.subplot(gs[1,c]) for c in range(0, ncols)]
#cmap=sns.color_palette("viridis_r", as_cmap=True)
cmap=sns.light_palette("#BA0C2F", as_cmap=True)
#cmap=sns.blend_palette(["#e32951", "#ba0c2f", "#3d0511"], as_cmap=True)
for ax_idx, ax in enumerate(axs):
  # plot heatmap with logarithmic normalization
  ax = sns.heatmap(mean_matrix[:,ranges[ax_idx]:ranges[ax_idx+1]], norm=norm, cmap=cmap, ax=ax, square=True, cbar=False, annot=False, fmt="g")
  # only set y label on the first subplot and disable ticks otherwise
  if ax_idx == 0:
    ax.set_ylabel("Problem", labelpad=8, fontsize=12)
    ax.set_yticks(np.arange(len(experiments)) + 0.5)
    ax.set_yticklabels(map(lambda x: f"{experiment_labels[x]}", experiments), rotation=0, fontsize=12) 
  else:
    ax.set_yticks([])
    ax.set_yticklabels([])
  # set the x ticks and labels
  _files = files[ranges[ax_idx]:ranges[ax_idx+1]]
  ax.set_xticks(np.arange(len(_files)) + 0.5) # center the ticks horizontally
  ax.set_xticklabels(map(lambda x: format_paths(int(x.split('/')[-1])), _files), rotation=0, ha="center", fontsize=12) # ticks are either mode/pahths or samples
  ax.set_xlabel("/".join(_files[0].split("/")[:-1]), labelpad=5, fontsize=12) # label is just the estimator name
# add colorbar
cbar_ax = fig.add_subplot(gs[0,:])
cbar = fig.colorbar(axs[0].collections[0], ax=axs[0], cax=cbar_ax, orientation="horizontal", location="top", pad=0.5, extend="both", extendfrac=0.02)#, ticks=LogLocator(subs=range(10)))
cbar.ax.minorticks_on()
cbar.ax.invert_xaxis()
cbar.set_label("mean MAE (logarithmic scale)", labelpad=10)
# remove all margins (handled by the GridSpec)
plt.subplots_adjust(
  left=0,
  bottom=0,
  right=1,
  top=1,
  wspace=0,
  hspace=0
)
fig.savefig("mse_matrix.pdf")
#plt.show()
