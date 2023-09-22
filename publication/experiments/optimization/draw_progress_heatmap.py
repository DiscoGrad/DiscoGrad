#!/usr/bin/python3

import os
import pandas
import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from functools import cmp_to_key

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns

def fname_to_label(fname):
  #estimator_to_label = {"dgsi": "SI+AD", "dgo": "MC+AD", "pgo": "NRS", "reinforce": "REINFORCE", "crisp__SA": "SA", "crisp__GA": "GA", "crisp_enable_ad=True": "IPA"}
  #restrict_mode_to_label = {0: "Truncate", 3: "Chaudhuri", 4: "Ign. Weights", 5: " Weights Only"}

  estimator_to_label = {"dgsi": "DGSI", "dgo_single_pass": "DGO", "pgo": "PGO", "reinforce": "RF", "crisp__SA": "SA", "crisp__GA": "GA", "crisp_enable_ad=True": "IPA"}
  restrict_mode_to_label = {0: "Di", 3: "Ch", 4: "IW", 5: "WO"}

  #print(fname)
  m = re.search('reps=\d+-(.+?)_num_', fname)
  if not m:
    m = re.search('reps=\d+-(.+?)-', fname)
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
  m = re.search('restrict_mode=(\d+)', fname)
  if m:
    restrict_mode = restrict_mode_to_label[int(m.group(1))]

  paths_samples_str = ""
  if not "GA" in estimator_label and not "SA" in estimator_label:
    #paths_samples_str = f", {num_paths if num_paths > 1 else num_samples} {'paths' if num_paths > 1 else 'samples'}"
    paths_samples_str = f"/{num_paths if num_paths > 1 else num_samples}"

  #label = f"{estimator_label}{(', ' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"
  label = f"{estimator_label}{('/' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"

  return label

results_path = sys.argv[1]

time_limit_portions = (0.01, 0.05, 1.0)

programs = { "HOTEL": ("hotel", 1800),
             "EPIDEMICS": ("epidemics", 3600),
             "AC": ("ac", 1800),
             "TRAFFIC\n10x10": ("traffic_grid_populations_10x10", 1800),
             "TRAFFIC\n20x20": ("traffic_grid_populations_20x20", 7200),
             "TRAFFIC\n40x40": ("traffic_grid_populations_40x40", 14400) }

fig, axes = plt.subplots(1, len(sys.argv[2:]), figsize=(5.5, 6.05))

fname_filter_strings = ('enable', 'num_paths=16', 'num_paths=32', 'delta')

ax_id = 0
for program_title, (program_substr, time_limit_s) in programs.items():
  found = False
  for substr in sys.argv[2:]:
    if substr in program_title:
      found = True
      break
  if not found:
    continue

  print("drawing", program_title)

  print(time_limit_portions, time_limit_s)
  time_limits_s = [(prop * time_limit_s) for prop in time_limit_portions]
  
  min_y_crisp = float("inf")
  y_crisps = defaultdict(list)
  initial_y_crisps = []
  
  k = 0
  paths = sorted(os.listdir(results_path))
  for fname in paths:
    if 'dgo' in fname and not 'dgo_single_pass' in fname:
      continue

    if not program_substr in fname:
      continue
 
    if not 'mean_time.txt' in fname:
      continue

    skip = False
    for s in fname_filter_strings:
      if s in fname:
        skip = True
        break
    if skip:
      continue
  
    #if not ("SA" in fname or "GA" in fname or "sampling" in fname or "stoch" in fname or ("mode=0" in fname and "paths=8" in fname)) or 'delta' in fname: # XXX
    #if not ("SA" in fname or "GA" in fname or "sampling" in fname or "stoch" in fname or ("mode=0" in fname and "paths=8" in fname)) or 'delta' in fname: # XXX
    #  continue

    if "traffic" in fname and (("dgsi_num_paths" in fname and not "mode=3" in fname) or "GA" in fname):
      continue
  
    fpath = results_path + "/" + fname
    print(fpath)
    df = pandas.read_csv(fpath)
    if df.empty:
      continue

    if "SA" not in fname and "GA" not in fname: # these start from different solutions than the gradient-based ones
      initial_y_crisps.append(df.iloc[0].y_crisp)
  
    est_name = fname_to_label(fname)
    curr_y_crisps = []
    for time_limit_s in time_limits_s:
      idx = df[df.cumulative_time / 1e6 >= time_limit_s].first_valid_index()
      print(fname, "checking at", time_limit_s)
      if idx != None:
        curr_y_crisps.append(df.iloc[idx].y_crisp)
        print("got", float(df.iloc[idx].y_crisp))
      else:
        print("not found")
        curr_y_crisps.append(df.iloc[-1].y_crisp)
      min_y_crisp = min(curr_y_crisps[-1], min_y_crisp)
      print(f"current for {est_name}: {curr_y_crisps}")
      print(f"new min: {min_y_crisp}")

    if est_name not in y_crisps or y_crisps[est_name][-1] > curr_y_crisps[-1]:
      print(f"best for {est_name}: {curr_y_crisps}")
      y_crisps[est_name] = curr_y_crisps
  
  progress_df = pd.DataFrame(columns=['Estimator', 'Time budget [%]', 'Progress'])

  mean_initial_y_crisp = np.mean(initial_y_crisps)
  print("mean initial:", mean_initial_y_crisp)
  min_y_crisp_diff = min_y_crisp - mean_initial_y_crisp
  print("min final:", min_y_crisp_diff)
  
  for i, (est_name, ys) in enumerate(y_crisps.items()):
    diffs = [y - mean_initial_y_crisp for y in ys]
    print(est_name, diffs)
    ndiffs = [max(0, 100 * diff / min_y_crisp_diff) for diff in diffs]
    ndiffs = np.array(ndiffs, dtype=float)
    print(est_name, ndiffs)
    #ndiffs = [diff / min_y_crisp_diff for diff in diffs]
    #print(ndiffs)
    #diffs_str = ", ".join([f"{max(0, ndiff):.2f}" for ndiff in ndiffs])
    #print(f"{est_name}: {diffs_str}")
    
    for j, ndiff in enumerate(ndiffs):
      row = {'Estimator': est_name, 'Time budget [%]': int(time_limit_portions[j] * 100), 'Progress': ndiff}
      
      print(row)
      #print(row['Estimator'])
      progress_df = progress_df.append(row, ignore_index=True)

  def cmp_estimator_str(a, b):
    order = ['SA', 'GA', 'RF', 'PGO', 'DGO', 'DGSI', 'Ch', 'WO', 'IW', 'Di']
    print(a, b)
    for i, s in enumerate(order):
      if s in a[1]:
        a_idx = i
      if s in b[1]:
        b_idx = i


    if a_idx < b_idx:
      return -1
    if a_idx > b_idx:
      return 1
    return 0

  def sort_estimator_strs(ser):
    print(ser)
    r = sorted(zip(ser.index, ser), key=cmp_to_key(cmp_estimator_str))
    r_ = np.zeros(len(r))
    for i, v in enumerate(r):
      print(v)
      r_[v[0]] = i
    return r_

  if '40x40' in program_title:
    for est in ('RF/100', 'RF/1000', 'DGSI/Ch/4', 'DGSI/Ch/8', 'SA'):
      for tb in (1, 5, 100):
        progress_df = progress_df.append({'Estimator': est, 'Time budget [%]': tb, 'Progress': float("nan")}, ignore_index=True)

  print(progress_df.Estimator)
  #print(progress_df.to_string())
  progress_df = progress_df.pivot(*list(progress_df.columns))
  progress_df = progress_df.sort_values(by=['Estimator'], key=lambda ser: sort_estimator_strs(ser))

  
  
  xticks = [f"{int(prop * 100)}" for prop in time_limit_portions]

  cmap=sns.light_palette("#005500", as_cmap=True)

  def get_line_style(est_name):
    substr_to_style = {"SA": ("#78BE20", 0.75, 'dashed'), "GA": ("#00843D", 0.75, 'dotted'), "NRS/1000": ("#009CA6", 0.75, 'dashed'), "NRS/100": ("#007377", 0.75, 'dashed'), "RF/1000": ("#981D97", 0.75, 'dashdot'), "RF/100": ("#772583", 0.75, 'dashdot'), "MC+AD/1000": ("#FFA300", 1.5, 'solid'), "MC+AD/100": ("#e87722", 1.5, 'solid'), "SI+AD": ("#BA0C2F", 1.5, 'solid'), "IPA": ("#FFD100", 0.75, 'dashdot')}
  #  substr_to_style = {"SA": ("#78BE20", 0.75, 'dotted'), "GA": ("#00843D", 0.75, 'dashed'), "NRS/1000": ("#70f5ff", 0.75, 'dashed'), "NRS/100": ("#009CA6", 0.75, 'dashed'), "RF/1000": ("#14a9ff", 0.75, 'dashdot'), "RF/100": ("#00629B", 0.75, 'dashdot'), "MC+AD/1000": ("#ffc766", 1.5, 'solid'), "MC+AD/100": ("#FFA300", 1.5, 'solid'), "SI+AD": ("#BA0C2F", 1.5, 'solid'), "IPA": ("#981D97", 0.75, 'dashdot')}
    for substr in sorted(substr_to_style.keys(), key=lambda x: len(x), reverse=True):
      if substr in est_name:
        return substr_to_style[substr]
    print(est_name)
    assert(False)

  print(progress_df.to_string())
  if ax_id == 0:
    hmap = sns.heatmap(progress_df, cmap=cmap, square=True, annot=True, fmt=".0f", cbar=False, ax=axes[0])
    ticks = hmap.get_yticklabels()
    #hmap.set_yticklabels(ticks, ha="center") # causes labels to overlap with heatmap
    #for tick in ticks:
    #  tick.set_color(get_line_style(tick.get_text())[0])
  else:
    sns.heatmap(progress_df, cmap=cmap, square=True, annot=True, fmt=".0f", cbar=False, yticklabels=[], ax=axes[ax_id])
    axes[ax_id].set_ylabel('')

  if 'x' in sys.argv[2]: # traffic nxn
    axes[ax_id].hlines([1, 3, 5, 7], *axes[ax_id].get_xlim(), color="white", lw=4)
  else:
    axes[ax_id].hlines([1, 2, 4, 6, 8, 10, 12, 14], *axes[ax_id].get_xlim(), color="white", lw=4)
  
  axes[ax_id].set_xticklabels(xticks)
  axes[ax_id].set_xlabel('')
  axes[ax_id].set_title(program_title)

  ax_id += 1


plt.tight_layout(rect=[0, 0.01, 1, 1]) # absolute madness
if 'x' in sys.argv[2]: # traffic nxn
  fig.text(0.44, 0.11, "Portion of Time Budget [%]")
else:
  fig.text(0.44, 0.0049, "Portion of Time Budget [%]")
plt.savefig("progress.pdf", bbox_inches='tight')
