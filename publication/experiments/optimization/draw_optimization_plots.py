#!/usr/bin/python3

import pandas as pd
import numpy as np
import sys
import scienceplots
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import itertools
import os
import re

results_path = sys.argv[1]

zoom_segment = 0.333 # what (final) proportion of the ctime plot to zoom in on

select_ctime_factor = 1

plt.style.use(["science", "ieee"])

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

#plt.figure(figsize=(3.3, 1.8))

program_opt_data = {
  "HOTEL": (1800, 500, 1, 1, -53400, -52600, "-Revenue [USD]", ["GA", "PGO", "RF", "DGSI", "DGO"]),
  "AC": (1800, 400, 1, 10, 1.975, 2.4, "Cost", ["GA", "PGO", "RF", "DGSI", "DGO"]),
  "EPIDEMICS": (3600, 50, 1, 1, 26, 37, "MSE", ["GA", "PGO", "RF", "DGSI", "DGO"]),
  "TRAFFIC_10x10": (1800, 500, 1, 1, -280, -250, "-Traffic Flow [Veh./Step]", ["SA", "GA", "PGO", "RF", "DGSI", "DGO"]),
  "TRAFFIC_20x20": (7200, 200, 1, 1, -2230, -2130, "-Traffic Flow [Veh./Step]", ["SA", "GA", "PGO/100", "PGO/1000", "DGO/100", "DGO/1000"]), # only single-digit number of steps with DGSI
  "TRAFFIC_40x40": (28800, 50, 1, 1, -17500, -13000, "-Traffic Flow [Veh./Step]", ["PGO/100", "PGO/1000", "DGO/100", "DGO/1000"])
}

prog_substr = { 'HOTEL': 'hotel',
                'EPIDEMICS': 'epidemics',
                'AC': 'ac',
                'TRAFFIC_10x10': 'traffic_grid_populations_10x10',
                'TRAFFIC_20x20': 'traffic_grid_populations_20x20',
                'TRAFFIC_40x40': 'traffic_grid_populations_40x40'
              }

est_substrs = { 'DGSI': ('dgsi_num_paths',),
                'DGO': ('dgo',),
                'DGO/1000': ('dgo_single_pass', 'samples=1000_'),
                'DGO/100': ('dgo_single_pass', 'samples=100_'),
                'PGO': ('pgo',),
                'PGO/1000': ('pgo', 'samples=1000_'),
                'PGO/100': ('pgo', 'samples=100_'),
                'RF': ('reinforce',),
                'RF/1000': ('reinforce', 'samples=1000_'),
                'RF/100': ('reinforce', 'samples=100_'),
                'IPA': ('enable_ad=True',),
                'GA': ('GA',),
                'SA': ('SA',)
              }

def get_line_style(est_name):
  substr_to_style = {"SA": ("#78BE20", 0.75, 'dashed'), "GA": ("#00843D", 0.75, 'dotted'), "PGO/100": ("#009CA6", 0.75, 'dashed'), "PGO/1000": ("#007377", 0.75, 'dashed'), "RF/100": ("#981D97", 0.75, 'dashdot'), "RF/1000": ("#772583", 0.75, 'dashdot'), "DGO/100": ("#FFA300", 1.5, 'solid'), "DGO/1000": ("#e87722", 1.5, 'solid'), "DGSI": ("#BA0C2F", 1.5, 'solid'), "IPA": ("#FFD100", 0.75, 'dashdot')}
#  substr_to_style = {"SA": ("#78BE20", 0.75, 'dotted'), "GA": ("#00843D", 0.75, 'dashed'), "NRS/1000": ("#70f5ff", 0.75, 'dashed'), "NRS/100": ("#009CA6", 0.75, 'dashed'), "RF/1000": ("#14a9ff", 0.75, 'dashdot'), "RF/100": ("#00629B", 0.75, 'dashdot'), "MC+AD/1000": ("#ffc766", 1.5, 'solid'), "MC+AD/100": ("#FFA300", 1.5, 'solid'), "SI+AD": ("#BA0C2F", 1.5, 'solid'), "IPA": ("#981D97", 0.75, 'dashdot')}
  for substr in sorted(substr_to_style.keys(), key=lambda x: len(x), reverse=True):
    if substr in est_name:
      return substr_to_style[substr]
  print(est_name)
  assert(False)

def set_ax_size(w, h):
  ax = plt.gca()
  l = ax.figure.subplotpars.left
  r = ax.figure.subplotpars.right
  t = ax.figure.subplotpars.top
  b = ax.figure.subplotpars.bottom
  figw = float(w)/(r-l)
  figh = float(h)/(t-b)
  ax.figure.set_size_inches(figw, figh)

def flip_legend(items, ncol):
  return itertools.chain(*[items[i::ncol] for i in range(ncol)])

for prog_name, (ctime_limit, step_limit, window_ctime, window_step, zoom_y_lower, zoom_y_upper, ylabel, est_names) in program_opt_data.items():
  min_y_crisp_fname = {}

  p_substr = prog_substr[prog_name]

  full_est_name = {} # estimator name plus suffix for samples or paths

  for est_name in est_names:
    e_substrs = est_substrs[est_name]
    all_substrs = e_substrs + (p_substr,) + ("mean_time",)

    curr_est_min_y_crisp = float("inf")
   
    for fname in os.listdir(results_path):
      found_all = True
      for substr in all_substrs:
        if substr not in fname:
          found_all = False
          break
      if not found_all:
        continue
    
      df = pd.read_csv(results_path + "/" + fname)
    
      df.cumulative_time /= 1e6

      idx = df[df.cumulative_time >= ctime_limit * select_ctime_factor].first_valid_index()

      #print(f"got {df.iloc[idx].y_crisp} for {est_name}, {fname}")
    
      if idx and df.iloc[idx].y_crisp < curr_est_min_y_crisp:
          curr_est_min_y_crisp = df.iloc[idx].y_crisp
          min_y_crisp_fname[est_name] = fname

    if not est_name in min_y_crisp_fname:
      print(f"no results found for program {prog_name}, estimator {est_name}")
      continue

    print(f"best for {prog_name}, {est_name}: {min_y_crisp_fname[est_name]} {curr_est_min_y_crisp}")

    num_paths = 1
    num_samples = 1
    restrict_mode = None

    full_est_name[est_name] = est_name

    m = re.search('num_samples=(.+?)_', min_y_crisp_fname[est_name])
    if m:
      num_samples = int(m.group(1))

    m = re.search('num_paths=(.+?)-', min_y_crisp_fname[est_name])
    if m:
      num_paths = int(m.group(1))

    m = re.search('mode=(.+?)-', min_y_crisp_fname[est_name])
    if m:
      restrict_mode = int(m.group(1))
  
    print("got restrict mode", restrict_mode)

    if not "/" in est_name:
      restrict_mode_names = { 0: "Di", 3: "Ch", 4: "IW", 5: "WO" }
      if restrict_mode != None:
        full_est_name[est_name] += "/" + restrict_mode_names[restrict_mode]

      if num_paths > 1:
        full_est_name[est_name] += "/" + str(num_paths)
      if num_samples > 1:
        full_est_name[est_name] += "/" + str(num_samples)


    print("full est name:", full_est_name[est_name])

  add_inset = not '40x40' in prog_name

  ax = plt.gca()
  ax.margins(x=0)
  #axins = inset_axes(ax, 0.75, 1.92, loc=5, bbox_to_anchor=(-0.0, -0.0))

  if add_inset:
    axins = plt.axes((0, 0, 1, 1))
    ip = InsetPosition(ax, (1 - zoom_segment, 0, zoom_segment, 1))
    #axins.tick_params(axis='y', which='both', bottom=False, top=False, left=False, right=True, labelbottom=False)
    axins.set_axes_locator(ip)
    axins.yaxis.tick_right()
    axins.set_xticklabels("")
    axins.margins(x=0)
    #axins.xaxis.set_minor_locator(ax.xaxis.get_minor_locator())

  for est_id, (est_name, est_fname) in enumerate(min_y_crisp_fname.items()):
    df = pd.read_csv(results_path + "/" + est_fname)
    df.cumulative_time /= 1e6
    df = df[df.cumulative_time < ctime_limit]
    df["y_crisp_rolling"] = df.y_crisp.rolling(window=window_ctime).mean()
    col, lw, ls = get_line_style(full_est_name[est_name])

    if add_inset:
      axins_df = df[df.cumulative_time >= ctime_limit * (1 - zoom_segment)]
      axins.plot(axins_df.cumulative_time / 60, axins_df.y_crisp_rolling, label=full_est_name[est_name], color=col, lw=lw, ls=ls)
    ax.plot(df.cumulative_time / 60, df.y_crisp_rolling, label=full_est_name[est_name], color=col, lw=lw, ls=ls)

  ticks = (ax.xaxis.get_minorticklocs(), ax.xaxis.get_majorticklocs())
  print(ticks)
  tick_intervals = (ticks[0][2] - ticks[0][1], ticks[1][2] - ticks[1][1])
  print(f"using {tick_intervals} steps")

  if add_inset:
    axins.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(tick_intervals[0]))
    axins.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_intervals[1]))

  #print(minor_ticks[0].get_loc())
  #print(minor_ticks[1].get_loc())

#  print(ax.xaxis.get_minor_locator())


  ax.set_xlabel("Wall Time [Minutes]")
  ax.set_ylabel(ylabel)
  ax.set_xlim(0, ctime_limit / 60)
  if 'AC' in prog_name:
    ax.set_ylim(1.8, 4)
  title = prog_name.replace("_", " ")
  ax.set_title(title)
  if add_inset:
    axins.set_ylim(zoom_y_lower, zoom_y_upper)
  #set_ax_size(2.7, 1.0)
  #plt.tight_layout()
  set_size(2.4, 1.3)
  plt.savefig(prog_name + "_ctime.pdf")

  plt.clf()
  #set_ax_size(2.5, 2)

  for est_id, (est_name, est_fname) in enumerate(min_y_crisp_fname.items()):
    est_fname = est_fname.replace("mean_time", "mean")
    df = pd.read_csv(results_path + "/" + est_fname)
    df.cumulative_time /= 1e6
    df = df[df.cumulative_time < ctime_limit]
    df = df[df.step < step_limit]
    df["y_crisp_rolling"] = df.y_crisp.rolling(window=window_step).mean()
    col, lw, ls = get_line_style(full_est_name[est_name])
    plt.plot(df.step, df.y_crisp_rolling, label=full_est_name[est_name], color=col, lw=lw, ls=ls)

  plt.xlabel("Optimization Step")
  plt.ylabel(ylabel) # or loss
  plt.xlim(window_step, step_limit)
  plt.margins(x=0)
  #set_ax_size(2.5, 2)
  handles, labels = plt.gca().get_legend_handles_labels()
  ncols = 2 if '40x40' in prog_name else 3
  mode = None if '40x40' in prog_name else "expand"
  plt.legend(flip_legend(handles, 3), flip_legend(labels, 3), bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode=mode, borderaxespad=0, ncol=ncols, handletextpad=0.4, fontsize=7, handlelength=1.8)

  set_size(2.4, 1.3)
  plt.savefig(prog_name + "_step.pdf")
  plt.clf()
