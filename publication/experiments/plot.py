#!/usr/bin/python3

# usage:
# ./plot.py [--cm | --sc=substrstart[,substrend]] |--min | --max | --ylim_ref | --mov_avg=<num> | --norm | --log | --percentile=<num> | --drop_elems=<num> ] files... xaxisvar [varname=<val>...] yaxisvar
#
# Only the first option is used at the moment.
#
# --cm               continuiously update figure (exit with Ctrl+C in terminal)
# --sc               apply shades of the same color (sc) to data from every input file with the same substring,
#                    delianated by the two strings `substrstart` and `substrend` (they may also contain a `=` character)
# --min              plot minimum until point
# --max              plot maximum until point
# --ylim_ref         if set, the first file provided determines the vertical limits for plotting
# --mov_avg=<num>    moving average with window size num
# --norm             normalize all data to be between -1 and 1
# --log              log plot for values outside of [-1, 1]
# --percentile=<num> drop all elements outside the provided percentile
# --drop_elems=<num> restrict y range to [-num,num]
# files...           list of files to read data from
# 
# Continue to read beyond this point at your own risk!

import sys
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#matplotlib.use('pdf')

plt.rcParams['figure.figsize'] = (12, 8)

continuous_mode = False
if sys.argv[1] == "--cm":
  continuous_mode = True

# special coloring option
same_color = [] # either 1 or 2 elements
substrs = set()
nsubstrs = defaultdict(lambda: 0)
if sys.argv[1].startswith("--sc"):
  firstequals = sys.argv[1].find("=")
  if firstequals == -1:
    print("Usage of option --sc: --sc=start[,end]")
    exit(-1)
  params = sys.argv[1][firstequals+1:]
  args = params.split(",")
  if len(args) == 0:
    print("Usage of option --sc: --sc=start[,end]")
    exit(-1)
  elif len(args) <= 2:
    same_color = args
    print(f"{same_color=}")
  else:
    print("Usage of option --sc: --sc=start[,end]")
    exit(-1)
# get the substring between same_color[0] and same_color[1]
def sc_get_substring(s: str):
  sc0 = s.find(same_color[0])
  if sc0 != -1 and len(same_color) == 1:
    return s[sc0 + len(same_color[0]):]  
  sc1 = s.find(same_color[1])
  if sc1 != -1 and len(same_color) == 2:
    ret = s[s.find(same_color[0])+len(same_color[0]):sc1]  
    return ret
  return ""

old_curves = []
def update_plot():
  global old_curves
  for old_curve in old_curves:
    old_curve.remove()
  old_curves = []

  dfs = []

  x_lower = x_upper = None # currently deactivated...

  show_legend = True

  for i, s in enumerate(sys.argv[1:]):
    if s.endswith(":"):
      x_lower = int(s[:-1])
      continue
    if s.startswith(":"):
      x_upper = int(s[1:])
      continue
    if s.startswith('--'):
      continue
    if s.endswith('.txt'):
      try:
        print("reading", s)
        if len(same_color) > 0:
          substr = sc_get_substring(s)
          substrs.add(substr)
          nsubstrs[substr] += 1
        dfs.append([s, pd.read_csv(s)])
      except Exception as ex:
        print("EXCEPTION: ", ex)
        continue
    else:
      break

  print(substrs)
  ncolors = len(substrs)

  i += 1

  x_axis = sys.argv[i]

  if x_axis == "ctime":
    x_axis = "cumulative_time"
    

  ys = []
  for j in range(i + 1, len(sys.argv)):
    if '=' in sys.argv[j]:
      print("trying to split", sys.argv[j])
      k, v = sys.argv[j].split('=')
      for i in range(len(dfs)):
        dfs[i][1] = dfs[i][1][dfs[i][1][k] == float(v)]
      continue

    if sys.argv[j] == "crisp_y":
      ys.append("y_crisp")
    else:
      ys.append(sys.argv[j])

  #plt.axhline(y = 0, color = 'black', linestyle = '--')

  color = iter(plt.cm.rainbow(np.linspace(0, 1, len(dfs))))
  if len(same_color) > 0:
    colorscales = ["Purples", "Greens", "Oranges", "Blues", "Reds"]
    color = {s: iter(getattr(plt.cm, colorscales[i%5])(np.linspace(0.3, 0.9, nsubstrs[s]))) for i, s in enumerate(substrs)}
    print(color)

  # convert from microseconds to seconds
  if x_axis == 'cumulative_time':
    for df in dfs:
      df[1][x_axis] /= 1e6

  if not x_lower:
    x_lower = float("inf")
    for df in dfs:
      for y in ys:
        #print(df[1][x_axis].values)
        if len(df[1][x_axis].values > 0):
          val = df[1][x_axis].values[0]
          if val != None:
            x_lower = min(x_lower, val)

  if not x_upper:
    x_upper = float("-inf")
    for df in dfs:
      for y in ys:
        if len(dfs[0][1][x_axis].values > 0):
          val = dfs[0][1][x_axis].values[-1]
          if val != None:
            x_upper = max(x_upper, val)

  lw = 3
  lower = float("inf")
  upper = float("-inf")

  if sys.argv[1] == '--ylim_ref':
    lower = np.min(dfs[0][1][ys[0]])
    upper = np.max(dfs[0][1][ys[0]])

    print(f"lower, upper: {lower}, {upper}")

  for df in dfs:
    for y in ys:
      col = df[1][y]

      if sys.argv[1] == '--min':
        col.loc[0] = float("inf")
        col = np.minimum.accumulate(col)

      if sys.argv[1] == '--max':
        col = np.maximum.accumulate(col)

      if sys.argv[1] == '--norm':
        min_, max_ = min(col), max(col)
        min_ = min_ if min_ != 0 else 1
        max_ = max_ if max_ != 0 else 1
          
        col = (col / max_).where(col >= 0, -col / min_)
        #col /= max(abs(col))
        #col -= np.mean(col)

      def symmetric_log(x):
        if x == 0:
          return 0

        #x *= 1e1

        if -1 <= x < 0 or 0 < x <= 1:
          return x
        if x < 1:
          return -np.log(-x)
        else:
          return np.log(x)


      if sys.argv[1] == '--log':
        col = np.array([symmetric_log(x) for x in col])

      if sys.argv[1].startswith('--mov_avg='):
        window_size = min(int(sys.argv[1].split('=')[1]), len(col))
        col = np.convolve(col, np.ones(window_size) / window_size, mode='same')
        col[:window_size] = [float("nan")] * window_size
        col[-window_size:] = [float("nan")] * window_size

      if sys.argv[1].startswith('--percentile='):
        print(f"percentiles: {lower}, {upper}")
        lower = min(lower, -np.percentile(-col, float(sys.argv[1].split('=')[1])))
        upper = max(upper, np.percentile(col, float(sys.argv[1].split('=')[1])))
        print(f"percentiles after: {lower}, {upper}")

      if sys.argv[1].startswith('--drop_elems='):
        num_elems = int(sys.argv[1].split('=')[1])
        sorted_col = sorted(col)
        lower = sorted_col[num_elems]
        upper = sorted_col[-num_elems]

        col = np.asarray([x if lower <= x <= upper else float("nan") for x in col])
      
      #plt.plot(df[1][x_axis][x_lower:x_upper], col[x_lower:x_upper], label=f"{y}: {df[0]}", linewidth=3, c=next(color))
      if len(same_color) > 0:
        old_curve, = plt.plot(df[1][x_axis], col, label=f"{y}: {df[0]}", linewidth=lw, c=next(color[sc_get_substring(df[0])]))
      else:
        old_curve, = plt.plot(df[1][x_axis], col, label=f"{y}: {df[0]}", linewidth=lw, c=next(color))
      old_curves.append(old_curve)
      lw *= 0.95
  #plt.ylim((0, 10))
  #plt.xlim((x_lower, x_upper))

  #plt.xscale("log")

  if lower != float("inf"):
    plt.ylim((lower, upper))
  #plt.ylim((-9, -7.5))
  plt.xlabel(x_axis)

  if len(ys) == 1:
    plt.ylabel(ys[0])

  if show_legend:
    plt.legend(loc=(0.0, 1.02))

  if continuous_mode: plt.pause(2)


if continuous_mode:
  plt.ion()
  plt.show()
  plt.grid()
  while True:
    update_plot()
    plt.savefig("plot.pdf", bbox_inches='tight')
else:
  plt.grid()
  update_plot()
  plt.savefig("plot.pdf", bbox_inches='tight')
  #plt.show()

