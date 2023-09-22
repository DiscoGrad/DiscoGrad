#!/usr/bin/python3

import sys
import re
from collections import defaultdict
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from functools import cmp_to_key

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import seaborn as sns

max_ci_proportion = 0

max_num_results = 100 # just for consistency

def fname_to_label(fname):
  #estimator_to_label = {"dgsi": "SI+AD", "dgo": "MC+AD", "pgo": "NRS", "reinforce": "REINFORCE", "crisp__SA": "SA", "crisp__GA": "GA", "crisp_enable_ad=True": "IPA"}
  #restrict_mode_to_label = {0: "Truncate", 3: "Chaudhuri", 4: "Ign. Weights", 5: " Weights Only"}

  estimator_to_label = {"dgsi": "DGSI", "dgo": "DGO", "pgo": "PGO", "reinforce": "RF", "crisp__SA": "SA", "crisp__GA": "GA", "crisp_enable_ad=True": "IPA", "crisp_enable_ad=False": "Crisp"}
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
  m = re.search('restrict_mode=(.+?)-', fname)
  if m:
    #restrict_mode = restrict_mode_to_label[int(m.group(1))]
    restrict_mode = m.group(1)

  paths_samples_str = ""
  if not "GA" in estimator_label and not "SA" in estimator_label and not "Crisp" in estimator_label:
    #paths_samples_str = f", {num_paths if num_paths > 1 else num_samples} {'paths' if num_paths > 1 else 'samples'}"
    paths_samples_str = f"/{num_paths if num_paths > 1 else num_samples}"

  #label = f"{estimator_label}{(', ' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"
  label = f"{estimator_label}{('/' + restrict_mode) if restrict_mode else ''}{paths_samples_str}"

  return label

def cmp_estimator_str(a, b):
  order = ["Crisp", 'IPA', 'SA', 'GA', 'RF', 'PGO', 'DGO', 'Ch', 'IW', 'WO', 'Di']
  #print(a, b)
  for i, s in enumerate(order):
    if s in a[1]:
      a_idx = i
    if s in b[1]:
      b_idx = i

  if a_idx < b_idx:
    return -1
  if a_idx > b_idx:
    return 1
  return -1 if len(a[1]) < len(b[1]) else 1

def sort_estimator_strs(ser):
  #print(ser)
  r = sorted(zip(ser.index, ser), key=cmp_to_key(cmp_estimator_str))
  r_ = np.zeros(len(r))
  for i, v in enumerate(r):
    #print(v)
    r_[v[0]] = i
  return r_

# https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def stats(data, confidence=0.95):
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a), scipy.stats.sem(a)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
  return m, h, np.std(data, ddof=1)

est_times_s = defaultdict(list)
num_paths = defaultdict(list)
with open(sys.argv[1]) as f:
  lines = f.readlines()
  for line in lines:
    line = line.replace('si_stddev_proportion=1-', '')
    m = re.search('(.+)replication_....\.txt took .+ \(cumulative estimation time was (.+)s\)', line)
    if m:
      fname = m.group(1)
      est_time_s = float(m.group(2))
      #print(f"got a result for {fname}: {est_time_s}")
      if len(est_times_s[fname]) < max_num_results:
        est_times_s[fname].append(est_time_s)

    m = re.search('(.+)replication_....\.txt took .+ \(cumulative estimation time was .+s\), num_paths: (.+)', line)
    if m:
      fname = m.group(1)
      curr_num_paths = float(m.group(2))
      #print(f"got a result for {fname}: {est_time_s}")
      if len(num_paths[fname]) < max_num_results:
        num_paths[fname].append(curr_num_paths)

program_fname_to_label = { 'hotel_hotel': 'HOTEL',
                           'ac_ac': 'AC',
                           'epidemics_epidemics': 'EPIDEMICS',
                           'traffic_grid_populations_traffic_grid_populations_5x5': 'TRAFFIC 5x5', }

crisp_time = {}
for fname, est_time_s in est_times_s.items():
  print(fname)
  program = program_fname_to_label[re.search('results/(.+)_stddev.+num_samples=(\d+)', fname).group(1)]
  if 'crisp_enable_ad=False-num_samples=1000-' in fname:
    mean, ci, std = stats(est_time_s)
    crisp_time[program] = mean / 1e3
    print(f"{len(est_time_s)} samples for {fname}: {est_time_s}, relative ci size: {ci / mean}")

df = pd.DataFrame(columns=['Program', 'Estimator', 'Slowdown'])
for fname, est_time_s in est_times_s.items():
  if 'crisp_enable_ad=False-num_samples=1-' in fname:
    continue
  if 'dgo' in fname and not 'dgo' in fname:
    continue
  #print(fname)
  program = program_fname_to_label[re.search('results/(.+)_stddev.+num_samples=(\d+)', fname).group(1)]
  num_samples = int(re.search('num_samples=(\d+)', fname).group(1))
  #m = re.search('num_paths=(\d+)', fname)
  #num_paths = 1
  #if m:
  #  num_paths = int(m.group(1))
  mean, ci, std = stats(est_time_s)
  print(f"{len(est_time_s)} samples for {fname}: {est_time_s}, relative ci size: {ci / mean}")
  max_ci_proportion = max(max_ci_proportion, ci / mean)
  curr_num_paths = 1 
  if fname in num_paths:
    print(f"num_paths: {num_paths[fname]}")
    curr_num_paths = np.mean(num_paths[fname])

  slowdown = mean / num_samples / curr_num_paths / crisp_time[program]
  print(f"{mean} / {num_samples} / {curr_num_paths} / {crisp_time[program]} == {slowdown}")
  #slowdown = mean / crisp_time[program]
  #print(f"{fname}, mean: {mean:.2f} +- {ci:.2f}, stddev: {std:.2f}, slowdown: {slowdown:.2f}")

  row = {'Program': program, 'Estimator': fname_to_label(fname), 'Slowdown': slowdown}

  print(f"slowdown: {slowdown}")

  df = df.append(row, ignore_index=True)

print(df)
  
#df = df.pivot(columns='Program', index='Estimator', values='Slowdown')
df = df.pivot(columns='Program', index='Estimator', values='Slowdown')
#print(df)
#exit()
df = df.sort_values(by='Estimator', key=lambda ser: sort_estimator_strs(ser))
program_order = ['HOTEL', 'EPIDEMICS', 'AC', 'TRAFFIC 5x5']
#df = df.sort_index(axis=1)
df = df[program_order]
#df = df.sort_values(by='Program')

prev_prefix = None
def shorten_label(l):
  global prev_prefix
  slash_pos = l.rfind("/")
  if slash_pos < 0:
    return l

  prefix = l[:l.rfind("/")]
  suffix = l[l.rfind("/") + 1:]

  if prev_prefix and l.startswith(prev_prefix):
    return suffix
  else:
    prev_prefix = prefix
    return l

df.index = df.index.map(lambda l: shorten_label(l))
print(df.index)

#prev = None
#for i in range(len(df.columns.values)):
#  print(df.columns.values[i])
#  exit()
#  df.columns.values[i] = (df.columns.values[i][0], "test")
#  #if prev and df.columns.values[i][1].startswith(prev):
#  #  df.columns.values[i][1] = "test"
#  #else:
#  #  prev = df.columns.values[i][1]

#print(df.columns.values)
#exit()
#
#print(df.index)

##df.reset_index(inplace=True)
#print(df)
#
#prev = None
#for idx, row in df.iterrows():
#  print(row.name)

cmap=sns.color_palette("Blues", as_cmap=True)

sns.set(rc={'figure.figsize': (6, 6)})
labels=np.zeros((25, 4), dtype=object)

for row_i, (idx, row) in enumerate(df.iterrows()):
  for i, program in enumerate(('HOTEL', 'EPIDEMICS', 'AC', 'TRAFFIC 5x5')):
    val_str = "{:.2g}".format(row[program])

    print(f"got val_str {val_str}, row[program] was {row[program]}")

    m = re.search("(.+)e\+0(.+)", val_str)
    if m:
      val_str = round(float(m.group(1)) * 10**int(m.group(2)))
    
    labels[row_i][i] = val_str

print(labels)
print(df)
ax = sns.heatmap(df, cmap=cmap, annot=labels, fmt="", cbar=False, norm=LogNorm())
ax.hlines([1, 5, 9, 13, 17, 21], *ax.get_xlim(), color="white")
plt.xlabel('')

plt.tight_layout(rect=[0, 0, 1, 1]) # absolute madness
#fig.text(0.45, 0.07, "Portion of Time Budget [%]")
plt.savefig("slowdown.pdf")

print("max ci proportion:", max_ci_proportion)
