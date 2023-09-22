#!/usr/bin/python3

import os
import sys
import pandas as pd

results_path = sys.argv[1]
ctime = int(sys.argv[2]) * 1e6

prog_substr = { 'HOTEL': 'hotel',
                'EPIDEMICS': 'epidemics',
                'AC': 'ac',
                'TRAFFIC\n10x10': 'traffic_grid_populations_10x10',
                'TRAFFIC\n20x20': 'traffic_grid_populations_20x20'
              }


est_substrs = { 'DGSI': ('dgsi_num_paths',),
                'DGO': ('dgo',),
                'PGO': ('pgo',),
                'RF': ('reinforce',),
                'IPA': ('enable_ad=True',),
                'GA': ('GA',),
                'SA': ('SA',)
              }

for p, p_substr in prog_substr.items():
  for e, e_substrs in est_substrs.items():
    all_substrs = e_substrs + (p_substr,) + ("mean_time",)

    min_y_crisp = float("inf")
    min_y_crisp_fname = None
    
    for fname in os.listdir(results_path):
      found_all = True
      for substr in all_substrs:
        if substr not in fname:
          found_all = False
          break
      if not found_all:
        continue
    
      df = pd.read_csv(results_path + "/" + fname)
    
      idx = df[df.cumulative_time >= ctime].first_valid_index()
    
      if idx and df.iloc[idx].y_crisp < min_y_crisp:
          min_y_crisp = df.iloc[idx].y_crisp
          min_y_crisp_fname = fname
    
    if min_y_crisp_fname:
      print(p, e, all_substrs)
      print("min_y_crisp:", min_y_crisp)
      print("min_y_crisp_fname:", min_y_crisp_fname)
      print()
