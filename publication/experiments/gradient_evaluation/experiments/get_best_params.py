import os
import glob
import pandas as pd
import numpy as np

results_dirs = ["hotel_05_08", "traffic_2x2_05_09", "traffic_5x5_05_09"]#, "epidemics_11_08"]
results_exps = ["hotel", "traffic2x2", "traffic5x5"]#, "epidemics"]

for result, exp in zip(results_dirs, results_exps):
  results_dir = os.path.join("/usr/data_share/si_ad_ieee_access/optimization/", result)
  # determine minimum y value
  min_y = float("inf")
  files = []
  for file in glob.glob(results_dir + "/*[0-9].txt"):
    data = pd.read_csv(file)
    last_y_val = data["y_crisp"].iloc[-1]
    files.append((file, last_y_val))
    print(file, last_y_val)
    if last_y_val < min_y:
      min_y = last_y_val
  print("minimum value is", min_y)
  # determine all the files that produced this value
  min_files = []
  for file, last_y_val in files:
    if last_y_val <= min_y:
      min_files.append(file)
  print("found", len(min_files), "estimator confs that produce the same minimum. Choosing the smallest final params.")
  # choose the one with the smallest parameters 
  min_abs_norm = float("inf")
  min_file = ""
  for file in min_files:
    min_params = pd.read_csv(os.path.splitext(file)[0]+"_final_params.txt")
    abs_norm = np.linalg.norm(min_params.iloc[0].to_numpy())
    if abs_norm < min_abs_norm:
      min_abs_norm = abs_norm
      min_file = file
  print("choosing", min_file, "because it has the smallest abs norm", min_abs_norm)
  min_params = pd.read_csv(os.path.splitext(min_file)[0]+"_final_params.txt")
  print("params", min_params.iloc[0].to_numpy())
  with open(os.path.join(exp, "ref_args.txt"), "w") as file:
    for x in min_params.iloc[0].to_numpy():
      file.write(str(x)+"\n")
