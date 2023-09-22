#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
import scipy.stats

ref_df = pd.read_csv(sys.argv[1]) #[:20]
dfs = []

print(f"reference is {sys.argv[1]}")

for i, s in enumerate(sys.argv[2:]):
  if s.endswith('.txt'):
    df = pd.read_csv(s) #[:20]
    if len(df) != len(ref_df): # or df.isnull().values.any():
      print(f"skipping {s}")
      continue
    dfs.append((s, df))
  else:
    break

i += 1

ys = []
for j in range(i + 1, len(sys.argv)):
  ys.append(sys.argv[j])

sign_agreement = defaultdict(defaultdict)
mse = defaultdict(defaultdict)
cc = defaultdict(defaultdict)
sdmse = defaultdict(defaultdict)
for df in dfs:
  for y in ys:
    mse[y][df[0]] = (np.mean((df[1][y] - ref_df[y])**2))
    sign_agreement[y][df[0]] = (df[1][y] * ref_df[y] >= 0).sum() / len(df[1][y])
    sdmse[y][df[0]] = mse[y][df[0]] * (1 - sign_agreement[y][df[0]])

    try:
      cc[y][df[0]] = scipy.stats.pearsonr(df[1][y], ref_df[y])[0]
    except:
      pass

for y in ys:
  print(f"mse of {y}:")
  for k, v in sorted(mse[y].items(), key=lambda item: item[1]):
    print(f"{k}: {v}")
  print()

for y in ys:
  print(f"ratio of correct signs of {y}:")
  for k, v in sorted(sign_agreement[y].items(), key=lambda item: item[1], reverse=True):
    print(f"{k}: {v}")
  print()

# note: does not seem helpful to judge derivative estimates
for y in ys:
  print(f"cross correlation of {y}:")
  for k, v in sorted(cc[y].items(), key=lambda item: item[1], reverse=True):
    print(f"{k}: {v}")
  print()

#for y in ys:
#  print(f"sdmse {y}:")
#  for k, v in sorted(sdmse[y].items(), key=lambda item: item[1]):
#    print(f"{k}: {v}")
#  print()
