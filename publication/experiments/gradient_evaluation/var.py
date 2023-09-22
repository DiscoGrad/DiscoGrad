#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dfs = []
for i, s in enumerate(sys.argv[1:]):
  if s.endswith('.txt'):
    dfs.append((s, pd.read_csv(s)))
  else:
    break

#i += 1

ys = []
for j in range(i + 1, len(sys.argv)):
  ys.append(sys.argv[j])

num_rows = len(dfs[0][1][ys[0]])

for y in ys:
  for row_idx in range(num_rows):
    row = [df[1][y][row_idx] for df in dfs]
    print(np.var(row))
