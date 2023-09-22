#!/usr/bin/python3

import pandas
import sys
from math import sqrt, floor
import os
import time

df = pandas.read_csv(sys.argv[1])
col_prefix = sys.argv[2] if len(sys.argv) > 2 else "dydx"
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

x_cols = list(filter(lambda x: x.startswith(col_prefix), df.columns))

row_width = int(sqrt(len(x_cols)))

for step, row in df.iterrows():
  os.system("clear")

  print(f"step {step}")
  print()

  for j, p in enumerate(x_cols):
    row[p] = float(row[p])
    #if row[p] == 0.0:
    print(f"{row[p]:2.2f} ", end='')
    #else:
    #  print(f"\033[1m\033[91m{int(row[p]):4d}\033[0m.{int((row[p] - floor(row[p])) * 10)} ", end='')
    if (j + 1) % row_width == 0:
      print()
      print()
  time.sleep(delay)
  #print(" ".join(map(str, list(row[x_cols]))))
