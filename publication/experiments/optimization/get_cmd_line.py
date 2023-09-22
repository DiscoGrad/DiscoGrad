#!/usr/bin/python3

import sys
import re
import pandas

fname = sys.argv[1]

m = re.search("stddev=(.+?)-", fname)
var = float(m[1])**2

m = re.search("num_paths=(.+?)-", fname)
num_paths = m[1] if m else -1

m = re.search("restrict_mode=(.+?)-", fname)
restrict_mode = m[1] if m else -1

m = re.search("use_dea=(.+?)-", fname)
use_dea = m[1] if m else -1

m = re.search("num_samples=(.+?)_", fname)
num_samples = m[1]

df = pandas.read_csv(fname)

x_cols = list(filter(lambda x: x.startswith("x"), df.columns))

initial_params = None
initial_y = None

lowest_y_params = None
lowest_y = float("inf")
for i, row in df.iterrows():
  if not initial_params:
    initial_y = row['y']
    initial_params = lowest_y_params = " ".join(map(str, list(row[x_cols])))

  if row['y'] < lowest_y:
    lowest_y = row['y']
    lowest_y_params = " ".join(map(str, list(row[x_cols])))


print(f"initial params (y == {initial_y}): {initial_params} {var} {num_paths} {restrict_mode} {use_dea} {num_samples}")
print(f"lowest y params (y == {lowest_y}): {lowest_y_params} {var} {num_paths} {restrict_mode} {use_dea} {num_samples}")
