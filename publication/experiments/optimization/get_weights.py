#!/usr/bin/python3
# Return the weights of a neural network in a form
# that can be used for piping into other commands.

import sys
import re
import pandas

fname = sys.argv[1]
flag = sys.argv[2]

if flag != "-b" and flag != "-i":
  print("unknown option", flag)
  print("usage: get_weights.py fname [-b|-i]")

df = pandas.read_csv(fname)

x_cols = list(filter(lambda x: x.startswith("x"), df.columns))

initial_params = None

lowest_y_params = None
lowest_y = float("inf")

for i, row in df.iterrows():
  if not initial_params:
    initial_params = lowest_y_params = " ".join(map(str, list(row[x_cols])))

  if flag == "-b" and row['y'] < lowest_y:
    lowest_y = row['y']
    lowest_y_params = " ".join(map(str, list(row[x_cols])))


if flag == "-b":
  print(lowest_y_params)
elif flag == "-i":
  print(initial_params)

