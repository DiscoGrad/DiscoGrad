#!/bin/python3
# usage: python generate.py prog_name num_dims stddev
# stdin = optimal parameter conf
# stdout = experiment file

import sys
import fileinput
import pprint
import argparse
import time

import numpy as np

# read from cli args
parser = argparse.ArgumentParser(prog="run.py", description="stdin = optimal parameter configuration; stdout = experiment file")
parser.add_argument("prog_name", type=str)
parser.add_argument("prog_seed", type=int)
parser.add_argument("prog_reps", type=int)
parser.add_argument("num_params", type=int)
parser.add_argument("stddev", type=float)
parser.add_argument("num_points", type=int, default=100)
parser.add_argument("lower_value", type=float, default=1)
parser.add_argument("upper_value", type=float, default=0)
parser.add_argument("--seed", "-s", type=int, default=42)
args = parser.parse_args()

if args.seed:
  np.random.seed(args.seed)
else:
  np.random.seed(time.time())

# read optimal params and ranges from stddin
opt_params = []
lower_ranges = []
upper_ranges = []
for idx, value in enumerate("".join(fileinput.input(files=("-",))).split("\n")):
  if idx < args.num_params:
    opt_params.append(float(value))
    lower_ranges.append(args.lower_value)
    upper_ranges.append(args.upper_value)
  #elif idx < 2 * nparams:
  #  lower_ranges.append(float(value))
  #elif idx < 3 * nparams:
  #  upper_ranges.append(float(value))

prog_template = { 'name': args.prog_name, 'stddevs': args.stddev, 'seed': (args.prog_seed,), 'nreps': (args.prog_reps,), 'params': tuple(opt_params) }

estimators_template = (
  { 'name': 'dgsi', 'params': { 'num_paths': (4,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (8,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (64,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths': (4,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (8,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },
  #{ 'name': 'dgsi', 'params': { 'num_paths': (64,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'si_stddev_proportion': 1, 'num_samples': (1,) } },

  #{ 'name': 'dgo', 'params': { 'num_samples': (10) } },
  { 'name': 'dgo', 'params': { 'num_samples': (100) } },
  { 'name': 'dgo', 'params': { 'num_samples': (1000) } },
  { 'name': 'dgo', 'params': { 'num_samples': (10000) } },

  #{ 'name': 'pgo', 'params': { 'num_samples': (10) } },
  { 'name': 'pgo', 'params': { 'num_samples': (100) } },
  { 'name': 'pgo', 'params': { 'num_samples': (1000) } },
  { 'name': 'pgo', 'params': { 'num_samples': (10000) } },

  #{ 'name': 'reinforce', 'params': { 'num_samples': (10) } },
  { 'name': 'reinforce', 'params': { 'num_samples': (100) } },
  { 'name': 'reinforce', 'params': { 'num_samples': (1000) } },
  { 'name': 'reinforce', 'params': { 'num_samples': (10000) } },

  { 'name': 'pgo', 'params': { 'num_samples': (500000) } }, # reference for mse calculation = last entry in the list
)

# generate experiments to calculate gradient over a random range for each the first 25 dimensions (or all, if the problem has less than 25 dimensions)
progs = []
for dim in range(min(25, args.num_params)):
  prog = prog_template.copy()
  params = opt_params.copy()
  # sample in the specified range around the optimal value
  params[dim] = sorted(np.random.uniform(params[dim] - lower_ranges[dim], params[dim] + upper_ranges[dim], args.num_points))
  ## sample in the specified range
  #params[dim] = sorted(np.random.uniform(lower_ranges[dim], upper_ranges[dim], args.num_points))
  prog.update(params=tuple(params))
  progs.append(prog)

print("programs = ", end="")
pprint.pprint(tuple(progs), width=512)

print("estimators = ", end="")
pprint.pprint(estimators_template, width=512)

