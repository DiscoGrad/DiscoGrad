#!/usr/bin/python3

import numpy as np
import sys
import subprocess
import re
import argparse
import hashlib
from itertools import product
from time import time as pythons_time
from multiprocessing import Process, Pool, cpu_count, set_start_method
from multiprocessing import Process
from itertools import product
from heapq import heappush, heappop
from datetime import datetime, timedelta
import os

sys.path.append('..')
from optimizers import RandomGrad, GA, Rand, RandMut, ConstrainedOpt, ClipOpt
from ensmallen_optimizers import SA
from estimator_wrappers import *

def run(args):
  if len(args) == 4:
    prog, estim, opt, rep_range = args
  else:
    prog, estim, opt = args
    rep_range = range(num_macroreps)

  for rep in rep_range:
    #seed = (id(prog) ^ id(estim) ^ id(rep)) % 2**32
    #np.random.seed(seed)
    # somewhat arbitrary, but reproducible:
    #seed = ((sum(map(ord, [*prog["name"]])) ^ int(1000 * sum(prog["stddevs"]))) ^ (6700417 * (rep + 1))) % 2**32
   
    seed_str = "".join([str(v) if k != "params" else "" for k, v in prog.items()])
    seed_str += str(rep)
    seed_str += str(global_seed)
    seed = int.from_bytes(hashlib.sha256(seed_str.encode()).digest()[:4], "big")
    rs = np.random.RandomState(seed)
    for k, v in estim['params'].items():
      if isinstance(v, (float, int)):
        estim['params'][k] = (v,)

    estim_param_names = list(estim['params'].keys())
    estim_param_vals = list(estim['params'].values())
    estim_param_combs = tuple(dict(zip(estim_param_names, list(param_val))) for param_val in product(*estim_param_vals))
    
    for estim_param_comb in estim_param_combs:
      estim_param_str = '-'.join([f"{k}={v}" for k, v in estim_param_comb.items()])
      if isinstance(prog['stddevs'], (float, int)):
        prog['stddevs'] = (prog['stddevs'],)
      if isinstance(prog['seed'], (float, int)):
        prog['seed'] = (prog['seed'],)
      if isinstance(prog['nreps'], (float, int)):
        prog['nreps'] = (prog['nreps'],)
      for stddev in prog['stddevs']:
        for seed in prog['seed']:
          for nreps in prog['nreps']:
            overall_start_time = pythons_time()
            prog_name_clean = prog['name'].replace('/', '_')

            opt_str = str(opt).replace('(', '-').replace('=', '_').replace(', ', '-').replace('))', '').replace(')', '')
            fname = f"{out_path}/{prog_name_clean}_stddev={stddev:.1g}_seed={seed}_nreps={nreps}-{estim['name']}_{estim_param_str}_{opt_str}-rep_{rep:04d}.txt"
            param_fname = f"{out_path}/{prog_name_clean}_stddev={stddev:.1g}_seed={seed}_nreps={nreps}-{estim['name']}_{estim_param_str}_{opt_str}-rep_{rep:04d}_final_params.txt"
            if os.path.exists(param_fname):
              print(f"{param_fname} exists, skipping macroreplication")
              continue

            print(f"{fname} started")
       
            with open(fname, 'w') as f:

              prog_params = np.asarray(prog['params'](rs))
              num_prog_params = len(prog_params)
              if not dump_params:
                f.write("step,y,y_crisp,cumulative_time\n")
              else:
                f.write("step,y,y_crisp," + ','.join([f"x{i}" for i in range(num_prog_params)]) + ',' + ','.join([f"dydx{i}" for i in range(num_prog_params)]) + ",deriv_norm,cumulative_time\n")
              cumulative_time = 0.0

              #output, derivs = globals()[estim['name']](prog['name'], stddev, prog_params, estim_param_comb)
              opt.fitness = globals()[estim["name"]]
              output, derivs, time = opt.initialize(
                globals()[estim["name"]],
                dict(prog=prog["name"], prog_param_comb=prog_params, stddev=stddev, seed=seed, nreps=nreps, estim_param_comb=estim_param_comb),
                param_init_func=prog['params']
              )
              if estim["name"] == "crisp":
                output_crisp = output
              else:
                output_crisp = crisp(prog=prog["name"], prog_param_comb=prog_params, stddev=0, seed=seed, nreps=nreps, estim_param_comb={})[0] # run crisp version without AD
              cumulative_time += time

              if not dump_params:
                f.write("0," + str(output) + "," + str(output_crisp) + "," + str(cumulative_time) + "\n")
              else:
                f.write("0," + str(output) + "," + str(output_crisp) + ',' + ','.join(map(str, prog_params)) + ',' + ','.join(map(str, derivs)) + "," + str(np.linalg.norm(derivs)) + "," + str(cumulative_time) + "\n")
              f.flush()

              step = 0
              prog_time_limit = prog["time_limit"] if "time_limit" in prog else float("inf")
              if type(prog_time_limit) is tuple:
                prog_time_limit = prog_time_limit[0]

              while step < num_opt_steps and cumulative_time < time_limit * 1e6 and cumulative_time < prog_time_limit * 1e6:
                step += 1
                #prog_params = opt.update(prog_params, derivs, None, output)
                #output, derivs = globals()[estim['name']](prog['name'], stddev, prog_params, estim_param_comb)
                output, derivs, prog_params, time = opt.update(
                  globals()[estim["name"]],
                  dict(prog=prog["name"], prog_param_comb=prog_params, stddev=stddev, seed=seed, nreps=nreps, estim_param_comb=estim_param_comb)
                )
                output_crisp = crisp(prog=prog["name"], prog_param_comb=prog_params, stddev=stddev, seed=seed, nreps=nreps, estim_param_comb={})[0] # run crisp version without AD
                cumulative_time += time
                if not dump_params:
                  f.write(str(step) + "," + str(output) + "," + str(output_crisp) + "," + str(cumulative_time) + "\n")
                else:
                  f.write(str(step) + "," + str(output) + "," + str(output_crisp) + ',' + ','.join(map(str, prog_params)) + ',' + ','.join(map(str, derivs)) + "," + str(np.linalg.norm(derivs)) + "," + str(cumulative_time) + "\n")
                f.flush()

            with open(param_fname, "w") as param_f:
              param_f.write(",".join([f"x{i}" for i in range(num_prog_params)]) + "\n")
              param_f.write(",".join(map(str, prog_params)) + "\n")

            overall_finish_time = pythons_time()
            print(f"{fname} took {overall_finish_time - overall_start_time:.2f}s (cumulative estimation time was {cumulative_time*1e-6:.2f}s)")

def filter_pool_args(pool_args): # drop meaningless combinations of estimator and optimizers, parallelize macroreps if enabled
  out_args = []

  for prog, estim, opt in pool_args:
    is_global_search = lambda o: isinstance(o, GA) or isinstance(o, SA) or isinstance(o, Rand) or isinstance(o, RandMut)
    if (is_global_search(opt) or
      (isinstance(opt, ConstrainedOpt) and is_global_search(opt.opt)) or
      (isinstance(opt, ClipOpt) and is_global_search(opt.opt))):

      if estim["name"] != "crisp":
        continue
      
      if "enable_ad" in estim["params"] and estim["params"]["enable_ad"]:
        continue

    else: # no crisp without AD when using a gradient-based optimizer
      if estim["name"] == "crisp":
        if not "enable_ad" in estim["params"] or not estim["params"]["enable_ad"]:
          continue

    # only use crisp + AD with RandomGrad
    if isinstance(opt, RandomGrad):
      if estim["name"] != "crisp":
        continue

      if not "enable_ad" in estim["params"] or not estim["params"]["enable_ad"]:
        continue

    if parallelize_macroreps:
      for rep in range(num_macroreps):
        out_args.append((prog, estim, opt, (rep,)))
    else:
      out_args.append((prog, estim, opt))

  if parallelize_macroreps: # prioritize progress across all prog, estim, opt combinations rather than reps
    return sorted(out_args, key=lambda x: x[3])

  return out_args

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="optimize.py")
  parser.add_argument("experiment_path", type=str)
  parser.add_argument("--num_steps", "-s", type=int, default=float("inf"))
  parser.add_argument("--num_macroreps", "-r", type=int, default=1)
  parser.add_argument("--parallelize_macroreps", "-pm", default=False, action='store_true')
  parser.add_argument("--max_processes", "-p", type=int, default=cpu_count())
  parser.add_argument("--time_limit", "-t", type=int, default=float("inf"))
  parser.add_argument("--dump_params", "-dp", default=False, action='store_true')
  parser.add_argument("--global_seed", "-gs", type=int, default=0)
  args = parser.parse_args()
  num_opt_steps = args.num_steps
  num_macroreps = args.num_macroreps
  parallelize_macroreps = args.parallelize_macroreps
  time_limit = args.time_limit
  dump_params = args.dump_params
  global_seed = args.global_seed

  sys.path.append(args.experiment_path)
  from experiment import programs, estimators, optimizers

  pool_args = filter_pool_args(list(product(programs, estimators, optimizers)))
  num_parallel_processes = min(len(pool_args), args.max_processes)

  print(f"running {len(pool_args) * num_macroreps if not parallelize_macroreps else len(pool_args)} configurations using {num_parallel_processes} processes")

  if num_opt_steps == float("inf"): # otherwise we don't know
    busy_times = []
    [heappush(busy_times, 0) for _ in range(num_parallel_processes)] 
    for a in pool_args:
      param_comb_time_s = time_limit
      if "time_limit" in a[0]:
        prog_time_limit = a[0]["time_limit"]
        if type(prog_time_limit) is tuple:
          prog_time_limit = prog_time_limit[0]
        param_comb_time_s = min(param_comb_time_s, prog_time_limit)
      start_time = heappop(busy_times)
      if parallelize_macroreps:
        heappush(busy_times, start_time + param_comb_time_s)
      else:
        heappush(busy_times, start_time + param_comb_time_s * num_macroreps)
    par_time_s = 0
    while busy_times:
      par_time_s = max(par_time_s, heappop(busy_times))
    if par_time_s != float("inf"):
      now = datetime.now()
      finish_time = now + timedelta(seconds=par_time_s)
      print(f"disregarding process startup overheads, experiment will take about {par_time_s / 3600:.1f} hours, estimated finish time: {finish_time.strftime('%Y-%m-%d, %H:%M')}")

  pool = Pool(num_parallel_processes)
  pool.map(run, pool_args)
