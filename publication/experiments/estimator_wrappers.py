# Python wrappers to run the a certain program binary that was smoothed by a certain estimator.
# Extracts the output of the program (output value, gradient and time) using regular expressions on the stdout.

import subprocess
import re
import math
import numpy as np
from time import time as pythons_time

discograd_path = '../../programs' # location of the folder where the program binaries reside

# compile regexes for better performance
re_exp = re.compile("expectation: (.+)")
re_pert = re.compile("perturbation: (.+)")
re_deriv = re.compile("derivative: (.+)")
re_time = re.compile("estimation_duration: (.+)us")

def call_cpp_program(cmd, prog_params):
  """Safely call the program using subprocess and return useful information upon error."""
  cmd = "../" + cmd
  try:
    return subprocess.run(cmd.split(), shell=False, input=prog_params, capture_output=True, text=True, check=True)
  except subprocess.CalledProcessError as ex:
    print(f"Discograd aborted with exception:\n{ex}.")
    print(f"Command to reproduce:\necho \"{prog_params}\" | {cmd}")
    exit(-1)

def crisp(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """
  Runs discograd-generated crisp program (i.e. the cpp program without smoothing).
  When providing a number of samples, it is possible to get the program output for
  different random perturbations, for example to realise infinitesimal perturbation 
  analysis (IPA).
  """
  if 'num_samples' not in estim_param_comb:
    estim_param_comb['num_samples'] = 1

  # disable perturbation if only one sample is requested
  if estim_param_comb['num_samples'] == 1: 
    stddev = 0

  # detect whether multiple parameter vectors were passed as a list
  num_param_combs = 1
  if isinstance(prog_param_comb[0], (int, float, np.float64)):
    prog_params = "\n".join(map(str, prog_param_comb))
  else:
    num_param_combs = len(prog_param_comb)
    prog_params = "\n".join(["\n".join(map(str, comb)) for comb in prog_param_comb])

  executable = f"{prog}_crisp"
  enable_ad = False
  if "enable_ad" in estim_param_comb and estim_param_comb["enable_ad"]:
    executable = f"{prog}_crisp_ad"
    enable_ad = True

  cmd = f"{discograd_path}/{executable} -s {seed} --nc {num_param_combs} --nr {nreps} --var {stddev**2} --ns {estim_param_comb['num_samples']}"
  proc = call_cpp_program(cmd, prog_params)

  expectations = []
  derivs = []
  time = 0

  for line in proc.stdout.splitlines():
    m = re_time.match(line)
    if m:
      time += float(m.group(1))
    m = re_exp.match(line)
    if m:
      expectations.append(float(m.group(1)))
    if enable_ad:
      m = re_deriv.match(line)
      if m:
        derivs.append(float(m.group(1)))

  if derivs == []:
    derivs = np.zeros(len(prog_param_comb))

  if len(expectations) == 1:
    expectations = expectations[0]

  return expectations, np.array(derivs), time

# TODO: this may be replacable by exec_crisp now...
def exec_perturbed(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """Runs discograd-generated crisp program with random perturbations"""
  prog_params = "\n".join(map(str, prog_param_comb))
  executable = f"{prog}_pgo"
  num_param_combs = 1
  cmd = f"{discograd_path}/{executable} -s {seed} --nc {num_param_combs} -nr {nreps} --var {stddev**2} --ns {estim_param_comb['num_samples']}"
  proc = call_cpp_program(cmd, prog_params)

  expectations = []
  perturbations = []
  derivs = []
  time = -99.0

  for line in proc.stdout.splitlines():
    m = re_time.match(line)
    if m:
      time = float(m.group(1))
    m = re_exp.match(line)
    if m:
      expectations.append(float(m.group(1)))
      perturbations.append([])
      derivs.append([])
    m = re_pert.match(line)
    if m:
      perturbations[-1].append(float(m.group(1)))
    m = re_deriv.match(line)
    if m:
      derivs[-1].append(float(m.group(1)))

  return np.asarray(expectations), np.asarray(perturbations), np.asarray(derivs), time

def dgo(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """Runs discograd-generated program that utilises the DGO gradient estimate."""
  prog_params = "\n".join(map(str, prog_param_comb))
  executable = f"{prog}_dgo"
  num_param_combs = 1
  cmd = f"{discograd_path}/{executable} -s {seed} --nc {num_param_combs} --nr {nreps} --var {stddev**2} --ns {estim_param_comb['num_samples']}"
  proc = call_cpp_program(cmd, prog_params)

  expectations = []
  derivs = []
  time = -99.0

  for line in proc.stdout.splitlines():
    m = re_time.match(line)
    if m:
      time = float(m.group(1))
    m = re_exp.match(line)
    if m:
      expectations.append(float(m.group(1)))
      derivs = []
    m = re_deriv.match(line)
    if m:
      derivs.append(float(m.group(1)))

  return expectations[0], np.asarray(derivs), time

def dgsi(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """Runs discograd-generated program that utilizes the SI backend."""
  prog_params = "\n".join(map(str, prog_param_comb))

  if not 'si_stddev_proportion' in estim_param_comb:
    estim_param_comb['si_stddev_proportion'] = 1

  si_stddev = stddev * estim_param_comb['si_stddev_proportion']
  sampling_stddev = np.sqrt(stddev**2 - si_stddev**2)

  expectations = []
  derivs = []
  times = []
 
  num_paths = []

  for _ in range(estim_param_comb['num_samples']):
    normals = np.random.normal(0, sampling_stddev, len(prog_param_comb))
    perturbed_prog_param_comb = prog_param_comb + normals
    prog_params = ' '.join(map(str, perturbed_prog_param_comb))
    num_param_combs = 1

    cmd = f"{discograd_path}/{prog}_dgsi -s {seed} --nc {num_param_combs} --nr {nreps} --var {si_stddev**2} --np {estim_param_comb['num_paths']} --rm {estim_param_comb['restrict_mode']} --up_var {estim_param_comb['use_dea']}"
    proc = call_cpp_program(cmd, prog_params)

    for line in proc.stdout.splitlines():
      m = re_time.match(line)
      if m:
        times.append(float(m.group(1)))
      m = re.search("expectation: (.+)", line)
      if m:
        expectations.append(float(m.group(1)))
        deriv_idx = 0
      m = re.search("derivative: (.+)", line)
      if m:
        if len(derivs) <= deriv_idx:
          derivs.append([])
        derivs[deriv_idx].append(float(m.group(1)))
        deriv_idx += 1
      m = re.search("average number of paths: (.+)", line)
      if m:
        num_paths.append(float(m.group(1)))

  if 'return_num_paths' in estim_param_comb:
    return np.mean(expectations), np.asarray(list(map(np.mean, derivs))), np.sum(times), np.mean(num_paths)

  return np.mean(expectations), np.asarray(list(map(np.mean, derivs))), np.sum(times)

def ipa(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """Runs infinitesimal perturbation analysis (IPA) on the program, by using the crisp backend."""
  expectations, deltas, derivs = exec_perturbed(prog, stddev, prog_param_comb, estim_param_comb, True)

  averaged_derivs = [0] * len(prog_param_comb)
  for row in derivs:
    for i in range(len(row)):
      averaged_derivs[i] += row[i]

  averaged_derivs = list(map(lambda x: x / estim_param_comb['num_samples'], averaged_derivs))

  return np.mean(expectations), np.asarray(averaged_derivs)

def reinforce(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """
  Runs discograd-generated program using the reinforce backend.
  REINFORCE paper: Williams, Ronald J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229-256, 1992.
  (https://doi.org/10.1007/BF00992696)
  For our specific application, refer to the discograd paper.
  """
  prog_params = "\n".join(map(str, prog_param_comb))
  executable = f"{prog}_reinforce"
  num_param_combs = 1
  cmd = f"{discograd_path}/{executable} -s {seed} --nc {num_param_combs} --nr {nreps} --var {stddev**2} --ns {estim_param_comb['num_samples']}"
  proc = call_cpp_program(cmd, prog_params)

  expectations = []
  derivs = []
  time = -99.0

  for line in proc.stdout.splitlines():
    m = re_time.match(line)
    if m:
      time = float(m.group(1))
    m = re_exp.match(line)
    if m:
      expectations.append(float(m.group(1)))
    m = re_deriv.match(line)
    if m:
      derivs.append(float(m.group(1)))

  return expectations[0], np.asarray(derivs), time

def pgo(prog, stddev, seed, nreps, prog_param_comb, estim_param_comb):
  """
  Runs the discograd-generated program using the PGO for gradient estimation.
  Basic scheme discussed in B. Polyak, Introduction to Optimization. Optimization Software - Inc., Publications Division, New York, 1987.
  (https://www.researchgate.net/publication/342978480_Introduction_to_Optimization)
  Convergence of optimization scheme (random search) analysed in Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions. Foundations of Computational Mathematics, 17:527-566, 2017.
  (https://doi.org/10.1007/s10208-015-9296-2)
  """
  prog_params = "\n".join(map(str, prog_param_comb))
  executable = f"{prog}_pgo"
  num_param_combs = 1
  cmd = f"{discograd_path}/{executable} -s {seed} --nc {num_param_combs} --nr {nreps} --var {stddev**2} --ns {estim_param_comb['num_samples']}"
  proc = call_cpp_program(cmd, prog_params)

  expectations = []
  derivs = []
  time = -99.0

  for line in proc.stdout.splitlines():
    m = re_time.match(line)
    if m:
      time = float(m.group(1))
    m = re_exp.match(line)
    if m:
      expectations.append(float(m.group(1)))
    m = re_deriv.match(line)
    if m:
      derivs.append(float(m.group(1)))

  return expectations[0], np.asarray(derivs), time

