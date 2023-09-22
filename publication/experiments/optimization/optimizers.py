from numpy_ml.neural_nets.optimizers import *
import numpy as np
from typing import Callable, Tuple, Dict
from abc import ABC
from ensmallen_optimizers import SA
from pyeasyga import GeneticAlgorithm
from time import time as pytime

# Abstract base class for all optimizers
class Optimizer(ABC):
  def __init__(self):
    pass
  # initialize the state of the optimizer based on the objective and initial parameters
  def initialize(self, func, args, param_init_func) -> Tuple[np.ndarray, np.ndarray]:
    pass
  # update the current parameter conf. using func and return output, gradient and new parameter conf.
  def update(self, func: Callable, args: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pass
  def __str__(self):
    return f"AbstractOptimizer()"

class ClipOpt(Optimizer):
  def __init__(self, opt, max_deriv):
    self.opt = opt
    self.max_deriv = max_deriv
  def initialize(self, func, args, param_init_func):
    return self.opt.initialize(func, args, param_init_func)
  def update(self, func, args):#params, derivs, name, output):
    output, derivs, params, time = self.opt.update(func, args)
    for i in range(len(derivs)):
      if abs(derivs[i]) > self.max_deriv:
        derivs[i] = min(derivs[i], self.max_deriv)
        derivs[i] = max(derivs[i], -self.max_deriv)
    return output, derivs, params, time

  def __str__(self):
    return f"ClipOpt({self.opt})"

class ConstrainedOpt(Optimizer):
  def __init__(self, opt, lower, upper):
    self.opt = opt
    self.lower = lower
    self.upper = upper
  def initialize(self, func, args, param_init_func):
    return self.opt.initialize(func, args, param_init_func)
  def update(self, func, args):#params, derivs, name, output):
    output, derivs, params, time = self.opt.update(func, args)
    for i, p in enumerate(params):
      if p < self.lower:
        params[i] -= np.fmod(p - self.lower, self.upper - self.lower)
      elif p > self.upper:
        params[i] -= np.fmod(p - self.upper, self.upper - self.lower)
    return output, derivs, params, time

  def __str__(self):
    return f"COpt({self.opt})"

# walk through the parameter space on a random trajectory
class RandomWalk(Optimizer):
  def __init__(self, lr=1e-2):
    self.lr = lr
  def initialize(self, func, args, param_init_func):
    return func(**args)
  def update(self, func, args):
    params += np.random.uniform(-self.lr, self.lr, len(params))
    output, derivs, time = func(**args)
    return output, derivs, params, time
  def __str__(self):
    return f"RandomWalk(lr={self.lr})"

# blindly test values within the range
class Random(Optimizer):
  def __init__(self, range_=(0, 1)):
    self.range = range_
  def initialize(self, func, args, param_init_func):
    return func(**args)
  def update(self, func, args):
    output, derivs, time = func(**args)
    return output, derivs, np.random.uniform(self.range[0], self.range[1], len(args["prog_param_comb"])), time
  def __str__(self):
    return f"Random(range={self.range})"

# gradient descent, but use noise if the gradient dies
class RandomGrad(Optimizer):
  def __init__(self, range_=(0, 1), min_grad_norm=1e-2, grad_opt=Adam()):
    self.range = range_
    self.min_grad_norm = min_grad_norm
    self.grad_opt = grad_opt
    self.derivs = np.array([])
    self.output = np.array([])
  def initialize(self, func, args, param_init_func):
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, time
  def update(self, func, args):
    if np.linalg.norm(self.derivs) < self.min_grad_norm:
      self.derivs = np.random.uniform(-self.range[0], self.range[1], len(args["prog_param_comb"]))
    prog_params = self.grad_opt.update(args["prog_param_comb"], self.derivs, None, self.output)
    args.update(prog_param_comb=prog_params)
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, prog_params, time
  def __str__(self):
    return f"RandomGrad(range={self.range}, min_grad_norm={self.min_grad_norm}, grad_opt={self.grad_opt})"

# gradient descent, but add gaussian noise to low derivatives
class RandomGradPerDim(Optimizer):
  def __init__(self, noise_stddev=1, min_deriv=1e-2, grad_opt=Adam()):
    self.noise_stddev = noise_stddev
    self.min_deriv = min_deriv
    self.grad_opt = grad_opt
    self.derivs = np.array([])
    self.output = np.array([])
  def initialize(self, func, args, param_init_func):
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, time
  def update(self, func, args):
    for dim in range(len(self.derivs)):
      if abs(self.derivs[dim]) < self.min_deriv:
        self.derivs[dim] += np.random.normal(0, self.noise_stddev)
    prog_params = self.grad_opt.update(args["prog_param_comb"], self.derivs, None, self.output)
    args.update(prog_param_comb=prog_params)
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, prog_params, time
  def __str__(self):
    grad_opt_name_end = str(self.grad_opt).find("(")
    return f"RGPerDim(n_std={self.noise_stddev}, m_d={self.min_deriv}, g_o={str(self.grad_opt)[:grad_opt_name_end]})"

# Normal gradient descent (variant chosen through the gd parameter)
class GradientDescent(Optimizer):
  def __init__(self, gd):
    self.gd = gd
    self.derivs = np.array([])
    self.output = np.array([])
  def initialize(self, func, args, param_init_func):
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, time
  def update(self, func, args):
    prog_params = self.gd.update(args["prog_param_comb"], self.derivs, None, self.output)
    args.update(prog_param_comb=prog_params)
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, prog_params, time
  def __str__(self):
    gd_str = self.gd.__str__()
    first_param_end_pos = gd_str.find(",")
    if first_param_end_pos:
      gd_str = gd_str[:first_param_end_pos]
    return gd_str

class GA:
  def __init__(self, pop_size=50):
    self.derivs = np.array([])
    self.output = np.array([])
    self.genetic_algorithm = None
    self.pop_size = pop_size
    self.time = 0

  def initialize(self, func, args, param_init_func):
    self.genetic_algorithm = GeneticAlgorithm(args["prog_param_comb"], population_size=self.pop_size, maximise_fitness=False, generations=float("inf"), param_init_func=param_init_func)
    self.fitness = None

    def fitness(individuals, data):
      _args = dict(args, prog_param_comb=individuals)
      self.outputs, self.derivs, time = func(**_args)
      self.time += time
      return self.outputs

    self.genetic_algorithm.fitness_function = fitness
    self.genetic_algorithm.initialize()
    return func(**args)
  def update(self, func, args):
    self.genetic_algorithm.step()
    best_output, self.derivs, time = func(**args)
    self.prev_time = self.time # the time taken by the func() calls in fitness(), not here
    self.time = 0

    return best_output, self.derivs, self.genetic_algorithm.best_individual()[1], self.prev_time

  def __str__(self):
    return f"GA(pop={self.pop_size})"

# Adam with warm start if there is no improvement over the previous 'period' seconds to the period before
class AdamRestart:
  def __init__(self, lr=1e-3, period=100):
    self.gd = Adam(lr=lr)
    self.lr = lr
    self.period = period
    self.period_start_time = pytime()
    self.counter = 0
    self.derivs = np.array([])
    self.output = np.array([])
    self.sum_output = 0
    self.prev_sum_output = float("inf")
  def initialize(self, func, args, param_init_func):
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, time
  def update(self, func, args):
    if pytime() - self.period_start_time > self.period:
      if self.sum_output >= self.prev_sum_output:
        self.gd.cache = {}
      self.prev_sum_output = self.sum_output
      self.sum_output = 0
      self.period_start_time = pytime()
    prog_params = self.gd.update(args["prog_param_comb"], self.derivs, None, self.output)
    args.update(prog_param_comb=prog_params)
    self.sum_output += self.output
    self.output, self.derivs, time = func(**args)
    return self.output, self.derivs, prog_params, time
  def __str__(self):
    gd_str = self.gd.__str__()
    first_param_end_pos = gd_str.find(",")
    if first_param_end_pos:
      gd_str = gd_str[:first_param_end_pos]
    return "AdamRestart"

# blindly test values based on the parameter init function
class Rand(Optimizer):
  def initialize(self, func, args, param_init_func):
    self.random_state = np.random.RandomState()
    self.param_init_func = param_init_func

    self.best_solution = args["prog_param_comb"]
    self.best_output, derivs, time = func(**args)

    return self.best_output, derivs, time
  def update(self, func, args):
    curr_solution = list(args["prog_param_comb"])
    args["prog_param_comb"] = self.param_init_func(self.random_state)
    output, derivs, time = func(**args)

    args["prog_param_comb"] = self.best_solution
    self.best_output, _, _ = func(**args)

    if output < self.best_output:
      self.best_output = output
      self.best_solution = curr_solution
    return self.best_output, derivs, self.best_solution, time
  def __str__(self):
    return f"Rand()"

# randomly mutate one parameter at a time, keeping the best seen solution
class RandMut(Optimizer):
  def initialize(self, func, args, param_init_func):
    self.random_state = np.random.RandomState()
    self.param_init_func = param_init_func

    self.best_solution = args["prog_param_comb"]
    self.best_output, derivs, time = func(**args)

    return self.best_output, derivs, time
  def update(self, func, args):
    curr_solution = list(args["prog_param_comb"])
    mut_idx = np.random.randint(0, len(args["prog_param_comb"]))
    args["prog_param_comb"][mut_idx] = self.param_init_func(self.random_state)[0]
    output, derivs, time = func(**args)
    args["prog_param_comb"] = self.best_solution
    self.best_output, _, _ = func(**args)
    if output < self.best_output:
      self.best_output = output
      self.best_solution = curr_solution
    return self.best_output, derivs, self.best_solution, time
  def __str__(self):
    return f"RandMut()"
