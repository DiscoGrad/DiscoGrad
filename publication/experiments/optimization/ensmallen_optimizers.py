#!/usr/bin/python3

# based on the SA implementation from ensmallen

from abstract_optimizer import Optimizer
import numpy as np

class SA(Optimizer):
  def __init__(self):
    self.initT = 10000.
    self.initMoves = 0
    #self.initMoves = 1000
    self.moveCtrlSweep = 100
    self.maxMoveCoef = 20
    self.initMoveCoef = 0.3
    self.gain = 0.3
    
    self.lmbda = 0.00001 # for the exponential cooling schedule

    self.temperature = self.initT

    self.idx = 0

    self.sweepCounter = 0


  def initialize(self, func, args, param_init_func):
    num_params = len(args['prog_param_comb'])
    self.accept = np.zeros(num_params)
    self.moveSize = np.zeros(num_params) + self.initMoveCoef

    self.energy, _, time = func(**args)
    self.last_func_output = self.energy

    for i in range(self.initMoves):
      self.generateMove(func, args)

    return self.energy, np.zeros(num_params), time

  def next_temperature(self, currentTemperature):
    return (1 - self.lmbda) * currentTemperature

  def update(self, func, args):
    self.generateMove(func, args)
    self.temperature = self.next_temperature(self.temperature)
    return self.last_func_output, np.zeros(len(args['prog_param_comb'])), args['prog_param_comb'], self.time

  def generateMove(self, func, args):
    params = args['prog_param_comb']
    prevEnergy = self.energy
    prevValue = params[self.idx]

    # It is possible to use a non-Laplace distribution here, but it is difficult
    # because the acceptance ratio should be as close to 0.44 as possible, and
    # MoveControl() is derived for the Laplace distribution.

    # Sample from a Laplace distribution with scale parameter moveSize(idx).
    unif = 2.0 * np.random.uniform() - 1.0
    #move = (unif < 0) ? (moveSize(idx) * np.log(1 + unif)) : (-moveSize(idx) * np.log(1 - unif))
    move = self.moveSize[self.idx] * np.log(1 + unif) if unif < 0 else -self.moveSize[self.idx] * np.log(1 - unif)

    params[self.idx] += move
    args.update(prog_param_comb=params)
    self.energy, _, self.time = func(**args)

    # may be overwritten below but reporting that value would be misleading due to variance, since self.energy may never be updated beyond a certain point
    self.last_func_output = self.energy

    # According to the Metropolis criterion, accept the move with probability
    # min{1, exp(-(E_new - E_old) / T)}.
    xi = np.random.uniform()
    delta = self.energy - prevEnergy
    criterion = np.exp(-delta / self.temperature)
    if delta <= 0. or criterion > xi:
      self.accept[self.idx] += 1
    else: # // Reject the move; restore previous state.
      params[self.idx] = prevValue
      self.energy = prevEnergy

    self.idx += 1
    if self.idx == len(params): # Finished with a sweep.
      self.idx = 0
      self.sweepCounter += 1

    if (self.sweepCounter == self.moveCtrlSweep): # Do MoveControl().
      self.MoveControl()
      self.sweepCounter = 0

  def MoveControl(self):
    target = np.zeros(len(self.accept)) + 0.44
    self.moveSize = np.log(self.moveSize)
    self.moveSize += self.gain * (self.accept / float(self.moveCtrlSweep) - target)
    self.moveSize = np.exp(self.moveSize)

    self.moveSize = [min(m, self.maxMoveCoef) for m in self.moveSize]

    self.accept = np.zeros(len(self.accept))

  def __str__(self):
    return f"SA()"
