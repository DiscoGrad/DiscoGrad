from abc import ABC
from typing import Callable, Tuple, Dict
import numpy as np
#from pyeasyga import GeneticAlgorithm

# Abstract base class for all optimizers
class Optimizer(ABC):
  def __init__(self):
    pass
  # initialize the state of the optimizer based on the objective and initial parameters
  def initialize(self, func, args) -> Tuple[np.ndarray, np.ndarray]:
    pass
  # update the current parameter conf. using func and return output, gradient and new parameter conf.
  def update(self, func: Callable, args: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    pass
  def __str__(self):
    return f"AbstractOptimizer()"
