from functools import partial
from optimizers import *

def normal(mean, stddev, num, random_state):
  return random_state.normal(mean, stddev, num)
def uniform(low, high, num, random_state):
  return random_state.uniform(low, high, num)
