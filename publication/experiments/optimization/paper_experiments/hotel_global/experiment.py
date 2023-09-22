from experiment_common import *

programs = ( { 'name': 'paper_programs/hotel/hotel', 'stddevs': (0.0,), 'seed': (-1,), 'nreps': (10,), 'time_limit': (1800 * 1.1,), 'params': partial(uniform, 0, 100, 56) },)

estimators = (
  { 'name': 'crisp', 'params': { } },
)

optimizers = (
  SA(),
  GA(),
)
