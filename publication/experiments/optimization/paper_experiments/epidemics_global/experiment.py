from experiment_common import *

programs = ( { 'name': 'paper_programs/epidemics/epidemics', 'stddevs': (0.0,), 'seed': (-1,), 'nreps': (100,), 'time_limit': (3600 * 1.1,), 'params': partial(uniform, 0, 1, 102) }, )

estimators = (
  { 'name': 'crisp', 'params': { } },
)

optimizers = (
  SA(),
  GA(),
)
