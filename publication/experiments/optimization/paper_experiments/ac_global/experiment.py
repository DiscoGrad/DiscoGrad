from experiment_common import *

programs = ( { 'name': 'ac/ac', 'stddevs': (0.0,), 'seed': (-1,), 'nreps': (10,), 'time_limit': (1800 * 1.1,), 'params': partial(normal, 0.0, 1, 82) }, )

estimators = (
  { 'name': 'crisp', 'params': { } },
)

optimizers = (
  GA(),
  SA(),
)
