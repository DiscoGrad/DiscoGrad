from experiment_common import *

programs = (
  { 'name': 'paper_programs/traffic_grid_populations/traffic_grid_populations_10x10', 'stddevs': (0.0,), 'seed': (1,), 'nreps': (1,), 'time_limit':      1800 * 1.1, 'params': partial(uniform, 0, 2, 100),   },
  { 'name': 'paper_programs/traffic_grid_populations/traffic_grid_populations_20x20', 'stddevs': (0.0,), 'seed': (1,), 'nreps': (1,), 'time_limit':  2 * 3600 * 1.1, 'params': partial(uniform, 0, 2, 400),   },
)
 

estimators = (
    { 'name': 'crisp', 'params': { } },
)

optimizers = (
  SA(),
  GA(),
)
