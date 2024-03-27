from experiment_common import *

programs = (
  { 'name': 'traffic_grid_populations/traffic_grid_populations_10x10', 'stddevs': (0.1,), 'seed': (1,), 'nreps': (1,), 'time_limit':      1800 * 1.1, 'params': partial(uniform, 0, 2, 100),   },
  { 'name': 'traffic_grid_populations/traffic_grid_populations_10x10', 'stddevs': (0.2,), 'seed': (1,), 'nreps': (1,), 'time_limit':      1800 * 1.1, 'params': partial(uniform, 0, 2, 100),   },
  { 'name': 'traffic_grid_populations/traffic_grid_populations_10x10', 'stddevs': (0.4,), 'seed': (1,), 'nreps': (1,), 'time_limit':      1800 * 1.1, 'params': partial(uniform, 0, 2, 100),   },
)
 
estimators = (
  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'reinforce', 'params': { 'num_samples': (100) } },
  { 'name': 'reinforce', 'params': { 'num_samples': (1000) } },

  { 'name': 'pgo', 'params': { 'num_samples': (100) } },
  { 'name': 'pgo', 'params': { 'num_samples': (1000) } },

  { 'name': 'dgo', 'params': { 'num_samples': (100) } },
  { 'name': 'dgo', 'params': { 'num_samples': (1000) } },
)

optimizers = (
  GradientDescent(Adam(2e-1)),
  GradientDescent(Adam(1e-1)),
  GradientDescent(Adam(5e-2)),
)
