from experiment_common import *

programs = (
  { 'name': 'epidemics/epidemics', 'stddevs': (0.01,), 'seed': (-1,), 'nreps': (100,), 'time_limit': (3600 * 1.1,), 'params': partial(uniform, 0, 1, 102) },
  { 'name': 'epidemics/epidemics', 'stddevs': (0.02,), 'seed': (-1,), 'nreps': (100,), 'time_limit': (3600 * 1.1,), 'params': partial(uniform, 0, 1, 102) },
  { 'name': 'epidemics/epidemics', 'stddevs': (0.04,), 'seed': (-1,), 'nreps': (100,), 'time_limit': (3600 * 1.1,), 'params': partial(uniform, 0, 1, 102) },
)

estimators = (
  { 'name': 'reinforce', 'params': { 'num_samples': (100) } },
  { 'name': 'reinforce', 'params': { 'num_samples': (1000) } },

  { 'name': 'pgo', 'params': { 'num_samples': (100) } },
  { 'name': 'pgo', 'params': { 'num_samples': (1000) } },

  { 'name': 'dgo', 'params': { 'num_samples': (100) } },
  { 'name': 'dgo', 'params': { 'num_samples': (1000) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,) } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,) } },
)

optimizers = (
  GradientDescent(Adam(5e-3)),
  GradientDescent(Adam(1e-2)),
  GradientDescent(Adam(2e-2)),
)

