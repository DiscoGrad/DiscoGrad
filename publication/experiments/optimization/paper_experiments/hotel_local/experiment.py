from experiment_common import *

programs = (
{ 'name': 'hotel/hotel', 'stddevs': (0.5,), 'seed': (-1,), 'nreps': (10,), 'time_limit': (1800 * 1.1,), 'params': partial(uniform, 0, 100, 56) },
{ 'name': 'hotel/hotel', 'stddevs': (1.0,), 'seed': (-1,), 'nreps': (10,), 'time_limit': (1800 * 1.1,), 'params': partial(uniform, 0, 100, 56) },
{ 'name': 'hotel/hotel', 'stddevs': (2.0,), 'seed': (-1,), 'nreps': (10,), 'time_limit': (1800 * 1.1,), 'params': partial(uniform, 0, 100, 56) },
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
  GradientDescent(Adam(1)),
  GradientDescent(Adam(5e-1)),
  GradientDescent(Adam(2.5e-1)),
)
