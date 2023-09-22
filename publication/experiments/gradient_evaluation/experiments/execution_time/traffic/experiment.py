from experiment_common import *

programs = (
  { 'name': 'traffic_grid_populations/traffic_grid_populations_5x5', 'stddevs': (0.2,), 'seed': (1,), 'nreps': (1,), 'time_limit':      1800 * 1.1, 'params': (partial(uniform, 0, 2, 100),)   },)

estimators = (
  { 'name': 'crisp', 'params': { 'enable_ad': False, 'num_samples': (1000) } },
  { 'name': 'crisp', 'params': { 'enable_ad': True, 'num_samples': (1) } },
  { 'name': 'crisp', 'params': { 'enable_ad': True, 'num_samples': (10) } },
  { 'name': 'crisp', 'params': { 'enable_ad': True, 'num_samples': (100) } },
  { 'name': 'crisp', 'params': { 'enable_ad': True, 'num_samples': (1000) } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('Di',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('Ch',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('IW',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },

  { 'name': 'dgsi', 'params': { 'num_paths':  (4,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths':  (8,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (16,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },
  { 'name': 'dgsi', 'params': { 'num_paths': (32,), 'restrict_mode': ('WO',), 'use_dea': (0,), 'num_samples': (1,), 'return_num_paths': True } },

  { 'name': 'dgo', 'params': { 'num_samples': (1) } },
  { 'name': 'dgo', 'params': { 'num_samples': (10) } },
  { 'name': 'dgo', 'params': { 'num_samples': (100) } },
  { 'name': 'dgo', 'params': { 'num_samples': (1000) } },
)
