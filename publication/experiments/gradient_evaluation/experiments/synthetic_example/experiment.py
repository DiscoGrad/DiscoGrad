import numpy as np

# good seeds
# range -4 to 4
# 54321
# 23458
# 11876
# 58285298
# 2984


programs = (
  {'name': 'synthetic_example/synthetic_example', 'nreps': (1,), 'params': (np.linspace(-1, 4, 1000),), 'seed': (2468,), 'stddevs': 0.33},
)
estimators = (
    {'name': 'dgsi', 'params': {'num_paths': (32,), 'num_samples': (1,), 'restrict_mode': ('Ch',), 'si_stddev_proportion': 1, 'use_dea': (0,)}},
    {'name': 'dgsi', 'params': {'num_paths': (32,), 'num_samples': (1,), 'restrict_mode': ('Ch',), 'si_stddev_proportion': 1, 'use_dea': (0.33,)}},
    {'name': 'dgsi', 'params': {'num_paths': (4,), 'num_samples': (1,), 'restrict_mode': ('Ch',), 'si_stddev_proportion': 1, 'use_dea': (0.33,)}},
    {'name': 'pgo', 'params': {'num_samples': 1000000}},
 )
