#!/bin/bash

# traffic
for n in 2 5; do
  cat traffic${n}x$n/ref_args.txt | python3 generate.py traffic_grid_populations/traffic_grid_populations_${n}x$n 1 1 $(($n*$n)) 0.2 200 1 1 > traffic${n}x$n/experiment.py  
done

#epidemics
cat epidemics/ref_args.txt | python3 generate.py epidemics/epidemics 1234 1 102 0.02 100 0.5 0.5 > epidemics/experiment.py

# thermostat
cat ac/ref_args.txt | python3 generate.py ac/ac 1234 1 82 0.1 200 1 1 > ac/experiment.py

#hotel
cat hotel/ref_args.txt | python3 generate.py hotel/hotel 1234 1 56 1.0 200 50 50 > hotel/experiment.py
