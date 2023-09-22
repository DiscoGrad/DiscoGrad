#!/bin/bash

# Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#  
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#   
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
#   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

ref_seed=$1 # 9 looks good: too pose an actual optimization problem, we need a reference parameter combination
            # that doesn't permit a good trivial solution of all-zeros and where random parameter combinations
            # are unlikely to behave the same as the reference
num_reps=1000
num_locs=100
num_params=$(($num_locs+2))
echo $num_params

rm -f reference.csv
./generate_network.py $ref_seed $num_locs
#../../generate_random_params.py $(($ref_seed+1)) 0 1 $num_params > args.txt
(echo -e "0.9\n0.02" ; ../../generate_random_params.py $ref_seed 0 1 $(($num_params-2))) > args.txt

cat args.txt | ./epidemics_crisp -1 1 $num_reps 0 1 #> /dev/null

echo "perfect fit:"
cat args.txt | ./epidemics_crisp -1 1 $num_reps 0 1 | grep exp
echo

mv args.txt ref_args.txt

../../generate_random_params.py $ref_seed 0 0 $num_params > args.txt

echo "all zeros:"
cat args.txt | ./epidemics_crisp -1 1 $num_reps 0 1 | grep exp
echo

../../generate_random_params.py $ref_seed 1 1 $num_params > args.txt

echo "optimal head parameters:"
(head -n 2 ref_args.txt; cat args.txt )| ./epidemics_crisp -1 1 $num_reps 0 1 | grep exp
echo

echo "random parameters:"
for i in $(seq 1 30); do
  ./generate_args.py $RANDOM $num_params
  cat args.txt | ./epidemics_crisp -1 1 $num_reps 0 1 | grep exp
done
echo
