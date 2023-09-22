#!/usr/bin/python3

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

import numpy as np
import sys

if len(sys.argv) != 3:
  print("usage: generate_args.py <seed> <num_args>")
  exit(1)

np.random.seed(int(sys.argv[1]))

with open("args.txt", "w") as file:
  rands = np.random.uniform(0, 1, int(sys.argv[2]))
  for r in rands:
    file.write(str(r))
    file.write("\n")
