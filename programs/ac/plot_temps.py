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

import fileinput
import matplotlib.pyplot as plt
import sys

flag = ""
if len(sys.argv) > 1:
  flag = sys.argv[1]

temps = []
heat = []
cool = []
initial = 0.0
decay = 0.0
target = 0.0
steps = 0
hsteps = 0
csteps = 0
for line in fileinput.input(files=("-",)):
  print(line, end="")
  if line.startswith("temp: "):
    pass
  elif line.startswith("initial temp"):
    initial = float(
      line[len("initial temp"):]
    )
  elif line.startswith("temp decay"):
    decay = float(
      line[len("temp decay"):]
    )
  elif line.startswith("target temp"):
    target = float(
      line[len("target temp"):]
    )
  elif line.startswith("temp"):
    steps += 1
    temps.append(float(
      line[len("temp"):]
    ))
  if flag == "--nn":
    if line.startswith("heating on:"):
      heat.append(float(
        line[len("heating on:"):]
      ))
    elif line.startswith("heating off"):
      heat.append(0.0)
    elif line.startswith("cooling on:"):
      cool.append(float(
        line[len("cooling on:"):]
      ))
    elif line.startswith("cooling off"):
      cool.append(0.0)
  #else:
  #  print(line, end="")

xs = range(0, len(temps) - 1)

plt.plot(xs, temps[1:], label="temperature", color="black")
plt.plot(xs, [target]*len(xs), label="target", linestyle="--", color="grey")
plt.plot(xs, [decay ** x * initial for x in xs], label="without thermostat", color="green")
if flag == "--nn":
  plt.plot(xs, heat, label="heating", color="red")
  plt.plot(xs, cool, label="cooling", color="blue")
plt.legend()
plt.show()

