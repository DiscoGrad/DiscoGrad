import os
import subprocess
from functools import partial
from reproduce_functions import *

# list of benchmark programs
programs = ["hotel/hotel.cpp"] + [f"traffic_grid_populations/traffic_grid_populations_{n}x{n}.cpp" for n in [2, 5, 10, 20]] + ["ac/ac.cpp", "epidemics/epidemics.cpp", "synthetic_example/synthetic_example.cpp"]

print("This tool will guide you through the reproduction of the empirical results (figures) of our publication.")
print("First, it is advisable to rebuild the discograd tool and smooth and compile all programs but this step may be skipped if this was done recently.")
choice = input("Do you want to proceed? (Y/n): ")
if choice == "Y" or choice == "y" or choice == "":
  print("Ok, this may take a while...")
  # rebuild transformation 
  try:
    subprocess.run(["cmake . && cmake --build . -j 4"], cwd="../transformation", shell=True)
  except subprocess.CalledProcessError as ex:
    print("Could not build DiscoGrad tool. Please make sure you installed all prerequisites and refer to the README.")
    print("Error was:", ex)
    exit(-1)
  # generate traffic program variations 
  subprocess.run(["./generate_scaled_variants.sh"], cwd="../programs/traffic_grid_populations/", shell=True)
  # build all programs 
  for idx, program in enumerate(programs):
    print("I am now smoothing and compiling program", idx, "/", len(programs))
    cmd = ["./smooth_compile " + os.path.join("programs", program)]
    print(cmd)
    try:
      subprocess.run(cmd, cwd="..", shell=True)
    except subprocess.CalledProcessError as ex:
      print("Could not compile program", program, ". Please make sure you installed all prerequisites and refer to the README.")
      print("Error was:", ex)
      exit(-1)
    print("done.")

options = {**{
  #"Everything": everything,
  "Introductory figure": introduction_figure,
  "Figure regarding the assumptions of SI": synthetic_example,
  "Optimization Performance (All plots)": partial(optimization, [prog for prog in programs if not any(x in prog for x in ["2x2", "5x5", "synthetic"])]),
}, **{
  f"Optimization Performance of {prog}": partial(optimization, [prog]) for prog in programs if not any(x in prog for x in ["2x2", "5x5", "synthetic"])
}, **{
  "Gradient Fidelity (All plots and Matrix)": partial(fidelity, [prog for prog in programs if not any(x in prog for x in ["10x10", "20x20", "synthetic"])]),
  #"Gradient Fidelity for Traffic 5x5": todo,
  #"Gradient Fidelity for Epidemics": todo,
  #"Gradient Fidelity for AC": todo,
  "Runtime": todo,
  "Do nothing and exit": exit,
}} 
while True:
  print("Select, which figure you would like to reproduce:")
  for idx, option in enumerate(options):
    print(" ", f"{idx})", option)
  choice = input(f"Input a number (0-{len(options)-1}): ")
  if choice not in map(str, range(0, len(options))):
    print("Invalid choice", choice)
    continue
  options[list(options.keys())[int(choice)]]()

