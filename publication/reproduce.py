import os
import subprocess
from reproduce_functions import *

programs = ["hotel/hotel.cpp"] + [f"traffic_grid_populations/traffic_grid_populations_{n}x{n}.cpp" for n in [2, 5, 10, 20]] + ["ac/ac.cpp", "epidemics/epidemics.cpp", "synthetic_example/synthetic_example.cpp"]

print("This tool will guide you through the reproduction of the empirical results (figures) of our publication.")
print("First, it is advisable to rebuild the discograd tool and smooth and compile all programs, but this step may be skipped if this was done recently.")
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
  "Everything": everything,
  "Introductory figure": introduction_figure,
  "Figure regarding the assumptions of SI": synthetic_example,
  "Optimization Performance (All plots)": todo,
}, **{
  f"Optimization Performance of {prog}": todo for prog in programs if "2x2" not in prog and "5x5" not in prog and "synthetic" not in prog
}, **{
  "Gradient Accuracy (All plots and Matrix)": todo,
  "Gradient Accuracy for Traffic 5x5": todo,
  "Gradient Accuracy for Epidemics": todo,
  "Gradient Accuracy for AC": todo,
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

