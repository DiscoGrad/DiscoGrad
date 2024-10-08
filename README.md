# DiscoGrad

> Trying to do gradient descent using automatic differentiation over branchy programs?
Or to combine them with neural networks for end-to-end training?
Then this might be interesting to you.

Automatic Differentiation (AD) is a popular method to obtain the gradients of computer programs, which are extremely useful for adjusting program parameters using gradient descent to solve optimization, control, and inference problems. Unfortunately, AD alone often yields unhelpful (zero-valued and/or biased) gradients for programs involving both parameter-dependent branching control flow such as if-else statements and randomness, including various types of simulations.

DiscoGrad automatically transforms C++ programs to a version that efficiently calculates smoothed gradients *across* branches. Smoothing via external perturbations is supported, but is not required if the target program itself involves randomness. DiscoGrad includes several gradient estimation backends as well as the possibility to integrate neural networks via Torch. The tool supports basic C++ constructs, but is still a research prototype.

The repository includes a number of sample applications from domains such as transportation, crowd management, and epidemiology.

[DiscoGrad Use Cases](https://github.com/philipp-andelfinger/DiscoGrad/assets/59713878/6419fccf-1e20-4a2c-8fef-854197824b15)

[Illustration of DiscoGrad](https://github.com/philipp-andelfinger/DiscoGrad/assets/59713878/4fc691f2-d760-441a-8155-0dc266d5853a)

## 💾 Installation

Tested on `Ubuntu 24.04.1 LTS`, `Ubuntu 22.04.4 LTS`, `Arch Linux` and `Fedora 38 Workstation`

To compile the transformation code, you need the following packages (or their analogues provided by your Linux distribution):
- `clang`, `libclang-dev`, `libclang-cpp-dev` (version 13 or higher)
- `llvm`, `llvm-dev` (version 13 or higher)
- `cmake`

```
cd transformation
cmake .
make -j
```

## 🚀 Quickstart

You can use the code contained in `programs/hello_world/hello_world.cpp` as a quickstart template and reference. The `programs` folder also contains a number of more complex programs.

To compile the hello world example, which implements the Heaviside step function as shown in the video above:
```shell
discograd$ ./smooth_compile programs/hello_world/hello_world.cpp
```
`smooth_compile` is a shell script that invokes the commands for transforming and compiling the code for the different backends. Here, it will create a binary for each backend in the `programs/hello_world` folder.

AD on the original (crisp) C++ program yields a 0 derivative:
```shell
DiscoGrad$ echo 0.0 | ./programs/hello_world/hello_world_crisp_ad --var 0.0 --ns 1
expectation: 1
derivative: 0
```

Our estimator DiscoGrad Gradient Oracle (DGO) calculates a non-zero derivative useful for optimization:
```shell
DiscoGrad$ echo 0.0 | ./programs/hello_world/hello_world_dgo --var 0.25 --ns 1000
expectation: 0.527
derivative: -0.7939109206
```

You can run `./programs/hello_world/hello_world_{crisp,dgo,pgo,reinforce} -h` for CLI usage information.

## ❔Usage

### Usage of the DiscoGrad API

The use of our API requires some boilerplate, as detailed below. Please refer to the `programs` folder for some example usages.

1. At the top of your source file, define how many inputs your program has and include the discograd header (in this order).
```c++
const int num_inputs = 1;
#include "discograd.hpp"
```
2. Implement your entry function, by prepending `_DiscoGrad_` to the name and using the differentiable type `adouble` as return value. An object of the type `aparams` holds the program inputs. As in traditional AD libraries, the type `adouble` represents a double precision floating point variable. In addition to differentiating through adoubles, DiscoGrad allows branching on (functions of) adoubles and generates gradients that reflect the dependence of the branch taken on the condition.
```c++
adouble _DiscoGrad_my_function(DiscoGrad<num_inputs>& _discograd, aparams& p) {
  adouble inputs[num_inputs];
  for (int i = 0; i < num_inputs; i++)
    inputs[i] = p[i];
  adouble output = 0.0;
  ... // calculations and conditional branching based on inputs
  return output;
}
```
3. In the main function, interface with the DiscoGrad API by creating an instance of the `DiscoGrad` class and a wrapper for your smooth function. Call `.estimate(func)` on the DiscoGrad instance to invoke the backend-specific gradient estimator.
```c++
int main(int argc, char** argv) {
  // interface with backend and provide the CLI arguments, such as the variance
  DiscoGrad<num_inputs> dg(argc, argv);
  // create a wrapper for the smooth function
  DiscoGradFunc<num_inputs> func(dg, _DiscoGrad_my_function);  
  // call the estimate function of the backend (chosen during compilation)
  dg.estimate(func);
}
```

### Including Additional Variables in Smooth Functions

To include additional variables besides the inputs (`aparams`) and to avoid having to pass the DiscoGrad instance between smooth functions, you need to wrap your function in a class that implements the `DiscoGradProgram` interface. The only requirement for this class is that it implements the `adouble run(DiscoGrad&, aparams&)` method. See `programs/ac` for a simple and `programs/epidemics` for a more elaborate example. Here is an example that replaces steps 2 and 3 above: 
```c++
class MyProgram : public DiscoGradProgram<num_inputs> {
public:
  // a parameter wrt. which we do not want to differentiate
  double non_input_parameter;
  MyProgram(DiscoGrad<num_inputs>& _discograd, double non_input_parameter) : DiscoGradProgram<num_inputs>(_discograd), non_input_parameter(non_input_parameter) {}

  adouble _DiscoGrad_f(aparams &p, double non_input_parameter) {
    // your code here
  }

  // implement the DiscoGradProgram interface, so that dg.estimate knows what to do
  adouble run(aparams &p) {
    return _DiscoGrad_f(p, non_input_parameter);
  }
};

int main(int argc, char** argv) {
  DiscoGrad<num_inputs> dg(argc, argv);
  MyProgram prog(dg, 0.42); 
  dg.estimate(prog);
}
```

### Compilation

To compile a program in the folder `programs/my_program/my_program.cpp` with every backend:

```shell
discograd$ ./smooth_compile programs/my_program/my_program.cpp
```

Custom compiler or linker flags can be set in the ``smooth_compile`` script.

You can find a list of backends below. By default, executables for all backends are generated. To restrict compilation to a subset of backends, add the flag ``-Cbackend1,backend2,...``.

To define preprocessor constants at compile time, you can use the `-D` flag, e.g., `-DNUM_REPS=10` to set the constant named `NUM_REPS` to 10.

### Executing a Smoothed Program

To run a smoothed program and compute its gradient, simply invoke the binary with the desired CLI arguments, for example
```shell
discograd$ ./programs/my_program/my_program_dgo --var 0.25 --ns 100
```
if you want to use the DGO backend. Parameters are entered via `stdin`, for example by piping the output of `echo` as shown in the quickstart guide. The output to `stdout` after `expectation` and `derivative` will provide the smoothed output and partial derivatives.


## Backends

This is an overview of all the current backends. More detailed explanations can be found in the following sections.

| ExecutableSuffix | Description                                                                                          
| -----------------|--------------------------------------------------------------------------------
| crisp            | The original program with optional input perturbations and AD
| dgo              | DiscoGrad Gradient Oracle, DiscoGrad's own gradient estimator based on automatic differentiation and Monte Carlo sampling.     
| pgo              | Polyak's Gradient-Free Oracle presented by Polyak and further analysed by Nesterov et. al. 
| reinforce        | Application of REINFORCE to programs with artificially introduced Gaussian randomness.
| rloo             | REINFORCE Leave-One-Out estimator as described by Kool et al.

Additionally, an implementation of gradient estimation via Chaudhuri and Solar-Lezama's method of Smooth Interpretation can be found in the branch 'discograd_ieee_access'.

Note: When all branches occur directly on discrete random variables drawn from distributions of known shape, [StochasticAD](https://github.com/gaurav-arya/StochasticAD.jl) may be a well-suited alternative to the above estimators.

**References:**
- Chaudhuri, Swarat, and Armando Solar-Lezama. "Smooth interpretation." ACM Sigplan Notices 45.6 (2010): 279-291.
- Boris T Polyak. "Introduction to optimization." 1987. (Chapter 3.4.2)
- Nesterov, Yurii, and Vladimir Spokoiny. "Random gradient-free minimization of convex functions." Foundations of Computational Mathematics 17 (2017): 527-566.
- Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free! DeepRLStructPred@ICLR, 2019.

## ⚖️ License

This project is licensed under the MIT License.
The DiscoGrad tool includes some parts from third parties, which are licensed as follows:
- `backend/ankerl/unordered_dense.h`, MIT license
- `backend/genann.hpp`, zlib license
- `backend/discograd_gradient_oracle/kdepp.hpp`, MIT license
- `backend/args.{h,cpp}`, MIT license
- Doxygen Awesome theme, MIT license

## 📄 Cite

```
@article{kreikemeyer2023smoothing,
     title={Smoothing Methods for Automatic Differentiation Across Conditional Branches},
     author={Kreikemeyer, Justin N. and Andelfinger, Philipp},
     journal={IEEE Access},
     year={2023},
     publisher={IEEE},
     volume={11},
     pages={143190-143211},
     doi={10.1109/ACCESS.2023.3342136}
}
```

An alternative derivation of the DGO estimator can be found in:

```
@inproceedings{andelfinger2024automatic,
  booktitle = {24th International Conference on Computational Science (ICCS 2024)},
  title = {{Automatic Gradient Estimation for Calibrating Crowd Models with Discrete Decision Making}},
  author = {Philipp Andelfinger and Justin N. Kreikemeyer},
  pages = {227--241},
  year = {2024}
}
```

## Acknowledgement

This work was supported by Deutsche Forschungsgemeinschaft (DFG), German Research Foundation, under Grant 497901036.
