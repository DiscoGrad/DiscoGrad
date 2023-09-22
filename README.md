# DiscoGrad
*Automatically differentiate across conditional control flow of C++ programs.* 

![](docs/banner.mp4){height=310px}

Automatic Differentiation (AD) is a popular method to obtain the gradient of computer programs. This is extremely useful to identify optimal program parameters via gradient descent optimization procedures. Unfortunately, a naive use of AD does typically not provide useful gradients for programs involving parameter-dependent branching control flow, such as if-then-else or for-loop constructs. DiscoGrad provides an automatic transformation of C++ programs to a version that calculates a smoothed gradient *across* those constructs. It supports several backends for smoothing listed below. The tool is functional and supports many C++ constructs, but is still a *research prototype*.

## 💾 Installation

Supported operating systems: `Linux`
Tested on: `Fedora 38 Workstation`, `Ubuntu 22.04.1 LTS` and `Arch Linux`

To compile the transformation code, you need the following packages (or their counterparts provided by your Linux distribution):
- `clang`, `clang-devel`
- `llvm`, `llvm-devel`
- `cmake`

```
cd transformation
cmake .
cmake --build .
```

## 🚀 Quickstart

You can use the code contained in `programs/hello_world/hello_world.cpp` as a quickstart template and reference. The `programs` folder also contains a number of more complex programs.

To compile the hello world example, which implements the heaviside function as shown in the video above:
```shell
discograd$ ./smooth_compile programs/hello_world/hello_world.cpp
```
`smooth_compile` is a shell script that invokes the commands for transforming and compiling the code for the different backends. Here, it will create a binary for each backend in the `programs/hello_world` folder.

AD on the original (crisp) C++ program yield a 0 derivative:
```shell
discograd$ echo 0.0 | ./programs/hello_world/hello_world_crisp_ad
expectation: 1
derivative: 0
```

The smooth semantics of Smooth Interpretation provide a non-zero derivative useful for optimization:
```shell
discograd$ echo 0.0 | ./programs/hello_world/hello_world_dgsi --var 0.25
expectation: 0.5
derivative: -0.7978845608
```

You can run `./programs/hello_world/hello_world_{dgsi,dgo,pgo,reinforce,crisp} -h` for CLI usage information.

## Usage

### Use of the DiscoGrad API

The use of our API requires some boilerplate, as detailed below. Please refer to the `programs` folder for some example usages.

1. At the top of your source file, define how many inputs your program has and include the discograd header (in this order).
```c++
const int num_inputs = 1;
#include "discograd.hpp"
```
2. Implement your entry function, by prepending `_DiscoGrad_` to the name and using the differentiable type `adouble` as return value. Use `dg.get_variance()` to initialize smoothed doubles and `aparams` to access the program inputs. At the end of the function, return the `.expectation()` of the `sdouble` output.
```c++
adouble _DiscoGrad_my_function(DiscoGrad& _dg, aparams x) {
  // smooth inputs with the user-provided variance
  sdouble inputs[num_inputs];
  for (int i = 0; i < num_inputs; i++) {
    inputs[i] = sdouble({x[i], _dg.get_variance()});
  }
  sdouble output = 0.0;
  ...
  return output.expectation();
}
```
3. In the main function, interface with the DiscoGrad API by creating an instance of the `DiscoGrad` class and a wrapper for your smooth function. Call `.estimate(func)` on the DiscoGrad instance to invoke the backend-specific gradient estimator.
```c++
int main(int argc, char** argv) {
  // interface with backend and provide the CLI arguments, such as the variance
  DiscoGrad<num_inputs> dg(argc, argv);
  // create a wrapper for the smooth function
  DiscoGradFunc<num_inputs> func(_DiscoGrad_my_function);  
  // call the estimate function of the backend (chosen during compilation)
  dg.estimate(func);
}
```

### Including Additional Variables in Smooth Functions

To include additional variables besides the inputs (`aparams`), you need to wrap your function in a class that implements the `DiscoGradProgram` interface. The only requirement for this class is that it implements the `adouble run(DiscoGrad&, aparams&)` method. See `programs/ac` for a simple and `programs/epidemics` for a more elaborate example. Here is an example that replaces steps 2 and 3 above: 
```c++
class MyProgram : public DiscoGradProgram<num_inputs> {
public:
  // a parameter wrt. which we do not want to differentiate
  double non_input_parameter;
  MyProgram(double _non_input_parameter) {
    non_input_parameter = _non_input_parameter;
  }
  // implement the DiscoGradProgram interface, so that dg.estimate knows what to do
  adouble run(DiscoGrad<num_inputs> &_discograd, aparams &p) {
    return _DiscoGrad_f(_discograd, p, non_input_parameter);
  }
};

int main(int argc, char** argv) {
  DiscoGrad<num_inputs> dg(argc, argv);
  MyProgram prog(0.42); 
  dg.estimate(prog);
}
```

### Smoothing and Executing Programs

To compile a program in the folder `programs/my_program/my_program.cpp` with every backend:

```shell
discograd$ ./smooth_compile programs/my_program/my_program.cpp
```

This will create a binary for every backend. You can find a list of backends in the "How it works" section.

### Run a Smoothed Program

To run a smoothed program, simply invoke the binary with the desired CLI arguments, for example
```shell
discograd$ ./programs/my_program/my_program_dgo --var 0.25 --ns 100
```
if you want to use the DGO backend. Parameters are entered via `stdin`, for example by piping the output of `echo` as shown in the quickstart guide. The output to `stdout` after `expectation` and `derivative` will provide the smoothed output and partial derivatives.


## ❓ How it works

### Backends

This is an overview of all the current backends. More detailed explanations can be found in the following sections.

| Abbreviation | Name                             | Description                                                                                            | Current Limitations                                                 |
| -----:       | :------------------------------- | :----------------------------------------------------------------------------------------------------- | :---------                                                          |
| DGSI         | DiscoGrad Smooth Interpretation | Efficient implementation of Smooth Interpretation, a technique proposed by Chaurhuri et. al.           | See "Limitations of DGSI"                                           |
| DGO          | DiscoGrad Gradient Oracle        | DiscoGrads own gradient estimator based on automatic differentiation and Monte Carlo sampling.         | No nested branches |
| PGO          | Polyak's Gradient-Free Oracle    | "Gradient-free" oracle presented by Polyak and further analysed by Nesterov et. al.   | None, possibly requires many samples                                |
| REINFORCE    | n/a                              | Application of REINFORCE to programs with artificially introduced Gaussian randomness.                 | None, generally requires more samples than PGO                      |

**References:**
- Chaudhuri, Swarat, and Armando Solar-Lezama. "Smooth interpretation." ACM Sigplan Notices 45.6 (2010): 279-291.
- Boris T Polyak. "Introduction to optimization." 1987. (Chapter 3.4.2)
- Nesterov, Yurii, and Vladimir Spokoiny. "Random gradient-free minimization of convex functions." Foundations of Computational Mathematics 17 (2017): 527-566.

### DGSI Backend

Coming soon.

#### Development Status of the DGSI backend

Of all backends, DGSI is the most complex, as it has to be deeply integrated with the program execution. Thus, there are currently some limitations on which language constructs may be used. Regular C++ code *not* operating on smoothed doubles (`sdouble`) can be used as-is. The following C++ constructs **are currently supported** by the smoothing (not comprehensive):
- `if-then-else` control flow
- `for` and `while` loops on "crisp" (`int`, `double`, ...)
- support for smooth `for` loops (on `sdouble` counters) can be enabled by an experimental switch when compiling the transformation
- `foreach` loops using `iterator`s
- `while` loops
- functions returning an `sdouble`
- pointer and references to `sdoubles`
- arrays of `sdouble`
- vector of `sdouble` when using preallocation
- `lambdas`, as long as they get the `DiscoGrad` instance as an argument
- default, copy, direct, direct list initialization and copy assignment of `sdouble`s (e.g. `sdouble a = 42`, `sdouble b(42)`, `sdouble c{42}`, `sdouble d{}` and `sdouble e; e = 42;`)

The following C++ constructs are **not yet supported** (not comprehensive):
- ternary operator (use if-else instead)
- `switch-case`
- bitwise operations on `sdouble`
- Exceptions and `try-catch`
- templates
- using `sdoubles` from the global scope
- using `sdouble` in preprocessor macros 
- using `malloc` for memory allocation of `sdoubles` with variance
- class member functions defined outside the class using the scope resolution operator (e.g. `MyClass::method(sdouble)`)
- functions returning `auto` only work for `adouble`, not for `sdouble`
- copy list initialization of `sdoubles` (e.g. `sdouble a = {10.0}`)
- for loop over pointer to `sdouble`
- `push_back` in vectors of `sdouble`

### DGO Backend

Coming soon.

### PGO Backend

Coming soon.

### REINFORCE Backend

Coming soon.

## License

This project is licensed under the MIT License.
The DiscoGrad tool includes some parts from third parties, which are licensed as follows:
- `backend/genann.hpp`, zlib license
- `backend/discograd_gradient_oracle/kdepp.hpp`, MIT license
- `backend/args.{h,cpp}`, MIT license
- Doxygen Awesome theme, MIT license

Additionally, for reproduction of the results from the paper, we include the following code:
- `pyeasyga.py`, BSD-3-Clause license

## 📄 Cite

```
Coming soon.
```
