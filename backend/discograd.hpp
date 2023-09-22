/** The main header of DiscoGrad. Include this in your program to use DiscoGrad. 
 * 
 *  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 *  and associated documentation files (the “Software”), to deal in the Software without
 *  restriction, including without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 *  Software is furnished to do so, subject to the following conditions:
 *   
 *    The above copyright notice and this permission notice shall be included in all copies or
 *    substantial portions of the Software.
 *    
 *    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include <array>
#include <iostream>
#include <limits>
#include <math.h>
#include <float.h>
#include <chrono>
#include <ratio>
#include <random>
#include <stdlib.h>
#include "args.h"

using namespace std;

/** Number of inputs to the program. This needs to be specified in the user program. */
extern const int num_inputs;

/* Compile-time options */

// AD varieties
//#define FW_AD         // forward-mode AD
//#define NO_AD         // disable AD

// smoothing varieties
//#define SI            // smooth interpretation
//#define DGO           // DiscoGrad Gradient Oracle
//#define PGO           // crisp execution and sampling for Polyak Gradient Oracle
//#define REINFORCE     // crisp execution and sampling for reinforce
//#define CRISP         // completely crisp

// choose AD backend
#if defined FW_AD
  #define ENABLE_AD
#else
  #undef ENABLE_AD
#endif

#include "ad/fw_ad.hpp"
typedef fw_adouble<num_inputs> adouble;

typedef array<adouble, num_inputs> aparams;

template<int num_inputs>
class DiscoGrad;

/** Wrapper for a program that supports estimating the (smoothed) output and gradient. */
template<int num_inputs>
class DiscoGradProgram {
public:
  virtual adouble run(DiscoGrad<num_inputs> &_discograd, aparams &p) = 0;
};

/** Wrapper for a smoothly interpreted function that takes only the parameters read from stdin. */
template<int num_inputs>
class DiscoGradFunc : public DiscoGradProgram<num_inputs> {
private:
  adouble (*func)(DiscoGrad<num_inputs>&, aparams&);
public:
  DiscoGradFunc(adouble (*func)(DiscoGrad<num_inputs>&, aparams&)) {
    this->func = func;
  }
  adouble run(DiscoGrad<num_inputs> &_discograd, aparams &p) {
    return func(_discograd, p);
  }
};

/** Base class for different kinds of DiscoGrad estimators. */
template<int num_inputs>
class DiscoGradBase {
protected: 
  bool debug;
  int num_param_combs = 1;
  int num_replications = 1; /**< For stochastic programs; A value > 1 entails averaging of results. */
  int seed_arg = 1;
  default_random_engine rep_seed_gen;  /**< For generating seeds for each replication */
  uniform_int_distribution<unsigned> seed_dist; /**< Seed generator. */
  unsigned current_seed; /**< The seed for the current run of the program. */
  adouble exp_val = 0.0;     /**< The current expected value of the smoothed program. */
  array<adouble, num_inputs> parameters;
  uint64_t start_time_us; /**< Start time of estimation. */
  uint64_t estimate_duration_us; /**< Duration of estimation. */
  /** Print the program expectation and derivatives to stdout. */
  void print_results() const {
    printf("estimation_duration: %ldus\n", estimate_duration_us);
    printf("expectation: %.10g\n", expectation());
#if not defined CRISP or defined ENABLE_AD
    for (int dim = 0; dim < num_inputs; ++dim)
      printf("derivative: %.10g\n", derivative(dim));
#endif
  }
  uint64_t get_time_us() { return chrono::time_point_cast<chrono::microseconds>(chrono::system_clock::now()).time_since_epoch().count(); }
  /** Log the time when starting to estimate. */
  void start_timer() { start_time_us = get_time_us(); }
  /** Calculate the duration of the estimation execution. */
  void stop_timer() { estimate_duration_us = get_time_us() - start_time_us; }
public:
  default_random_engine rng; /**< Random number generator to be used by the program. */
  /** Initialize a new DiscoGrad program.
   * Reads the input parameters from stdin, according to the num_inputs.
   * Reads seed and number of repliacitons from argv 1-2. If seed == -1, a random seed is chosen.
   * @param argc Number of input arguments.
   * @param argv Standard input arguments.
   * @param debug Whether to print debugging information.
   */
  DiscoGradBase(int argc, char **argv, bool debug=false) { }
  /** The variance used for smoothing. */
  double get_variance() const { return 0.0; }

  void estimate(DiscoGradProgram<num_inputs> &program) {
    for (int param_comb = 0; param_comb < num_param_combs; param_comb++) {
      int seed = this->seed_arg;
      if (seed == -1)
        seed = random_device()();

      this->rep_seed_gen.seed(seed);
      this->seed_dist = uniform_int_distribution<unsigned>(0, numeric_limits<unsigned>::max());

      for (int dim = 0; dim < num_inputs; dim++) {
        double p;
        if (scanf("%lf", &p) != 1) {
          printf("program expects %d parameters, exiting\n", num_inputs);
          exit(1);
        }
        parameters[dim] = p;
        parameters[dim].set_initial_adj(dim, 1);
      }

      start_timer();
      estimate_(program);
      stop_timer();
      print_results();
    }
  }

  /** Execute the DiscoGradProgram with a smoothed execution and estimate or calculate the gradient. */
  virtual void estimate_(DiscoGradProgram<num_inputs> &program) = 0;
  /** The expected program output of the most recent estimation. */
  double expectation() const { return exp_val.get_val(); }
  /** The (expected) program derivative for input dimension dim of the most recent estimation. */
  virtual double derivative(int dim) const { return exp_val.get_adj(dim); }
};

// define dummy sdouble for non-SI variations
#if defined DGO || defined CRISP || defined PGO || defined REINFORCE
  class sdouble : public adouble {
  public:
    sdouble(adouble m, adouble v) : adouble(m){};
    sdouble(adouble m) : adouble(m){};
    sdouble(double m, double v) : adouble(m){};
    sdouble(double m) : adouble(m){};
    explicit sdouble(int m, int v) : adouble((double)m){};
    explicit sdouble(int m) : adouble((double)m){};
    sdouble(){};
    void print() {
      printf("%.2f (", val);
      for (int i = 0; i < num_inputs - 1; i++) {
        printf("%.2f ", get_adj(i));
      }
      printf("%.2f)\n", get_adj(num_inputs - 1));
    }
    void enforce_range(double lower, double upper, double max_variance = 0.0) { *this = max(lower, min(upper, val)); };
    void enforce_range_hard(double lower, double upper, double max_variance = DBL_MAX) {
      enforce_range(lower, upper, max_variance);
    }
    adouble expectation() const { return *this; };
  };
#endif

// choose smoothing variety
#if defined SI
  #define ENABLE_SI
  #include "si/discograd.hpp"
#elif defined PGO
  #include "polyak_gradient_oracle/discograd.hpp"
#elif defined REINFORCE
  #include "reinforce/discograd.hpp"
#elif defined DGO
  #define ENABLE_SI
  #include "discograd_gradient_oracle/discograd.hpp"
#elif defined CRISP
  #undef ENABLE_SI
  #include "discograd_gradient_oracle/discograd.hpp"
#endif

#include "genann.hpp" // for (smoothed) neural networks
