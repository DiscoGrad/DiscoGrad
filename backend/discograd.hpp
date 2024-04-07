/** The main header of DiscoGrad. Include this in your program to use DiscoGrad. 
 * 
 *  Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
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

// choose AD backend
#if defined FW_AD
  #define ENABLE_AD
#else
  #undef ENABLE_AD
#endif

#include "ad/fw_ad.hpp"

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
  int num_replications = 1;
  int num_samples = 1;
  int seed_arg = 1;
  int seed;

  default_random_engine rep_seed_gen;  /**< For generating seeds for each replication */
  default_random_engine sampling_rng; /**< To draw perturbations */
  uniform_int_distribution<unsigned> seed_dist; /**< Seed generator. */
  normal_distribution<double> normal_dist;
  double variance, stddev;
  int perturbation_dim = 1;
  bool rs_mode = false;
  unsigned current_seed; /**< The seed for the current run of the program. */
  adouble exp_val = 0.0;     /**< The current expected value of the smoothed program. */
  double lowest_sample_val = DBL_MAX;
  array<adouble, num_inputs> parameters;
  uint64_t start_time_us; /**< Start time of estimation. */
  uint64_t estimate_duration_us; /**< Duration of estimation. */
  /** Print the program expectation and derivatives to stdout. */
  void print_results() const {
    printf("estimation_duration: %ldus, %.2fs\n", estimate_duration_us, estimate_duration_us * 1e-6);
    printf("expectation: %.10g\n", expectation());
    //printf("lowest: %.10g\n", lowest_val());
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
   * Reads seed and number of replicaitons from argv 1-2. If seed == -1, a random seed is chosen.
   * @param argc Number of input arguments.
   * @param argv Standard input arguments.
   * @param debug Whether to print debugging information.
   */
  DiscoGradBase(int argc, char **argv, bool debug=false) {
    sampling_rng.seed(random_device()());

    string path(argv[0]);
    args::ArgParser parser("Usage: " + path + " -s [seed = 1] --nc [#parameter combinations = 1] --nr [#replications = 1] --var [variance = 1] --pd [perturbation dimension = -1] --ns [#samples = 1]");

    parser.option("s");
    parser.option("nc");
    parser.option("nr");
    parser.option("var");
    parser.option("pd");
    parser.option("ns");

    parser.parse(argc, argv);

    if (parser.found("s"))
      this->seed_arg = stoi(parser.value("s"));

    if (parser.found("nc"))
      this->num_param_combs = stoi(parser.value("nc"));

    if (parser.found("nr"))
      this->num_replications = stoi(parser.value("nr"));

    variance = stddev = 1;
    if (parser.found("var")) {
      variance = stod(parser.value("var"));
      stddev = sqrt(variance);
    }

    perturbation_dim = -1;
    if (parser.found("pd"))
      perturbation_dim = stod(parser.value("pd"));

    normal_dist = normal_distribution<>(0, stddev);

    num_samples = 1;
    if (parser.found("ns"))
      num_samples = stoi(parser.value("ns"));

    this->debug = debug;

    // enable "random search" mode
    if (num_samples == 1) {
      rs_mode = true;
      num_samples = this->num_replications;
      this->num_replications = 1;
    }

    if (debug) {
      printf("variance: %.10g\n", stddev * stddev);
      printf("num_replications: %d\n", num_replications);
      printf("num_samples: %d\n", num_samples);
    }
  }

  void estimate(DiscoGradProgram<num_inputs> &program) {
    for (int param_comb = 0; param_comb < num_param_combs; param_comb++) {
      this->seed = this->seed_arg;
      if (this->seed == -1)
        this->seed = random_device()();

      this->rep_seed_gen.seed(this->seed);
      this->seed_dist = uniform_int_distribution<unsigned>(0, numeric_limits<unsigned>::max());

      for (int dim = 0; dim < num_inputs; dim++) {
        double p;
        if (scanf("%lf", &p) != 1) {
          printf("program expects %d parameters, exiting\n", num_inputs);
          exit(1);
        }
        parameters[dim] = p;
        parameters[dim].set_initial_tang(dim, 1);
      }

      this->exp_val = 0.0;

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
  double lowest_val() const { return lowest_sample_val; }
  /** The (expected) program derivative for input dimension dim of the most recent estimation. */
  virtual double derivative(int dim) const { return exp_val.get_tang(dim); }
};

// choose smoothing variety
#if defined PGO
  #include "polyak_gradient_oracle/discograd.hpp"
#elif defined REINFORCE
  #include "reinforce/discograd.hpp"
#elif defined DGO
  #include "discograd_gradient_oracle/discograd.hpp"
#elif defined CRISP
  #include "crisp/discograd.hpp"
#endif
