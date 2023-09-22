/** Backend library for the SI-enabled version of DiscoGrad.
 *  Contains classes and functions for enabling a smooth interpretation of C++ programs.
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

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cstdint>
#include <deque>
#include <fenv.h>
#include <math.h>
#include <random>
#include <stack>
#include <stdio.h>
#include <unordered_map>
#include <vector>
#include <queue>
#include <array>
#include <iostream>
#include "si_state.hpp"
#include "si_types.hpp"

using namespace std;

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {

private:
  double variance = 0.0;
  default_random_engine sampling_rng;
  array<double, num_inputs> perturbations;

public:
  /** @copydoc DiscoGradBase 
   *
   */
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {
    // important: assign a random seed to the generator
    sampling_rng.seed(random_device()());

    string path(argv[0]);
    args::ArgParser parser("Usage: " + path + " -s [seed = 1] --nc [#parameter combinations = 1] --nr [#replications = 1] --var [variance = 1] --np [#paths = 8] --rm [restrict mode (Ch, WO, IW, Di) = Ch] --up_var [uncertainty propagation input variance = 0]");

    parser.option("s");
    parser.option("nc");
    parser.option("nr");
    parser.option("var");
    parser.option("np");
    parser.option("rm");
    parser.option("up_var");

    parser.parse(argc, argv);

    if (parser.found("s"))
      this->seed_arg = stoi(parser.value("s"));

    if (parser.found("nc"))
      this->num_param_combs = stoi(parser.value("nc"));
    
    if (parser.found("nr"))
      this->num_replications = stoi(parser.value("nr"));

    this->variance = 1;
    if (parser.found("var"))
      this->variance = stof(parser.value("var"));

    int maxPathStates = 8;
    if (parser.found("np"))
      maxPathStates = stoi(parser.value("np"));

    int restrictMode = 0;
    auto restrict_modes = {"Ch", "WO", "IW", "Di"};
    if (parser.found("rm")) {
      string val = parser.value("rm");
      auto r = find(restrict_modes.begin(), restrict_modes.end(), val);
      if (r == restrict_modes.end()) {
        printf("Unkown restrict mode %s\n", val.c_str());
        exit(1);
      }
      restrictMode = r - restrict_modes.begin();
    }

    double dea_input_variance = 0.0;
    if (parser.found("up_var"))
      dea_input_variance = stof(parser.value("up_var"));

    si_stack.set_restrict_mode(restrictMode);
    si_stack.set_max_path_states(maxPathStates);
    si_stack.set_dea_input_variance(dea_input_variance);

    if (debug) {
      printf("variance: %lf\n", variance);
      printf("maxPathStates: %d\n", maxPathStates);
      printf("restrictMode: %s\n", *(restrict_modes.begin() + restrictMode));
    }
    this->debug = debug;
  }

  double get_variance() { return this->variance; }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->exp_val = 0.0;
    for (int rep = 0; rep < this->num_replications; ++rep) {
      this->current_seed = this->seed_dist(this->rep_seed_gen);
      this->rng.seed(this->current_seed);
      // run the program and accumulate value and gradient
      // its ok to do this on an adouble, because (f+g)' = f' + g'
      this->exp_val += program.run(*this, this->parameters);
      if (this->num_replications > 1)
        si_stack.clear();
    }
    // calculate mean
    // its ok to do this on an adouble, because (f/c)' = f'/c (for constant c)
    this->exp_val /= this->num_replications;

    printf("number of branches: %ld\n", si_stack.num_branches);
    printf("maximum number of paths: %ld\n", si_stack.max_instantiated_path_states);
    printf("average number of paths: %.2f\n", (double)si_stack.cumulative_instantiated_path_states / si_stack.num_branches);
    printf("average number of variables: %.2f\n", (double)si_stack.cumulative_num_variables / si_stack.num_branches);
  }
};

