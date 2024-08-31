/** Backend library of the sampling-enabled version of DiscoGrad.
 *  Contains classes and functions for enabling a smooth interpretation of C++ programs.
 *
 * Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *  
 *   The above copyright notice and this permission notice shall be included in all copies or
 *   substantial portions of the Software.
 *   
 *   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE. */

using namespace std;

extern uint64_t global_branch_id;
extern uint32_t branch_level;

template<int num_inputs>
class DiscoGradBase;

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {
public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {};

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->sampling_rng.seed(random_device()());
    for (uint64_t rep = 0; rep < this->num_replications; ++rep) {

      if (!this->rs_mode)
        this->current_seed = this->seed_dist(this->rep_seed_gen);

      for (uint64_t sample = 0; sample < this->num_samples; sample++) {
        if (this->rs_mode)
          this->current_seed = this->seed_dist(this->rep_seed_gen);

        
        array<adouble, num_inputs> pm_perturbed = this->parameters;
        array<double, num_inputs> perturbation = {};

        if (this->stddev > 0) {
          for (int dim = 0; dim < num_inputs; dim++) {
            if (this->perturbation_dim == -1 || this->perturbation_dim == dim) {
              perturbation[dim] = this->normal_dist(this->sampling_rng);
            }

            pm_perturbed[dim] += perturbation[dim];
          }
        }

        this->rng.seed(this->current_seed);

        adouble r = program.run(pm_perturbed);
        this->exp_val += r;
      }
    }
    this->exp_val /= this->num_replications * this->num_samples;
  }
};
