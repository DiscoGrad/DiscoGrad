/*
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

#include <random>
#include <array>

using namespace std;

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {

private:
  array<double, num_inputs> perturbations = {};
  double exp = 0.0;
  double deriv[num_inputs] = { 0.0 };

public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {};

  double deriv_log_norm_pdf(double x, double mu) { return (x - mu) / this->variance; }

  /** 
   * REINFORCE as proposed in Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256
   * https://doi.org/10.1007/BF00992696
   */
  void estimate_(DiscoGradProgram<num_inputs> &program) {
    for (int rep = 0; rep < this->num_replications; ++rep) {
      this->current_seed = this->seed_dist(this->rep_seed_gen);
      this->rng.seed(this->current_seed);
      for (int sample = 0; sample < this->num_samples; ++sample) {
        array<adouble, num_inputs> pm_perturbed;
        for (int dim = 0; dim < num_inputs; ++dim) {
          if (this->perturbation_dim == -1 || this->perturbation_dim == dim)
            perturbations[dim] = this->normal_dist(this->sampling_rng);

          pm_perturbed[dim] = this->parameters[dim] + perturbations[dim];
        }
        // execute program on perturbed parameters
        this->rng.seed(this->current_seed);
        double perturbed = program.run(*this, pm_perturbed).get_val(); // f(x+u*stddev)
        this->lowest_sample_val = min(this->lowest_sample_val, perturbed);
        exp += perturbed;
        for (int dim = 0; dim < num_inputs; ++dim)
          deriv[dim] += perturbed * deriv_log_norm_pdf(pm_perturbed[dim].val, pm_perturbed[dim].val - perturbations[dim]) / this->num_samples;
      }
    }
    
    // statistics over replications
    this->exp_val = (exp / this->num_samples) / this->num_replications;
    for (int dim = 0; dim < num_inputs; ++dim)
      deriv[dim] /= this->num_replications;
  }

  double derivative(int dim) const { return deriv[dim]; }
};
