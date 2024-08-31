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
  vector<array<double, num_inputs>> perturbations;
  double exp = 0.0;
  double deriv[num_inputs] = { 0.0 };

public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {};

  double deriv_log_norm_pdf(double x, double mu) { return (x - mu) / this->variance; }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    vector<double> perturbed;
    for (uint64_t rep = 0; rep < this->num_replications; ++rep) {
      this->current_seed = this->seed_dist(this->rep_seed_gen);
      this->rng.seed(this->current_seed);
      for (uint64_t sample = 0; sample < this->num_samples; ++sample) {
        array<double, num_inputs> perturbation;

        array<adouble, num_inputs> pm_perturbed;
        for (int dim = 0; dim < num_inputs; ++dim) {
          if (this->perturbation_dim == -1 || this->perturbation_dim == dim)
            perturbation[dim] = this->normal_dist(this->sampling_rng);

          pm_perturbed[dim] = this->parameters[dim] + perturbation[dim];
        }

        perturbations.push_back(perturbation);

        // execute program on perturbed parameters
        this->rng.seed(this->current_seed);
        perturbed.push_back(program.run(pm_perturbed).get_val()); // f(x+u*stddev)
        exp += perturbed.back();
      }
    }

    int total_runs = this->num_samples * this->num_replications;
    this->exp_val = exp / total_runs;

    for (int s = 0; s < total_runs; s++) {
      for (int dim = 0; dim < num_inputs; ++dim) {
        double fs = perturbed[s];
        double b = (exp - fs) / (total_runs - 1);
        deriv[dim] += (fs - b) * deriv_log_norm_pdf(perturbations[s][dim], 0);
      }
    }

    for (int dim = 0; dim < num_inputs; ++dim)
      deriv[dim] /= total_runs;
  }

  double derivative(int dim) const { return deriv[dim]; }
};
