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
  double exp = 0.0;
  double deriv[num_inputs] = { 0.0 };

public:
  /** Estimator according to a formulation of Nesterov and Spokoiny
   * - Basic scheme discussed in B. Polyak, Introduction to Optimization. Optimization Software - Inc., Publications Division, New York, 1987
   * - Convergence of optimization scheme analyzed in Nesterov and Spokoiny, Random Gradient-Free Minimization of Convex Functions. Found Comput Math 17, 527-566 
   *   (https://link.springer.com/article/10.1007/s10208-015-9296-2)
   *
   *    f(x)     = 1/num_samples * sum( f(x+u*stddev) )                                                 [Monte-Carlo approximation]
   *    df(x)/dx = 1/num_samples * sum( (f(x+u*stddev) - f(x)) / stddev * u); u ~ N(0, 1, num_inputs)   [Gradient Estimator linked above]
   *
   *    The above is done for each replication and then averaged again over all replications.
   */
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {};

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    assert(this->stddev > 0);

    default_random_engine reference_seed_gen(this->seed + 1);
    for (uint64_t rep = 0; rep < this->num_replications; ++rep) {

      if (this->rs_mode) // single reference, one or more _unrelated_ reps (here: equal to samples)
        this->current_seed = this->seed_dist(reference_seed_gen);
      else // single reference per rep, one or more samples with the _same_ rep seed
        this->current_seed = this->seed_dist(this->rep_seed_gen);

      this->rng.seed(this->current_seed);

      double crisp_ref = program.run(this->parameters).get_val(); // f(x)

      array<double, num_inputs> perturbation = {};

      for (uint64_t sample = 0; sample < this->num_samples; sample++) {

        if (this->rs_mode)
          this->current_seed = this->seed_dist(this->rep_seed_gen);
 
        normal_distribution<double> normal_dist(0, 1);
        array<adouble, num_inputs> pm_perturbed = this->parameters;
        for (int dim = 0; dim < num_inputs; ++dim)
        {
          if (this->perturbation_dim == -1 || this->perturbation_dim == dim)
            perturbation[dim] = normal_dist(this->sampling_rng);

          pm_perturbed[dim] += perturbation[dim] * this->stddev;
        }
        // execute program on perturbed parameters
        this->rng.seed(this->current_seed);
        double perturbed = program.run(pm_perturbed).get_val(); // f(x+u*stddev)

        exp += perturbed;
        for (int dim = 0; dim < num_inputs; ++dim) {
          deriv[dim] += ((perturbed - crisp_ref) / this->stddev * perturbation[dim]) / this->num_samples; // 1/num_samples * sum( (f(x+u*stddev) - f(x)) / stddev * u )
        }
      }
    }
    
    // statistics over replications
    this->exp_val = (exp / this->num_samples) / this->num_replications;
    for (int dim = 0; dim < num_inputs; ++dim) {
      deriv[dim] /= this->num_replications;
    }
  }

  double derivative(int dim) const { return deriv[dim]; }
};
