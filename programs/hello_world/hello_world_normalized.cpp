/** Quickstart example showcasing the basic usage of DiscoGrad. */

// The forward-mode automatic differentiation needs to know the number of inputs
// to the program. This needs to be defined *before* the inclusion of the
// header.
const int num_inputs = 1;
// include the discograd header
#include "backend/discograd.hpp"

using namespace std;

/* Functions with the _DiscoGrad_ prefix are smoothed upon transformation.
 * The typedef aparams is for array<adouble, num_inputs>.
 * The typedef adouble is set depending on the mode of AD you chose (e.g. as
 * fw_adouble<num_inputs>).
 */
adouble _DiscoGrad_heaviside(DiscoGrad<num_inputs> &_discograd, aparams &p) {
  if (p[0] < 0) {
    return 0;
  } else {
  }

  return 1;
}

/* Wrapper class for the smoothed function (only necessary if the smoothed func
 * takes extra parameters) Otherwise the function can be wrapped directy (see
 * main).
 */
// class HelloSmoothing : public DiscoGradProgram<num_inputs> {
//   adouble run(DiscoGrad<num_inputs> &_discograd, array<adouble, num_inputs>
//   &p) {
//     return _DiscoGrad_heaviside(_discograd, p);
//   }
// };

int main(int argc, char **argv) {
  // create a new instance of the discograd estimator.
  // this will read the parameters (of which there are `num_inputs`) from stdin.
  // additional configuration options, such as the smoothing variance, are read
  // from the cli arguments.
  DiscoGrad<num_inputs> dg(argc, argv, false);
  // instantiate the program wrapper for the heaviside function
  // HelloSmoothing prog;
  DiscoGradFunc<num_inputs> func(_DiscoGrad_heaviside);

  // estimate or calculate output and gradient
  dg.estimate(func);

  return 0;
}
