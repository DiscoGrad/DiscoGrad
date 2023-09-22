/** Quickstart example showcasing the basic usage of DiscoGrad. */

// The forward-mode automatic differentiation needs to know the number of inputs to the program.
// This needs to be defined *before* the inclusion of the header.
const int num_inputs = 1;
// include the discograd header
#include "backend/discograd.hpp"

using namespace std;

/* Functions with the _DiscoGrad_ prefix are smoothed upon transformation.
 * The typedef aparams is for array<adouble, num_inputs>.
 * The typedef adouble is set depending on the mode of AD you chose (e.g. as fw_adouble<num_inputs>).
 */
adouble _DiscoGrad_heaviside(DiscoGrad<num_inputs> &_discograd, aparams &p)
{
  sdouble x({p[0], _discograd.get_variance()}); // initialize the single input value with the input variance
  sdouble y;
  if (x <= 0)                   // a smoothed branch
    y = 1;
  else
    y = 0;
  y.print();                    // for debugging
  return y.expectation();       // return the smoothed output (expected value) 
}

/* Wrapper class for the smoothed function (only necessary if the smoothed func takes extra parameters) 
 * Otherwise the function can be wrapped directy (see main).
 */
//class HelloSmoothing : public DiscoGradProgram<num_inputs> {
//  adouble run(DiscoGrad<num_inputs> &_discograd, array<adouble, num_inputs> &p) {
//    return _DiscoGrad_heaviside(_discograd, p); 
//  }
//};

int main(int argc, char **argv)
{
  // ceate a new instance of the discograd estimator.
  // (can be chosen at compile time as `-DSI` or `-DSAMPLING`)
  // this will read the parameters (of which there are `num_inputs`) from stdin.
  // additional configuration options, such as the smoothing variance, are read from the cli arguments.
  DiscoGrad<num_inputs> dg(argc, argv, true);
  // instantiate the program wrapper for the heaviside function
  //HelloSmoothing prog;
  DiscoGradFunc<num_inputs> func(_DiscoGrad_heaviside);

  // estimate or calculate output and gradient
  dg.estimate(func);

  return 0;
}
