/*  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
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

const int num_inputs = 1;
#include "backend/discograd.hpp"
//constexpr double thresholds[] = { 0.7, -0.5, 0.8, -0.3, 0.6, -0.45, 0.38, -0.63};
double thresholds[32];

uniform_real_distribution<double> dist(-1, 1);
default_random_engine rng;

adouble _DiscoGrad_synthetic_test(DiscoGrad<num_inputs> &_discograd, aparams &p)
{
  sdouble x({p[0], _discograd.get_variance()});
  sdouble y = x / 2.0;              // correlate y with x
  sdouble v = x - y;                // here, y is correlated with x, necessitating DEA
  for (int i = 0; i < 32; ++i) {     // this loop just "simulates" repeated branching behaviour
    if (v < thresholds[i]) {
      v -= thresholds[i];           // subsequent branches depend on previous branches
    }
  }
  return v.expectation();
}

int main(int argc, char **argv)
{
  rng.seed(atoi(argv[1]));
  for (int i = 0; i < 32; i++) {
    thresholds[i] = dist(rng);
    printf("%lf\n", thresholds[i]);
  }
  DiscoGrad<num_inputs> dg(argc, argv, false);
  DiscoGradFunc<num_inputs> func(_DiscoGrad_synthetic_test);
  dg.estimate(func);
  return 0;
}
