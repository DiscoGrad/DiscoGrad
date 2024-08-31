/*  Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
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

/** 
 * Description
 * ===========
 *
 * This model implements a neural-network controlled air condition system.
 * A deep network, turns cooling on or off and also
 * determines the amount of cooling. The environment starts at a random
 * and with a random target temperature, which gradually approaches the outside
 * temperature, depending on a random insulation coefficient. The insulation may
 * experience a sudden drop when a window is opened.
 * The task of the network is to keep the temperature as close to a target
 * temperature as possible while maintaining low energy consumption.
 * The loss is the average error in temperature plus the energy consumption,
 * which is proportional to the degree of heating/cooling
 * per step.
 */

#include <stdio.h>
#include <cstdarg>

using namespace std;

constexpr int nn_inputs = 5;
constexpr int nn_hidden_layers = 1;
constexpr int nn_hidden = 10;
constexpr int nn_outputs = 2;

constexpr int nn_hidden_weights = nn_hidden_layers ? (nn_inputs+1) * nn_hidden +
                              (nn_hidden_layers-1) * (nn_hidden+1) * nn_hidden : 0;
constexpr int nn_output_weights = (nn_hidden_layers ? (nn_hidden+1) : (nn_inputs+1)) * nn_outputs;

constexpr int nn_total_weights = nn_hidden_weights + nn_output_weights;

constexpr int num_inputs = nn_total_weights;
#include "backend/discograd.hpp"
#include "backend/genann.hpp"

const bool print_debug = false;

random_device rd;

inline void printf_debug(const char *format, ...)
{
  if(!print_debug)
    return;

  va_list args;
  va_start(args, format);
  vfprintf(stdout, format, args);
  va_end(args);
}

adouble _DiscoGrad_f(DiscoGrad<num_inputs> &_discograd, array<adouble, num_inputs> &p, double init_temp)
{
  // define (smoothed) neural networks
  genann<num_inputs, nn_inputs, nn_hidden_layers, nn_hidden, nn_outputs> nn(p, 0);

  const int num_steps = 10; // determines the endtime of the simulation

  uniform_real_distribution target_temp_dist(18.0, 22.0);
  uniform_real_distribution insulation_dist(0.75, 0.99);
  uniform_real_distribution u;

  double target_temp = target_temp_dist(_discograd.rng); // the temperature the thermostat should maintain
  uniform_real_distribution outside_temp_dist(target_temp, 35.0);
  double outside_temp = outside_temp_dist(_discograd.rng);
  uniform_real_distribution initial_temp_dist(target_temp, outside_temp);
  double initial_temp = initial_temp_dist(_discograd.rng); // starting temperature
  double insulation = insulation_dist(_discograd.rng);

  double window_open_prob = 0.05;
  double window_open_insulation = 0.75;

  adouble nn_in[num_inputs];

  adouble prev_temp = initial_temp;

  nn_in[0] = (initial_temp * insulation) + (outside_temp * (1 - insulation));
  nn_in[1] = target_temp;
  nn_in[2] = prev_temp;

  nn_in[3] = 0;
  nn_in[4] = 0;

  adouble *nn_out = nn.run(nn_in); // dummy run without effect on the loss to initialize nn_in[3] and nn_in[4]

  nn_in[3] = nn_out[0];
  nn_in[4] = nn_out[1];

  printf_debug("initial temp %lf\n", initial_temp);
  printf_debug("temp %lf\n", nn_in[0].val);
  printf_debug("target temp %lf\n", target_temp);
  printf_debug("outside temp %lf\n", outside_temp);

  adouble loss = 0.0;

  for (int i = 0; i < num_steps; i++) {
    nn_in[2] = prev_temp;
    prev_temp = nn_in[0];

    printf_debug("temp: %.4g (%.4g degrees)\n", nn_in[0].val, nn_in[0].val);

    adouble energy_penalty = 0;

    adouble *nn_out = nn.run(nn_in);

    if (nn_out[0] >= 0.5) { // enable cooling unit
      adouble cooling = nn_out[1] * 3;
      nn_in[0] -= cooling; // amount of cooling
      printf_debug("cooling on: %lf\n", cooling.val);
      energy_penalty += nn_out[1] + 0.5;
    } else {
      printf_debug("cooling off\n");
    }

    if (u(_discograd.rng) < window_open_prob)
      insulation = window_open_insulation;

    nn_in[0] = (nn_in[0] * insulation) + (outside_temp * (1 - insulation));

    nn_in[3] = nn_out[0];
    nn_in[4] = nn_out[1];

    printf_debug("temp %lf\n", nn_in[0].val);

    adouble err = target_temp - nn_in[0];
    printf_debug("err %lf, adding %lf to loss\n", err.val, sqrt(err.val * err.val));
    loss += sqrt(err * err);
    loss += energy_penalty;
  }

  loss /= num_steps;

  return loss;
}

class NNThermostat : public DiscoGradProgram<num_inputs> {
public:
  double init_temp;
  NNThermostat(DiscoGrad<num_inputs>& _discograd, double init_temp) : DiscoGradProgram<num_inputs>(_discograd) {
    this->init_temp = init_temp;
  }
  adouble run(array<adouble, num_inputs> &p) {
    return _DiscoGrad_f(_discograd, p, init_temp); 
  }
};

int main(int argc, char **argv)
{
  DiscoGrad<num_inputs> dg(argc, argv);
  NNThermostat prog(dg, 20.0);
  dg.estimate(prog);

  return 0;
}
