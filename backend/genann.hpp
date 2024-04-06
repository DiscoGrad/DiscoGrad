/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * Adapted for use with DiscoGrad by Philipp Andelfinger
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#pragma once

#include <stdio.h>
#include <math.h>

template <int num_inputs, int inputs, int hidden_layers, int hidden, int outputs> class genann {
public:
  static const int hidden_weights =
      hidden_layers ? (inputs + 1) * hidden + (hidden_layers - 1) * (hidden + 1) * hidden : 0;
  static const int output_weights = (hidden_layers ? (hidden + 1) : (inputs + 1)) * outputs;
  static const int total_weights = (hidden_weights + output_weights);
  static const int total_neurons = (inputs + hidden * hidden_layers + outputs);

  adouble weight[total_weights];
  adouble output[total_neurons];

  //static adouble act(adouble &a) { return tanh(a); }
  static adouble act(adouble &a) { return (adouble)1.0 / (exp(-a) + 1); }
  static adouble act_output(adouble a) { return (adouble)1.0 / (exp(-a) + 1); }

  genann(array<adouble, num_inputs> &p, int offset = 0) {
    for (int i = 0; i < total_weights; i++)
      weight[i] = p[i + offset];
  }

  adouble *run(adouble *in) {
    adouble *w = weight;
    adouble *o = output + inputs;
    adouble *i = output;

    for (int i = 0; i < inputs; i++)
      output[i] = in[i];

    int h, j, k;

    if (!hidden_layers) {
      adouble *ret = o;
      for (j = 0; j < outputs; ++j) {
        adouble sum = *w++ * -1.0;
        for (k = 0; k < inputs; ++k) {
          sum += *w++ * i[k];
        }
        *o++ = act_output(sum);
      }

      return ret;
    }

    for (j = 0; j < hidden; ++j) {
      adouble sum = *w++ * -1.0;
      for (k = 0; k < inputs; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = act(sum);
    }

    i += inputs;

    for (h = 1; h < hidden_layers; ++h) {
      for (j = 0; j < hidden; ++j) {
        adouble sum = *w++ * -1.0;
        for (k = 0; k < hidden; ++k) {
          sum += *w++ * i[k];
        }
        *o++ = act(sum);
      }

      i += hidden;
    }

    adouble *ret = o;

    for (j = 0; j < outputs; ++j) {
      adouble sum = *w++ * -1.0;
      for (k = 0; k < hidden; ++k) {
        sum += *w++ * i[k];
      }
      *o++ = act_output(sum);
    }

    assert(w - weight == total_weights);
    assert(o - output == total_neurons);

    return ret;
  }
};
