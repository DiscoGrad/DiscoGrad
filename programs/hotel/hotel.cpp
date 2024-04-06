/**
 * Description
 * ===========
 *
 * This model implements a hotel revenue management model based on
 * https://simopt.readthedocs.io/en/latest/hotel.html. The model
 * simulates a hotel receiving product requests for a week. These
 * products vary in arrival date, stay length and rate paid. The
 * model inputs decide, how many of each product can be ordered.
 * The model output is the gained revenue. The model is stochastic
 * as the product requests' arrival processes are stationary Poisson
 * processes.
 *
 */

#include <stdio.h>
#include <math.h>
#include <numeric>

const int num_inputs = 56;

#include "backend/discograd.hpp"

using namespace std;

double get_new_arrival_time(DiscoGrad<num_inputs> &_discograd, double last_arrival, double lambda) {
  static uniform_real_distribution<> dist(0,1);

  double inter_arrival = -log(dist(_discograd.rng)) / lambda;
  return last_arrival + inter_arrival;
}

adouble _DiscoGrad_f(DiscoGrad<num_inputs> &_discograd, aparams &in) {
  const int num_products = 56;

  printf("Running with following booking limits:\n");
  for (int i = 0; i < num_products; i++) {
    if (in[i].val > 100)
      in[i].val = 100;
    printf("%.0f ", in[i].val);
  }
  printf("\n");

  bool is_rackrate[num_products];
  for (int i = 0; i < num_products; i++)
    is_rackrate[i] = (i % 2 == 0);

  const int discount_rate = 100;
  const int rack_rate = 200;

  const float lambdas_per_stay_length[7] = {1.0 / 168, 2.0 / 168, 3.0 / 168, 2.0 / 168, 1.0 / 168, 0.5 / 168, 0.25 / 168};
  float lambdas[num_products];
  int cost[num_products];
  int incidence[7][num_products];
  int time_limit[num_products];

  //initialize model variables such as product incidence
  int index = 0;
  for (int start = 0; start < 7; start++) {
    for (int end = start; end < 7; end++) {
      for (int day = 0; day < 7; day++) {
        incidence[day][index] = (start <= day && day <= end) ? 1 : 0;
        incidence[day][index + 1] = incidence[day][index];
      }

      int stay_length = end - start + 1;
      lambdas[index] = lambdas_per_stay_length[stay_length - 1];
      lambdas[index + 1] = lambdas[index];

      time_limit[index] = 3 + 24 * (start + 1);
      time_limit[index + 1] = time_limit[index];

      cost[index] = stay_length * rack_rate;
      cost[index+1] = stay_length * discount_rate;

      index += 2;
    }
  }

  const int run_length = 168;
  const int time_before = 168;
  const int start_time = -time_before;
  const int samples = 5;

  adouble booking_limits_base[num_products];
  adouble booking_limits[num_products];
  // note that limits should be capped at 100. This is not checked anymore and is to be ensured using the input parameters
  for (int i = 0; i < num_products; i++) {
    booking_limits_base[i] = adouble(in[i]);
  }

  // run #samples times
  adouble mean_revenue = 0.0;
  for (int run_ind = 0; run_ind < samples; run_ind++) {
    for (int i = 0; i < num_products; i++)
      booking_limits[i] = booking_limits_base[i];

    adouble revenue = 0;
    // starting arrival times
    double arrival_times[num_products];
    for (int i = 0; i < num_products; i++) {
      arrival_times[i] = get_new_arrival_time(_discograd, start_time, lambdas[i]);
    }

    for (int hour = start_time; hour < run_length; hour++) {
      vector<int> requested_products_indices = vector<int>();
      // accumulate currently arriving requests
      for (int i = 0; i < num_products; i++) {
        if (arrival_times[i] <= hour && hour <= time_limit[i]) {
          requested_products_indices.push_back(i);
          arrival_times[i] = get_new_arrival_time(_discograd, arrival_times[i], lambdas[i]);
        }
      }

      for (auto &requested_index: requested_products_indices) {
        if (booking_limits[requested_index] > 0.5) {
          revenue += cost[requested_index];

          int requested_product[7];
          for (int day = 0; day < 7; day++) {
            requested_product[day] = incidence[day][requested_index];
          }
          for (int other_index = 0; other_index < num_products; other_index++) {
            int other_product[7];
            for (int day = 0; day < 7; day++) {
              other_product[day] = incidence[day][other_index];
            }
            int product_overlap = inner_product(requested_product, requested_product + 7, other_product, 0);
            if (product_overlap >= 1) {
              booking_limits[other_index] -= 1;
            }
          }

          //printf("%d booked at time %d. %d revenue.\n", requested_index + 1, hour, cost[requested_index]);
        } else {
          //printf("Product %d could not be requested at time %d.\n", requested_index + 1, hour);
        }
      }
    }

    printf("Revenue of run %d: %f\n", run_ind, revenue.val);
    mean_revenue += revenue;
  }

  mean_revenue /= samples;
  adouble loss = -mean_revenue;
  return loss;
}

int main(int argc, char **argv) {

  DiscoGrad<num_inputs> dg(argc, argv);
  DiscoGradFunc<num_inputs> func(_DiscoGrad_f);
  dg.estimate(func);

  return 0;
}
