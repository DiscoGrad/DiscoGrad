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

#include "../vec2.hpp"
#include <fenv.h>

#ifndef NUM_BINS
#define NUM_BINS 10
#endif

#ifndef NUM_AGENTS
#define NUM_AGENTS 10
#endif

#ifndef SF_AD
#define SF_AD true
#endif

#ifndef DGO_IGNORE_SF_BRANCHES
#define DGO_IGNORE_SF_BRANCHES false
#endif

#ifndef INTERPOLATE_THETA_SIGN
#define INTERPOLATE_THETA_SIGN false
#endif

#ifndef RETURN_Y_POS
#define RETURN_Y_POS false
#endif

#ifndef HIST_EXPERIMENT
#define HIST_EXPERIMENT false
#endif

#ifndef LEAPFROG
#define LEAPFROG true
#endif

#ifndef PRINT_TRACE
#define PRINT_TRACE false
#endif

#ifndef CALIBRATION
#define CALIBRATION false
#endif

const int num_agents = NUM_AGENTS;
#if HIST_EXPERIMENT == true
const int num_emp_dist_bins = NUM_BINS;
const int num_inputs = NUM_BINS;
#else
const int num_inputs = 3;
#endif

#include "backend/discograd.hpp"

using namespace std;

#if SF_AD == true
typedef adouble xdouble;
#define VAL(X) X.val
#else
typedef double xdouble;
#define VAL(X) X
#endif

constexpr double scenario_width = 30;
constexpr double interaction_radius = 3.0;
constexpr double cell_width = interaction_radius;
const double delta_t = 0.1;

const double min_w = 0.5;

const bool print_trace = PRINT_TRACE;

typedef int aid_t;
typedef vec2<int> int2;
typedef avec<2, num_inputs> dbl2;

const double v_desired_mean = 1.29;
const double v_desired_stddev = 0.1;

const int end_step = 200;

const double lambda = 2.0;
const double gamma_ = 0.35;
const double n = 2;
const double n_prime = 3;

const double sigma = 0.8;

const double waypoint_tol = 3; // in this radius the wp is considered to be reached
const double congestion_radius = 10; // agents in this radius are counted as part of the congestion around the waypoint
const double agent_radius = 0.4;

double t_sim = 0;

#if CALIBRATION
#if RETURN_Y_POS

#if NUM_AGENTS == 3
const double ref_r = 20.0;
#elif NUM_AGENTS == 10
const double ref_r = 20.0;
#else
const double ref_r = 10.0;
#endif

#else // RETURN_Y_POS == false

#if NUM_AGENTS == 3
const double ref_total_num_measurement_evacs = 2;
#elif NUM_AGENTS == 10
const double ref_total_num_measurement_evacs = 5;
#else
const double ref_total_num_measurement_evacs = 50;
#endif

#endif
#endif

const double pi = 3.141592653589793;


const double door_y = 0.5 * scenario_width;
const double door_width = 4;
const double door_congestion_offset = 3;
const double door_wp_offset = 6;

pair<dbl2, dbl2> obstacles[] = {
  {{door_y, 0}, {door_y, 0.5 * scenario_width - door_width / 2}},
  {{door_y, 0.5 * scenario_width + door_width / 2}, {door_y, scenario_width}}};
size_t num_obs = sizeof(obstacles) / sizeof(pair<dbl2, dbl2>);

// y, x coordinates of waypoints and the expected center of congestion
dbl2 wp_alternatives[][2] = {
  {{0.5 * scenario_width + door_wp_offset, 0.5 * scenario_width},
    {0.5 * scenario_width - door_congestion_offset, 0.5 * scenario_width}},
};

template<int num_bins>
struct EmpDist {

  array<adouble, num_bins> hist;
  const int max_num_checks = ceil(log2(num_bins));
  double bin_width;
  double min_val, max_val;

  EmpDist(double min_val, double max_val, adouble *p) : min_val(min_val), max_val(max_val) {
    bin_width = (max_val - min_val) / num_bins;
    adouble p_sum = 0;
    for (int bin = 0; bin < num_bins; bin++) {
      adouble v = p[bin];
      if (v < 0)
        v = -v;
      p_sum += v;
      hist[bin] = p_sum;
    }

    for (int bin = 0; bin < num_bins; bin++) {
      hist[bin] /= p_sum;
    }
  }

  double _DiscoGrad_draw(DiscoGrad<num_inputs>& _discograd) {
    uniform_real_distribution<double> u_dist;
    double u = u_dist(_discograd.rng);
    adouble sum = 0;

    int res_bin = -1;
    for (int bin = 0; bin < num_bins; bin++) {
      if (res_bin == -1) {
        if (hist[bin] > u) {
          res_bin = bin;
        }
      }
    }

    double res_val = min_val + bin_width * (res_bin + 0.5);
    return res_val;
  }
};

xdouble norm(dbl2& vec) {
  dbl2 s = vec * vec;

  return sqrt(s[0] + s[1]);
}

xdouble dot(dbl2& a, dbl2&b) { return a[0] * b[0] + a[1] * b[1]; }

struct Agent {
  int aid;
  dbl2 p;
  dbl2 v;
  dbl2 a_old;

  dbl2 waypoint;

  int2 cell = {-1, -1};

  xdouble v_desired;

  xdouble waypoint_dist() {
    dbl2& wp = waypoint;
     if (wp[0] == -1)
       return abs(VAL(p[1]) - VAL(wp)[1]);
     if (wp[1] == -1)
       return abs(VAL(p[0]) - VAL(wp)[0]);

     dbl2 dist = p - wp;
     return dist.norm();
   }

  bool operator==(Agent& other) { return aid == other.aid; }

  bool is_neighbor(Agent &other) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int2 nb_cell = { cell[0] + dy, cell[1] + dx };
        if (other.cell == nb_cell)
          return true;
      }
    }
    return false;
  }
};

Agent agents[num_agents];
int num_active_agents = 0;

adouble normalize_param(adouble p, double min, double max, adouble p_sum) {
  printf("p is %.2f, p_sum is %.2f\n", p.val, p_sum.val);
  return (p / p_sum) * (min + (max - min));
}

void spawn_agent(DiscoGrad<num_inputs>& _discograd) {
  uniform_real_distribution<double> u_dist;
  normal_distribution<double> v_desired_dist(v_desired_mean, v_desired_stddev); // normal_distribution is stateful
  auto& ego = agents[num_active_agents];

  ego.aid = num_active_agents;
  
  ego.p = {u_dist(_discograd.rng) * 0.3 * scenario_width, (0.15 + u_dist(_discograd.rng) * 0.5) * scenario_width};

  ego.cell = { (int)(VAL(ego.p[0]) / cell_width), (int)(VAL(ego.p[1]) / cell_width) };
  ego.waypoint = wp_alternatives[0][0];

#if HIST_EXPERIMENT == true
  ego.v_desired = v_desired_dist._DiscoGrad_draw(_discograd);
#else
  ego.v_desired = v_desired_dist(_discograd.rng);
#endif

  ego.v = {0.0, 0.0};
  ego.a_old = { 0, 0 };

  num_active_agents++;
}

dbl2 left_normal(dbl2& vec) { return {-vec[1], vec[0]}; }

adouble _DiscoGrad_crowd(DiscoGrad<num_inputs> &_discograd, aparams &p)
{
  if (print_trace) {
    fprintf(stderr, "width %.4f\n", scenario_width);
    for (size_t oid = 0; oid < num_obs; oid++) {
      auto& o = obstacles[oid];
      if (print_trace)
        fprintf(stderr, "obstacle %.4f, %.4f; %4f, %.4f\n", VAL(o.first[0]), VAL(o.first[1]), VAL(o.second[0]), VAL(o.second[1]));
    }
    fprintf(stderr, "waypoint tol %.4f\n", waypoint_tol);
    fprintf(stderr, "congestion radius %.4f\n", congestion_radius);
    for (auto& wp : wp_alternatives) {
      fprintf(stderr, "waypoint %.4f, %.4f\n", VAL(wp[0][0]), VAL(wp[0][1]));
      fprintf(stderr, "congestion point %.4f, %.4f\n", VAL(wp[1][0]), VAL(wp[1][1]));
    }
    fprintf(stderr, "t");
  }

#if HIST_EXPERIMENT == true
  const double w_internal = 1;
  const double w_interaction = 2;
  const double w_obstacles = 1.25;

  EmpDist<num_emp_dist_bins> v_desired_dist(0.5, 2, &p[0]);
#else

  adouble w_internal = min_w + sqrt(p[0] * p[0]);
  adouble w_interaction = min_w + sqrt(p[1] * p[1]);
  adouble w_obstacles = min_w + sqrt(p[2] * p[2]);

  //printf("effective params: %.2f, %.2f, %.2f\n", w_internal.val, w_interaction.val, w_obstacles.val);
#endif

  num_active_agents = 0;

  for (int aid = 0; aid < num_agents; aid++) {
    spawn_agent(_discograd);

    if (print_trace)
      fprintf(stderr, ",a%d.active,a%d.y,a%d.x", aid, aid, aid);
  }
  if (print_trace)
    fprintf(stderr, "\n");

#if RETURN_Y_POS == false
  int total_num_measurement_evacs = 0;
#endif

  xdouble total_vel = 0;
  for (int step = 0; step < end_step; step++) {
    t_sim = step * delta_t;

    for (auto &ego : agents) {
      dbl2 f_internal = {0, 0};
      dbl2& wp = ego.waypoint;

      dbl2 t_dist = wp - ego.p;

      xdouble t_dist_norm = t_dist.norm();
      dbl2 e = t_dist / t_dist_norm;
      f_internal = ego.v_desired * e - ego.v;

      dbl2 f_interaction = {0, 0};

      dbl2 o_dist, o_dir, v_diff, int_v, int_dir, force_v, force_angle;
      xdouble o_dist_norm, int_norm, theta, B, npBt, nBt;

      for (auto &other : agents) {
        if (other != ego && ego.is_neighbor(other)) {
          o_dist = other.p - ego.p;

          o_dist_norm = o_dist.norm();
          o_dir = o_dist / o_dist_norm;

          v_diff = ego.v - other.v;

          int_v = lambda * v_diff + o_dir;

          int_norm = int_v.norm();
          int_dir = int_v / int_norm;

          xdouble angle_a = atan2(int_dir[1], int_dir[0]);
          xdouble angle_b = atan2(o_dir[1], o_dir[0]);
        
          theta = angle_b - angle_a;

          double theta_val = VAL(theta);

          if (theta_val > pi) { theta -= 2 * pi; }
          if (theta_val <= -pi) { theta += 2 * pi; }

          theta_val = VAL(theta);
       
          adouble theta_sign = 1;

#if INTERPOLATE_THETA_SIGN == true
          const double thresh = 0.1;
          if (DGO_IGNORE_SF_BRANCHES) {
            if (theta_val < thresh) {
              if (theta_val < -thresh)
                theta_sign = -1;
              else
                theta_sign = theta / thresh;
              //printf("theta: %.2g, theta_sign: %.2g\n", theta.val, theta_sign.val);
            }
          } else {
            // linear interpolation in [-thresh, thresh]
            if (theta < thresh) {
              if (theta < -thresh)
                theta_sign = -1;
              else
                theta_sign = theta / thresh;
              //printf("theta: %.2g, theta_sign: %.2g\n", theta.val, theta_sign.val);
            }
          }
#else
          if (DGO_IGNORE_SF_BRANCHES) {
            if (theta_val < 0.0)
              theta_sign = -1;
          } else {
            if (theta < 0.0)
              theta_sign = -1;
          }

#endif

          B = gamma_ * int_norm;

          npBt = n_prime * B * theta;
          nBt = n * B * theta;
          force_v = -exp(-o_dist_norm / B - npBt * npBt) * int_dir;
          force_angle = -theta_sign * exp(-o_dist_norm / B - nBt * nBt) * left_normal(int_dir);

          f_interaction += force_v + force_angle;
        } else {
        }
      }

      dbl2 &p = ego.p;
      pair<dbl2, dbl2> &o0 = obstacles[0], o1 = obstacles[1];

      dbl2 b_a_dist0 = o0.second - o0.first, b_a_dist1 = o1.second - o1.first;
      dbl2 p_a_dist0 = p - o0.first, p_a_dist1 = p - o1.first;

      xdouble lambda0 = p_a_dist0.dot(b_a_dist0) / b_a_dist0.squared_norm();
      xdouble lambda1 = p_a_dist1.dot(b_a_dist1) / b_a_dist1.squared_norm();
     
      dbl2 closest_point0 = o0.first + lambda0 * b_a_dist0;
      double lambda0_val = VAL(lambda0);
      double lambda1_val = VAL(lambda1);

      if (lambda0_val <= 0)
        closest_point0 = o0.first;
      if (lambda0_val >= 1)
        closest_point0 = o0.second;

      dbl2 closest_point1 = o1.first + lambda1 * b_a_dist1;
      if (lambda1_val <= 0)
        closest_point1 = o1.first;
      if (lambda1_val >= 1)
        closest_point1 = o1.second;

      dbl2 dist0 = p - closest_point0;
      dbl2 dist1 = p - closest_point1;

      xdouble dist_norm0 = dist0.norm();
      xdouble dist_norm1 = dist1.norm();

#if DGO_IGNORE_SF_BRANCHES == false
      //  note: if the obstacle is level with one of the axes and the closest point is not one of the end points,
      //        one dimension should be exactly 0 but is not for numerical reasons.
      //        this causes oscillation that creates lots of spurious branches
      double abs_dist_0_0 = abs(VAL(dist0[0])); // otherwise transformation occurs for the outer branches below
      double abs_dist_0_1 = abs(VAL(dist0[1]));
      double abs_dist_1_0 = abs(VAL(dist1[0]));
      double abs_dist_1_1 = abs(VAL(dist1[1]));
      if (abs_dist_0_0 > 1e-4) { if (dist0[0] < 0) {} }
      if (abs_dist_0_1 > 1e-4) { if (dist0[1] < 0) {} }
      if (abs_dist_1_0 > 1e-4) { if (dist1[0] < 0) {} }
      if (abs_dist_1_1 > 1e-4) { if (dist1[1] < 0) {} }
#endif

      xdouble norm0_full = dist_norm0 - agent_radius;
      xdouble norm1_full = dist_norm1 - agent_radius;

      dbl2 f_obstacles = exp(-norm0_full / sigma) * (dist0 / dist_norm0) +
                         exp(-norm1_full / sigma) * (dist1 / dist_norm1);

      dbl2 a = f_internal * w_internal +
               f_interaction * w_interaction +
               f_obstacles * w_obstacles;

#if LEAPFROG == true
      dbl2 p_new = ego.p + ego.v * delta_t + 0.5 * ego.a_old * delta_t * delta_t;
      dbl2 v_new = ego.v + 0.5 * (ego.a_old + a) * delta_t;
#else // ballistic update as described in Kesting and Treiber, Traffic Flow Dynamics: Data, Models and Simulation, 2013.
      dbl2 v_new = ego.v + a * delta_t;
      dbl2 p_new = ego.p + (ego.v + v_new) / 2 * delta_t;
#endif

      ego.v = v_new;
      ego.p = p_new;
      ego.a_old = a;

      // as a smooth branch, this creates high variance.
      // as a crisp branch, however, it causes bias when the path beyond the wp affects the return value
      double wp_dist = VAL(ego.waypoint_dist());
      if (wp_dist < waypoint_tol) {
        VAL(wp)[0] = scenario_width * 100;
        VAL(wp)[1] = VAL(ego.p)[1];
      }

      int2 new_cell = { (int)(VAL(ego.p)[0] / cell_width), (int)(VAL(ego.p)[1] / cell_width) };

      ego.cell = new_cell;
    }

    if (print_trace) {
      fprintf(stderr, "%.6f", t_sim);
      for (auto &ego : agents)
        fprintf(stderr, ",%d,%.6f,%.6f", true, VAL(ego.p)[0], VAL(ego.p)[1]);
      fprintf(stderr, "\n");
    }
  }

  #if RETURN_Y_POS == true
  adouble r = 0;
  for (auto &ego : agents) {
    r += ego.p[0];
  }
  r /= num_agents;

  #if CALIBRATION == true
  adouble diff = r - ref_r;
  return diff * diff;
  #else
  return r;
  #endif
  #endif

  #if RETURN_Y_POS == false
  for (auto &ego : agents) {
    if (ego.p[0] > door_y) {
      total_num_measurement_evacs++;
    }
  }

  // printf("returning %d evacs\n", total_num_measurement_evacs);

  #if CALIBRATION == true
  adouble diff = total_num_measurement_evacs - ref_total_num_measurement_evacs;
  return diff * diff;
  #else
  return total_num_measurement_evacs;
  #endif
  #endif
}

int main(int argc, char **argv)
{
  DiscoGrad<num_inputs> dg(argc, argv, false);
  DiscoGradFunc<num_inputs> func(dg, _DiscoGrad_crowd);

  dg.estimate(func);

  return 0;
}
