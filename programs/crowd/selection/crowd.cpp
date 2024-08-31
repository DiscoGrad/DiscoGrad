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

#ifndef NUM_BINS
#define NUM_BINS 20
#endif

#ifndef NUM_AGENTS
#define NUM_AGENTS 1000
#endif

#ifndef LEAPFROG
#define LEAPFROG true
#endif

#ifndef PRINT_TRACE
#define PRINT_TRACE false
#endif

const int num_agents = NUM_AGENTS;
const int num_emp_dist_bins = NUM_BINS;
const int num_inputs = num_emp_dist_bins;

#include "backend/discograd.hpp"

using namespace std;

constexpr int32_t cceil(double num) { return (int)num == num ?  num : (int)num + 1; }

const int num_reps = NUM_REPS;

const double warm_up_time = 100.0;

constexpr double scenario_width = 30;
constexpr double interaction_radius = 3.0;
constexpr double cell_width = interaction_radius;
constexpr int grid_width = cceil(scenario_width / cell_width);
const double delta_t = 0.1;

const double evac_time_hist_min = 10;
const double evac_time_hist_max = 75;
const int num_evac_time_bins = 20;
const double evac_time_bin_width = (evac_time_hist_max - evac_time_hist_min) / num_evac_time_bins;
double evac_time_hist[num_evac_time_bins];

double params_ref[num_emp_dist_bins] = {
0.0006,
0.0011,
0.0017,
0.0022,
0.0028,
0.0034,
0.0039,
0.0045,
0.0112,
0.0337,
0.1122,
0.2015,
0.1726,
0.1439,
0.1152,
0.0863,
0.0576,
0.0287,
0.0112,
0.0056,
};

// generated using seed 1, 100 reps
double evac_time_hist_ref[num_evac_time_bins] = {
0.0001,
0.0605,
0.2438,
0.1975,
0.1092,
0.0637,
0.0398,
0.0314,
0.0240,
0.0216,
0.0160,
0.0163,
0.0142,
0.0145,
0.0135,
0.0127,
0.0109,
0.0105,
0.0092,
0.0907,
};

const bool print_trace = PRINT_TRACE;

typedef int aid_t;
typedef vec2<double> dbl2;
typedef vec2<int> int2;

const double v_desired_mean = 1.29;
const double v_desired_stddev = 0.5;
const double min_v_desired = 0.25;

const double lambda = 2.0;
const double gamma_ = 0.35;
const double n = 2;
const double n_prime = 3;

const double sigma = 0.8;

const double w_internal = 1;
const double w_interaction = 15;
const double w_obstacles = 2;

const double waypoint_tol = 1.0; // in this radius the wp is considered to be reached
const double congestion_radius = 5; // agents in this radius are counted as part of the congestion around the waypoint
const double agent_radius = 0.4;

const double min_w_distance = 0.1;
const double max_w_distance = 1;

const int select_waypoint_period = 150;

double sum_evac_time;
int num_evac;

const int max_steps = (warm_up_time + evac_time_hist_max) / delta_t;

double t_sim = 0;

const int spawn_period = 2;

uniform_real_distribution<double> u_dist;
normal_distribution<double> v_desired_dist(v_desired_mean, v_desired_stddev);

const double door_width_m = 3;
const double door_width = door_width_m / scenario_width;
const double door_offset_front = 0.1;
const double door_offset_sides = 0.6;
const double wall_margin_y = 0.15;
const double wall_margin_x = 0.15;
const double door_wp_offset_m = 4;
const double door_wp_offset = door_wp_offset_m / scenario_width;

uniform_real_distribution<double> spawn_p_dist(wall_margin_x * scenario_width + agent_radius, (1 - wall_margin_x) * scenario_width - agent_radius);

pair<dbl2, dbl2> obstacles[] = {
  {{(1 - wall_margin_y) * scenario_width, wall_margin_x * scenario_width}, {(1 - wall_margin_y) * scenario_width, (wall_margin_x + door_offset_front) * scenario_width}},
  {{(1 - wall_margin_y) * scenario_width, (wall_margin_x + door_offset_front + door_width) * scenario_width}, {(1 - wall_margin_y) * scenario_width, (1 - wall_margin_x - door_offset_front - door_width) * scenario_width}},
  {{(1 - wall_margin_y) * scenario_width, (1 - wall_margin_x - door_offset_front) * scenario_width}, {(1 - wall_margin_y) * scenario_width, (1 - wall_margin_x) * scenario_width}},

  {{0, wall_margin_x * scenario_width}, {(1 - wall_margin_y - door_offset_sides) * scenario_width, wall_margin_x * scenario_width}},
  {{(1 - wall_margin_y - door_offset_sides + door_width) * scenario_width, wall_margin_x * scenario_width}, {(1 - wall_margin_y) * scenario_width, wall_margin_x * scenario_width}},

  {{0, (1 - wall_margin_x) * scenario_width}, {(1 - wall_margin_y - door_offset_sides) * scenario_width, (1 - wall_margin_x) * scenario_width}},
  {{(1 - wall_margin_y - door_offset_sides + door_width) * scenario_width, (1 - wall_margin_x) * scenario_width}, {(1 - wall_margin_y) * scenario_width, (1 - wall_margin_x) * scenario_width}}
};
size_t num_obs = sizeof(obstacles) / sizeof(pair<dbl2, dbl2>);

// y, x coordinates of waypoints and the expected center of congestion
dbl2 wp_alternatives[][2] = {
  {{(1 - wall_margin_y + door_wp_offset) * scenario_width, (wall_margin_x + door_offset_front + door_width / 2) * scenario_width},
   {(1 - wall_margin_y - door_wp_offset) * scenario_width, (wall_margin_x + door_offset_front + door_width / 2) * scenario_width}},

  {{(1 - wall_margin_y + door_wp_offset) * scenario_width, (1 - wall_margin_x - door_offset_front - door_width / 2) * scenario_width},
   {(1 - wall_margin_y - door_wp_offset) * scenario_width, (1 - wall_margin_x - door_offset_front - door_width / 2) * scenario_width}},

  {{(1 - wall_margin_y - door_offset_sides + door_width / 2) * scenario_width, (wall_margin_x - door_wp_offset) * scenario_width},
   {(1 - wall_margin_y - door_offset_sides + door_width / 2) * scenario_width, (wall_margin_x + door_wp_offset) * scenario_width}},

  {{(1 - wall_margin_y - door_offset_sides + door_width / 2) * scenario_width, (1 - wall_margin_x + door_wp_offset) * scenario_width},
   {(1 - wall_margin_y - door_offset_sides + door_width / 2) * scenario_width, (1 - wall_margin_x - door_wp_offset) * scenario_width}}
};
constexpr size_t num_wp_alternatives = sizeof(wp_alternatives) / sizeof(dbl2) / 2;

array<int, num_wp_alternatives> num_agents_near_wp, num_agents_near_wp_new;

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
      adouble v = sqrt(p[bin] * p[bin]);
      p_sum += v;
      hist[bin] = p_sum;
    }

    //printf("p_sum is %.2f\n", p_sum.val);
    for (int bin = 0; bin < num_bins; bin++) {
      adouble v = sqrt(p[bin] * p[bin]);
      printf("input bin %d: %.4f, ref: %.4f\n", bin, v.val / p_sum.val, params_ref[bin]);
      hist[bin] /= p_sum;
    }
  }

  double _DiscoGrad_draw(DiscoGrad<num_inputs>& _discograd) {
    double u = u_dist(_discograd.rng);

    int res_bin = -1;

    // linear search:
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

double norm(const dbl2& vec) {
  dbl2 s = vec * vec;

  return sqrt(s[0] + s[1]);
}

double angle(dbl2& a, dbl2& b) {
  double angle_a = atan2(a[1], a[0]);
  double angle_b = atan2(b[1], b[0]);

  double angle_diff = angle_b - angle_a;

  if (angle_diff > M_PI) angle_diff -= 2 * M_PI;
  else if (angle_diff <= -M_PI) angle_diff += 2 * M_PI;

  return angle_diff;
}

double dot(dbl2& a, dbl2&b) {
  return a[0] * b[0] + a[1] * b[1];
}

dbl2 compute_closest_point(pair<dbl2, dbl2>& o, dbl2& p) {
  dbl2 b_a_dist = o.second - o.first;
  dbl2 p_a_dist = p - o.first;

  double lambda = dot(p_a_dist, b_a_dist) / (b_a_dist[0] * b_a_dist[0] + b_a_dist[1] * b_a_dist[1]);

  if (lambda <= 0)
    return o.first;
  else if (lambda >= 1)
    return o.second;
  return o.first + lambda * b_a_dist;
}

dbl2 compute_obstacle_force(dbl2& p) {
  dbl2 min_dist;
  double min_norm = DBL_MAX;
  for (int oid = 0; oid < num_obs; oid++) {
    dbl2 closest_point = compute_closest_point(obstacles[oid], p);
    dbl2 dist = p - closest_point;
    double dist_norm = norm(dist);
    if (dist_norm < min_norm) {
      min_norm = dist_norm;
      min_dist = dist;
    }
  }

  double dist = min_norm - agent_radius;
  return exp(-dist / sigma) * (min_dist / min_norm);
}

struct Grid {
  unordered_set<aid_t> grid[grid_width][grid_width];
  unordered_set<aid_t>& operator[](int2& cell) { return grid[cell[0]][cell[1]]; }

  void clear() {
    for (int y = 0; y < grid_width; y++)
      for (int x = 0; x < grid_width; x++)
        grid[y][x].clear();
  }

  vector<unordered_set<aid_t> *> get_neighbor_cells(int2& c) {
    vector<unordered_set<aid_t> *> r;
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        if (c[0] + dy < 0 || c[0] + dy >= grid_width ||
            c[1] + dx < 0 || c[1] + dx >= grid_width)
          continue;
        r.push_back(&grid[c[0] + dy][c[1] + dx]);
      }
    }
    return r;
  }
};

Grid grid;

struct Agent {
  int aid;
  dbl2 p;
  dbl2 v;

  dbl2 a_old;

  bool active = false;
  vector<dbl2> waypoints;
  dbl2 prev_wp = { -1, -1 };

  size_t curr_waypoint = 0;

  int2 cell = {-1, -1};

  double t_spawn;

  double v_desired;

  int select_waypoint_timer;
  double w_distance;

  Agent() {}

  Agent(DiscoGrad<num_inputs>& _discograd, double w_distance) : w_distance(w_distance) {
    t_spawn = t_sim;
    v_desired = max(min_v_desired, v_desired_dist(_discograd.rng));
    select_waypoint_timer = int(u_dist(_discograd.rng) * select_waypoint_period);
    p = {0, spawn_p_dist(_discograd.rng)};
  }

  void _DiscoGrad_init_waypoints(DiscoGrad<num_inputs>& _discograd) {
    curr_waypoint = 0;
    waypoints.clear();
    waypoints.push_back(wp_alternatives[1][0]);
  }

  double waypoint_dist() {
    dbl2& wp = waypoints[curr_waypoint];
    if (wp[0] == -1)
      return abs(p[1] - wp[1]);
    if (wp[1] == -1)
      return abs(p[0] - wp[0]);
    
    dbl2 dist = p - waypoints[curr_waypoint];
    return norm(dist);
  }

  bool overlaps_with_obstacle(dbl2& p) {
    for (size_t oid = 0; oid < num_obs; oid++) {
      auto &o = obstacles[oid];
      dbl2 dist = compute_closest_point(o, p) - p;
      if (norm(dist) < agent_radius)
        return true;
    }
    return false;
  }


  void move(DiscoGrad<num_inputs>& _discograd, dbl2 &a) {

#if LEAPFROG == true
      dbl2 p_new = p + v * delta_t + 0.5 * a_old * delta_t * delta_t;
      dbl2 v_new = v + 0.5 * (a_old + a) * delta_t;
#else // ballistic update as described in Kesting and Treiber, Traffic Flow Dynamics: Data, Models and Simulation, 2013.
      dbl2 v_new = v + a * delta_t;
      dbl2 p_new = p + (v + v_new) / 2 * delta_t;
#endif

    for (int d = 0; d < 2; d++)
      p_new[d] = max(0.0, min(nextafter(scenario_width, 0), p_new[d]));

    assert(!overlaps_with_obstacle(p));

    if (overlaps_with_obstacle(p_new)) {
      dbl2 cand_p_new[3] = { { p_new[0], p[1] }, { p[0], p_new[1] }, { p[0] - v[0] * delta_t + 0.1 * u_dist(_discograd.rng), p[1] - v[1] * delta_t + 0.1 * u_dist(_discograd.rng)} };
      bool overlap = true;
      for (int i = 0; i < 3; i++) {
        if (!overlaps_with_obstacle(cand_p_new[i])) {
          overlap = false;
          p_new = cand_p_new[i];
          break;
        }
      }
      if (overlap) {
        return;
      }
    }

    v = v_new;
    p = p_new;
    
    // to handle agents that are pushed outside the building nowhere near their target exit
    bool outside_building = p[1] < wall_margin_x * scenario_width || p[1] > (1.0 - wall_margin_x) * scenario_width || p[0] > (1.0 - wall_margin_y) * scenario_width;

    if (waypoint_dist() < waypoint_tol || (curr_waypoint == 0 && outside_building)) {
      if (curr_waypoint == 1) {

        if (t_sim > warm_up_time) {
          double evac_time = t_sim - t_spawn;

          for (int bin = 0; bin < num_evac_time_bins; bin++) {
            if (evac_time < evac_time_hist_min + bin * evac_time_bin_width || bin == num_evac_time_bins - 1) {
              evac_time_hist[bin]++;
              break;
            }
          }
          sum_evac_time += t_sim - t_spawn;
          num_evac++;
        }
        grid[cell].erase(aid);
        active = false;
        return;
      }

      double y_dist = scenario_width - p[0];
      double x_left_dist = p[1];
      double x_right_dist = scenario_width - p[1];

      if (y_dist < x_left_dist && y_dist < x_right_dist) {
        waypoints.push_back({scenario_width, -1});
      } else if (x_left_dist < x_right_dist) {
        waypoints.push_back({-1, 0});
      } else {
        waypoints.push_back({-1, scenario_width});
      }
      curr_waypoint++;
    }

    int2 new_cell = { int(p[0] / cell_width), (int)(p[1] / cell_width) };

    for (int d = 0; d < 2; d++)
      assert(new_cell[d] >= 0 && new_cell[d] < grid_width);

    if (new_cell != cell) {
      if (cell[0] >= 0)
        grid[cell].erase(aid);
      grid[new_cell].insert(aid);
    }

    cell = new_cell;
  }

  bool operator==(Agent& other) { return aid == other.aid; }
};

Agent agents[num_agents];
int num_active_agents = 0;

void spawn_agent(DiscoGrad<num_inputs>& _discograd, EmpDist<num_emp_dist_bins>& w_distance_dist) {
  if (num_active_agents == num_agents)
    return;
  auto& ego = agents[num_active_agents];

  double w_distance = w_distance_dist._DiscoGrad_draw(_discograd);
  ego = Agent(_discograd, w_distance);
  ego._DiscoGrad_init_waypoints(_discograd);

  int min_cost_wpid;
  double min_cost = DBL_MAX;
  for (int wpid = 0; wpid < num_wp_alternatives; wpid++) {

    double cost = ego.w_distance * norm(ego.p - wp_alternatives[wpid][0]) +
             (1.0 - ego.w_distance) * (num_agents_near_wp[wpid] + 1);

    if (cost < min_cost) {
      min_cost = cost;
      min_cost_wpid = wpid;
    }
    //printf("%d agents at wp %d, dist is %.2f, cost is %.2f, min_cost is %.2f, w_distance is %.2f\n", num_agents_near_wp[wpid], wpid, norm(p - wp_alternatives[wpid][0]), cost.val, min_cost.val, w_distance.val);
    assert(cost >= 0.0);
  }

  ego.waypoints[ego.curr_waypoint] = (wp_alternatives[min_cost_wpid][0]);

  ego.aid = num_active_agents;
  ego.v = {0.0, 0.0};
  ego.a_old = {0.0, 0.0};
  ego.active = true;
  dbl2 a({0, 0});
  ego.move(_discograd, a);

  num_active_agents++;
}

dbl2 left_normal(dbl2& vec) { return {-vec[1], vec[0]}; }

adouble _DiscoGrad_crowd(DiscoGrad<num_inputs> &_discograd, aparams &p)
{
  EmpDist<num_emp_dist_bins> w_distance_dist(min_w_distance, max_w_distance, &p[0]);

  if (print_trace) {
    fprintf(stderr, "width %.4f\n", scenario_width);
    for (size_t oid = 0; oid < num_obs; oid++) {
      auto& o = obstacles[oid];
      if (print_trace)
        fprintf(stderr, "obstacle %.4f, %.4f; %4f, %.4f\n", o.first[0], o.first[1], o.second[0], o.second[1]);
    }
    fprintf(stderr, "waypoint tol %.4f\n", waypoint_tol);
    fprintf(stderr, "congestion radius %.4f\n", congestion_radius);
    for (auto& wp : wp_alternatives) {
      fprintf(stderr, "waypoint %.4f, %.4f\n", wp[0][0], wp[0][1]);
      fprintf(stderr, "congestion point %.4f, %.4f\n", wp[1][0], wp[1][1]);
    }
    fprintf(stderr, "t");
  }

  for (int rep = 0; rep < num_reps; rep++) {
    num_active_agents = 0;
    sum_evac_time = 0;
    num_evac = 0;

    grid.clear();
  
    for (int aid = 0; aid < num_agents; aid++) {
      agents[aid].aid = aid;
      agents[aid].active = false;
      if (print_trace)
        fprintf(stderr, ",a%d.active,a%d.y,a%d.x", aid, aid, aid);
    }
    if (print_trace)
      fprintf(stderr, "\n");

    //bool active_agents_left = true;
    for (int step = 0; /* active_agents_left && */ step < max_steps; step++) {
      //printf("step %d, active agents: %d\n", step, num_active_agents);
      //active_agents_left = false;

      t_sim = step * delta_t;

      if (step % spawn_period == 0)
        spawn_agent(_discograd, w_distance_dist);

      for (auto &ego : agents) {
        if (!ego.active)
          continue;

       // active_agents_left = true;

        dbl2 f_internal = {0, 0};
        dbl2 wp = ego.waypoints[ego.curr_waypoint];

        for (int d = 0; d < 2; d++)
          if (wp[d] == -1)
            wp[d] = ego.p[d];

        dbl2 t_dist = wp - ego.p;
        double t_dist_norm = norm(t_dist);
        dbl2 e = t_dist / t_dist_norm;
        f_internal = ego.v_desired * e - ego.v;

        auto nb_cells = grid.get_neighbor_cells(ego.cell);

        dbl2 f_interaction = {0, 0};
        for (auto &cell : nb_cells) {
          for (aid_t other_aid : *cell) {
            auto &other = agents[other_aid];
            if (other == ego) continue;

            dbl2 o_dist = other.p - ego.p;

            double o_dist_norm = norm(o_dist);
            dbl2 o_dir = o_dist / o_dist_norm;

            dbl2 v_diff = ego.v - other.v;

            dbl2 int_v = lambda * v_diff + o_dir;

            double int_norm = norm(int_v);
            dbl2 int_dir = int_v / int_norm;

            double theta = angle(int_dir, o_dir);
            int theta_sign = theta ? theta / abs(theta) : 0;

            double B = gamma_ * int_norm;

            double npBt = n_prime * B * theta;
            double nBt = n * B * theta;
            dbl2 force_v = -exp(-o_dist_norm / B - npBt * npBt) * int_dir;
            dbl2 force_angle = -theta_sign * exp(-o_dist_norm / B - nBt * nBt) * left_normal(int_dir);

            f_interaction += force_v + force_angle;
          }
        }

        dbl2 f_obstacles = compute_obstacle_force(ego.p);

        dbl2 a = f_internal * w_internal +
                 f_interaction * w_interaction +
                 f_obstacles * w_obstacles;

        ego.move(_discograd, a);

        for (int wpid = 0; wpid < num_wp_alternatives; wpid++) {
          if (norm(ego.p - wp_alternatives[wpid][1]) < congestion_radius)
            num_agents_near_wp_new[wpid]++;
        }

        if (--ego.select_waypoint_timer == 0 && ego.curr_waypoint == 0) {
          int min_cost_wpid;
          double min_cost = DBL_MAX;
          for (int wpid = 0; wpid < num_wp_alternatives; wpid++) {

            double cost = ego.w_distance * norm(ego.p - wp_alternatives[wpid][0]) +
                     (1.0 - ego.w_distance) * (num_agents_near_wp[wpid] + 1);

            if (cost < min_cost) {
              min_cost = cost;
              min_cost_wpid = wpid;
            }
            //printf("%d agents at wp %d, dist is %.2f, cost is %.2f, min_cost is %.2f, w_distance is %.2f\n", num_agents_near_wp[wpid], wpid, norm(p - wp_alternatives[wpid][0]), cost.val, min_cost.val, w_distance.val);
            assert(cost >= 0.0);
          }

          if ((wp_alternatives[min_cost_wpid][0]) != ego.prev_wp) {
            ego.prev_wp = ego.waypoints[ego.curr_waypoint];
            ego.waypoints[ego.curr_waypoint] = (wp_alternatives[min_cost_wpid][0]);
          }
          ego.select_waypoint_timer = select_waypoint_period;
        }
      }

      for (int wpid = 0; wpid < num_wp_alternatives; wpid++) {
        num_agents_near_wp[wpid] = num_agents_near_wp_new[wpid];
        num_agents_near_wp_new[wpid] = 0;
      }
 
      if (print_trace) {
        fprintf(stderr, "%.6f", t_sim);
        for (auto &ego : agents)
          fprintf(stderr, ",%d,%.6f,%.6f", ego.active,ego.p[0], ego.p[1]);
        fprintf(stderr, "\n");
      }
    }

    //double avg_evac_time = sum_evac_time / num_evac;
    //printf("%d total evacs, avg. time %.2f\n", num_evac, avg_evac_time);

    //for (int bin = 0; bin < num_evac_time_bins; bin++) {
    //  evac_time_hist[bin] /= num_evac;
    //}
  }

  double evac_time_freq_sum = 0;
  for (int bin = 0; bin < num_evac_time_bins; bin++) {
    evac_time_freq_sum += evac_time_hist[bin];
  }

  for (int bin = 0; bin < num_evac_time_bins; bin++) {
    evac_time_hist[bin] /= evac_time_freq_sum;
    printf("output bin %d: %.4f, ref: %.4f\n", bin, evac_time_hist[bin], evac_time_hist_ref[bin]);
  }

  // earth movers distance = Wasserstein distance
  array<double, num_evac_time_bins + 1> dists = {};
  for (int bin = 0; bin < num_evac_time_bins; bin++) {
    dists[bin + 1] = evac_time_hist_ref[bin] - evac_time_hist[bin] + dists[bin];
  }

  double dist = 0;
  for (int bin = 0; bin < num_evac_time_bins; bin++)
    dist += abs(dists[bin]);
  dist /= num_evac_time_bins;
  
  return dist;
}

int main(int argc, char **argv)
{
  DiscoGrad<num_inputs> dg(argc, argv, false);
  DiscoGradFunc<num_inputs> func(dg, _DiscoGrad_crowd);

  dg.estimate(func);

  return 0;
}
