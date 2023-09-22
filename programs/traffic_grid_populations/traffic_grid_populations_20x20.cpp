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

#include <deque>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <array>
#include <float.h>
#include <random>
#include <iostream>
#include <fstream>

const int grid_width = 20;
const int num_inputs = grid_width * grid_width;

#include "backend/discograd.hpp"

using namespace std;

const bool print_trace = false;

const int num_steps = grid_width;
const double time_step = 1.0;

const double turn_prob = 0.05;
const double signal_period = 4.0;

const int num_arrivals_per_step = grid_width;

unsigned int seed; // initialized in main

inline int pos_mod(int i, int n) { return (i % n + n) % n; }

enum TurnDirection { turn_left, turn_none, turn_right };

uniform_real_distribution<double> u_dist;

int draw_uniform_in_range(DiscoGrad<num_inputs> &_discograd, int lower, int upper) {
  return uniform_int_distribution<int>(lower, upper)(_discograd.rng);
}

enum LaneDirection { north, east, south, west };

struct Lane {
  sdouble num_dvus;
  vector<TurnDirection> turns;
};

// top left part of an intersection
class Intersection { 
  public:
  sdouble signal_offset;
  Lane lanes[4];
  Lane& operator[](int lidx) { return lanes[lidx]; }

  Intersection& operator=(Intersection &other) {
    for (int dir = 0; dir < 4; dir++)
      lanes[dir] = other.lanes[dir];
    return *this;
  }
};

void get_next_lane(Intersection (&grid)[grid_width][grid_width],
                   int y, int x, LaneDirection dir, TurnDirection turn_dir,
                   Intersection **out_is, Lane **out_lane, Intersection **out_signal_is = NULL) {
  int next_y = y, next_x = x;
  switch (dir) {
    case north:
      switch (turn_dir) {
        case turn_none: next_y--; break;
        case turn_left: next_y--; break;
        case turn_right: next_y--; next_x++; break;
      }
      break;
    case south:
      switch (turn_dir) {
        case turn_none: next_y++; break;
        case turn_left: next_x++; break;
        case turn_right: break;
      }
      break;
    case west:
      switch (turn_dir) {
        case turn_none: next_x--; break;
        case turn_left: next_y++; next_x--; break;
        case turn_right: next_x--; break;
      }
      break;
    case east:
      switch (turn_dir) {
        case turn_none: next_x++; break;
        case turn_left: break;
        case turn_right: next_y++; break;
      }
      break;
  }

  //if (next_x < 0 || next_x >= grid_width || next_y < 0 || next_y >= grid_width) {
  //  *out_is = NULL;
  //  *out_lane = NULL;
  //  return;
  //}

  LaneDirection next_dir = LaneDirection(pos_mod((int)dir + ((int)turn_dir - 1), 4));
  Intersection& next_is = grid[pos_mod(next_y, grid_width)][pos_mod(next_x, grid_width)];

  int y_rmd = pos_mod(next_y, grid_width);
  int x_rmd = pos_mod(next_x, grid_width);

  *out_is = &grid[y_rmd][x_rmd];
  *out_lane = &grid[y_rmd][x_rmd][next_dir];

  if (out_signal_is) {
    *out_signal_is = &next_is;
    if (dir == south || dir == east)
      *out_signal_is = &grid[y][x];
  }
}

TurnDirection draw_turn_dir(DiscoGrad<num_inputs> &_discograd)
{
  double r = u_dist(_discograd.rng);

  if (r < turn_prob / 2)
    return turn_left;
  if (r < turn_prob)
    return turn_right;

  return turn_none;
}

void print_grid(int step, double t_in_signal_period, Intersection (&grid)[grid_width][grid_width]) {
  for (int y = 0; y < grid_width; y++) {
    for (int x = 0; x < grid_width; x++) {
      double is_dvus = 0;
      for (int dir = 0; dir < 4; dir++)
        is_dvus += grid[y][x][dir].num_dvus.expectation().val;
      printf("%.2f,", is_dvus);
    }
  }
  printf("\n");
}

void create_random_dvu(DiscoGrad<num_inputs> &_discograd, Intersection (&w_grid)[grid_width][grid_width])
{
  int dir = draw_uniform_in_range(_discograd, 1, 2); // init_dir_dist(sim_rng);
  int coord_r = draw_uniform_in_range(_discograd, 0, grid_width - 1); //init_is_coord_dist(sim_rng);

  //printf("dir is %d, coord_r is %d\n", dir, coord_r);

  int is_y, is_x;
  if (dir == LaneDirection::north) {
    is_y = grid_width - 1;
    is_x = coord_r;
  } else if (dir == LaneDirection::east) {
    is_y = coord_r;
    is_x = 0;
  } else if (dir == LaneDirection::south) {
    is_y = 0;
    is_x = coord_r;
  } else if (dir == LaneDirection::west) {
    is_y = coord_r;
    is_x = grid_width - 1;
  } else {
    assert(false);
  }

  // printf("creating a dvu at %d/%d,%d\n", is_y, is_x, dir);

  w_grid[is_y][is_x][dir].num_dvus += 1;
}

void copy_grid(Intersection (&dst)[grid_width][grid_width], Intersection (&src)[grid_width][grid_width]) {
  for (int y = 0; y < grid_width; y++) {
    for (int x = 0; x < grid_width; x++) {
      dst[y][x] = src[y][x];
    }
  }
}

adouble _DiscoGrad_simulate(DiscoGrad<num_inputs> &_discograd, aparams &params)
{
  
  sdouble roads_passed = 0;

  if (print_trace && true) {
    for (int is_y = 0; is_y < grid_width; is_y++)
      for (int is_x = 0; is_x < grid_width; is_x++)
        printf("is_%d_%d,", is_y, is_x);
    printf("\n");
  }

  static Intersection r_grid[grid_width][grid_width], w_grid[grid_width][grid_width];

  for (int is_y = 0; is_y < grid_width; is_y++) {
    for (int is_x = 0; is_x < grid_width; is_x++) {
      for (int dir = 0; dir < 4; dir++) {
        w_grid[is_y][is_x][dir].num_dvus = r_grid[is_y][is_x][dir].num_dvus = 0;
        w_grid[is_y][is_x][dir].turns.clear();
        for (int i = 0; i < num_steps; i++)
          w_grid[is_y][is_x][dir].turns.push_back(draw_turn_dir(_discograd));
        r_grid[is_y][is_x][dir].turns = w_grid[is_y][is_x][dir].turns;
      }
    }
  }

  for (int y = 0; y < grid_width; y++) {
    for (int x = 0; x < grid_width; x++) {
      w_grid[y][x].signal_offset = r_grid[y][x].signal_offset = sdouble({params[y * grid_width + x], _discograd.get_variance()});
    }
  }

  double t_in_signal_period = 0;
  for (int step = 0; step < num_steps; step++) {

    for (int i = 0; i < num_arrivals_per_step; i++)
      create_random_dvu(_discograd, w_grid);
    // printf("step %d, time %.2f\n", step, step * time_step);

    for (int y = 0; y < grid_width; y++) {

      for (int x = 0; x < grid_width; x++) {
        for (int dir = 0; dir < 4; dir++) {

          auto& r_lane = r_grid[y][x][dir];
          auto& w_lane = w_grid[y][x][dir];

          //if (r_lane.num_dvus > 0.5 && true) {
          {
            // printf("number of dvus on %d/%d,%d: %.2f\n", y, x, dir, r_lane.num_dvus.expectation().get_val());
            sdouble horizontal_red = 0;
            Intersection *signal_is;

            Intersection *next_is;
            Lane *next_lane;
            get_next_lane(w_grid, y, x, (LaneDirection)dir, w_lane.turns.back(), &next_is, &next_lane, &signal_is);
            w_lane.turns.pop_back();

            //sdouble &signal_offset = signal_is->signal_offset;

            sdouble &horiz_green_lower = signal_is->signal_offset;
            sdouble horiz_green_upper = signal_is->signal_offset + signal_period / 2;

            if (horiz_green_upper <= t_in_signal_period)
              horizontal_red = 1;
            if (horiz_green_lower > t_in_signal_period)
              horizontal_red = 1;

            sdouble red = 0;
            if ((horizontal_red > 0.5 && (dir == east || dir == west)) ||
                (horizontal_red < 0.5 && (dir == north || dir == south))) {
              red = 1;
            }

            //printf("green signal from %.2f to %.2f, t_in_signal_period: %.2f, red: %.2f\n", horiz_green_lower.expectation().val, horiz_green_upper.expectation().val,  t_in_signal_period, red.expectation().val);

            if (r_lane.num_dvus > 0.5 && red < 0.5 && true) {
              
              next_lane->num_dvus += 1;

              w_lane.num_dvus -= 1;

              roads_passed += 1.0;

              // printf("a dvu exited %d/%d in direction %d\n", y, x, dir);
            }
          }
        }
      }
    }

    copy_grid(r_grid, w_grid);

    if (print_trace && true)
      print_grid(step, t_in_signal_period, r_grid);

    t_in_signal_period += time_step;
    if (t_in_signal_period >= signal_period && true)
      t_in_signal_period -= signal_period;
  }
  

  return -roads_passed.expectation();
}

int main(int argc, char **argv)
{
  DiscoGrad<num_inputs> dg(argc, argv); //, true);
  DiscoGradFunc<num_inputs> func(_DiscoGrad_simulate);

  dg.estimate(func);

  return 0;
}
