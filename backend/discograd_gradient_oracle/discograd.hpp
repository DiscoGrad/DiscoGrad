/** Backend library of the DiscoGrad Gradient Oracle (DGO).
 *  Contains classes and functions for enabling a smooth interpretation of C++ programs.
 * 
 *  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
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

#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <cstdint>
#include <deque>
#include <fenv.h>
#include <math.h>
#include <random>
#include <stack>
#include <stdio.h>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <queue>
#include <unistd.h>
#include <memory>
#include "kde.hpp"

using namespace std;

template<int num_inputs>
class DiscoGradBase;

// setting this enables smoothing. otherwise, the program is executed in its crisp form either with or without AD (see the bottom of this file)
#ifdef ENABLE_SI

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {
private:
  // tuning parameters
  double kde_delta = 65000; // branch conditions outside of [-kde_delta, kde_delta] are ignored entirely
  double adj_carrier_delta = 1e15; // branch conditions outside of [-adj_carrier_delta * stddev, adj_carrier_delta * stddev] do not carry weight adjoints
  uint64_t max_num_branches = 1e15; // maximum number of branches to collect data for, to limit memory consumption

  class branch_data_ {
  public:
    shared_ptr<array<double, num_inputs>> forward_weight_adjoint, backward_weight_adjoint;
    uint64_t num_adj_carriers_true = 0, num_adj_carriers_false = 0;
    vector<__fp16> branch_conditions;
    adouble branch_conditions_sum;

    branch_data_() : forward_weight_adjoint(nullptr), backward_weight_adjoint(nullptr), num_adj_carriers_true(0), num_adj_carriers_false(0) { };
  };

  unordered_map<uint64_t, branch_data_> all_branch_data;

  int num_samples;
  int pass;

  normal_distribution<> normal_dist;

  vector<double> ys;

  vector<vector<bool>> cond_lt_zero;

  vector<array<double, num_inputs>> deltaxs, dydxs, jump_adjoints;
  double ym = 0;
  array<double, num_inputs> dymdxm;

  double stddev;

  array<double, num_inputs> curr_sample_jump_adjoint;
  array<uint64_t, num_inputs> curr_sample_adj_carrier_diff;

  default_random_engine sampling_rng;

  const uint64_t initial_global_branch_id = 0;
  
  uint64_t global_branch_id = initial_global_branch_id;

  int sample_id;
  
public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {
    
    string path(argv[0]);
    args::ArgParser parser("Usage: " + path + " -s [seed = 1] --nc [#parameter combinations = 1] --nr [#replications = 1] --var [variance = 1] --ns [#samples = 100]");

    parser.option("s");
    parser.option("nc");
    parser.option("nr");
    parser.option("var");
    parser.option("ns");

    parser.parse(argc, argv);

    if (parser.found("s"))
      this->seed_arg = stoi(parser.value("s"));

    if (parser.found("nc"))
      this->num_param_combs = stoi(parser.value("nc"));
    
    if (parser.found("nr"))
      this->num_replications = stoi(parser.value("nr"));

    stddev = 1;
    if (parser.found("var"))
      stddev = sqrt(stof(parser.value("var")));

    num_samples = 100;
    if (parser.found("ns"))
      num_samples = stoi(parser.value("ns"));

    cond_lt_zero.resize(num_samples);

    if (debug) {
      printf("stddev: %.10g\n", stddev);
      printf("num_samples: %d\n", num_samples);
    }

    adj_carrier_delta *= stddev;

    normal_dist = normal_distribution<>(0, stddev);
  }

  /** Clean up the state containers to their initial (empty) state. */
  void clean_up() {
    global_branch_id = initial_global_branch_id;
    ys.clear();
    deltaxs.clear();
    dydxs.clear();
    jump_adjoints.clear();
    ym = 0.0;
    dymdxm.fill(0.0);
    cond_lt_zero.clear();
    cond_lt_zero.resize(num_samples);
    
    all_branch_data.clear();
  }

  double get_variance() const { return stddev * stddev; }

  void advance_global_branch_id(adouble& cond) {
    global_branch_id++;
  }

  void prepare_branch(adouble cond) {
    cond_lt_zero[sample_id].push_back(cond.val < 0);

    if (!cond.has_adj() || cond < -kde_delta || cond > kde_delta) {
      advance_global_branch_id(cond);
      return;
    }

    auto it = all_branch_data.find(global_branch_id);
    if (it == all_branch_data.end() && all_branch_data.size() >= max_num_branches) {
      advance_global_branch_id(cond);
      return;
    }

    branch_data_ *bd;

    if (it == all_branch_data.end()) {
      bd = &(all_branch_data.insert({global_branch_id, branch_data_()}).first->second);
    } else {
      bd = &(it->second);
    }

    bd->branch_conditions.push_back((float)cond.get_val());

    bool carries_adj = true;
    if (cond < 0 && cond > -adj_carrier_delta)
      bd->num_adj_carriers_true++;
    else if (cond >= 0 && cond < adj_carrier_delta)
      bd->num_adj_carriers_false++;
    else
      carries_adj = false;
   
    if (carries_adj) {
      if (bd->num_adj_carriers_true + bd->num_adj_carriers_false == 1) {
        bd->branch_conditions_sum = cond;
      } else {
        bd->branch_conditions_sum += cond;
      }
    }

    advance_global_branch_id(cond);
  }

  void pass_one(DiscoGradProgram<num_inputs> &program) {

    for (sample_id = 0; sample_id < num_samples; sample_id++) {
      array<adouble, num_inputs> pm_perturbed = this->parameters;
      array<double, num_inputs> perturbation;
      for (int dim = 0; dim < num_inputs; dim++) {
        perturbation[dim] = normal_dist(sampling_rng);
        pm_perturbed[dim] += perturbation[dim];
      }

      deltaxs.push_back(perturbation);

      global_branch_id = initial_global_branch_id;

      this->rng.seed(this->current_seed);
      adouble r = program.run(*this, pm_perturbed);

      ys.push_back(r.get_val());

      array<double, num_inputs> dydx;
      for (int dim = 0; dim < num_inputs; dim++)
        dydx[dim] = r.get_adj(dim);

      dydxs.push_back(dydx);
    }

    int num_branches_skipped = 0;
    for (auto &item : all_branch_data) {
      auto &bd = item.second;
      if (bd.num_adj_carriers_true < 1 || bd.num_adj_carriers_false < 1) {
        num_branches_skipped++;
        continue;
      }

      auto &cond_sum = bd.branch_conditions_sum;
      uint64_t adj_carriers = bd.num_adj_carriers_true + bd.num_adj_carriers_false;
      
      kdepp::Kde1d<__fp16, float> kde(bd.branch_conditions);

      assert(bd.forward_weight_adjoint == nullptr && bd.backward_weight_adjoint == nullptr);
      bd.forward_weight_adjoint = shared_ptr<array<double, num_inputs>>(new array<double, num_inputs>);
      bd.backward_weight_adjoint = shared_ptr<array<double, num_inputs>>(new array<double, num_inputs>);
      
      for (int dim = 0; dim < num_inputs; dim++) {
        double mean_cond_adj = cond_sum.get_adj(dim) / adj_carriers;
        if (mean_cond_adj == 0.0) {
          (*bd.forward_weight_adjoint)[dim] = (*bd.backward_weight_adjoint)[dim] = 0.0;
          continue;
        }
        double weight_adjoint = kde.eval(0) * mean_cond_adj;
        (*bd.forward_weight_adjoint)[dim] = -weight_adjoint * bd.branch_conditions.size() / bd.num_adj_carriers_true;
        (*bd.backward_weight_adjoint)[dim] = weight_adjoint * bd.branch_conditions.size() / bd.num_adj_carriers_false;

        if (this->debug) {
          printf("bd.forward_weight_adjoint[%d] = -%.2f * %ld / %ld == %.2f\n", dim, -weight_adjoint, bd.branch_conditions.size(), bd.num_adj_carriers_true, (*bd.forward_weight_adjoint)[dim]);
          printf("bd.backward_weight_adjoint[%d] = -%.2f * %ld / %ld == %.2f\n", dim, -weight_adjoint, bd.branch_conditions.size(), bd.num_adj_carriers_true, (*bd.backward_weight_adjoint)[dim]);
          printf("branch %lX, dim %d, integration results: %.2f, %.2f\n", item.first, dim, (*bd.forward_weight_adjoint)[dim], (*bd.backward_weight_adjoint)[dim]);
        }
      }
    }
    if (this->debug)
      printf("%ld branches total, %d skipped\n", all_branch_data.size(), num_branches_skipped);
  }

  void pass_two(DiscoGradProgram<num_inputs> &program) {
    for (sample_id = 0; sample_id < num_samples; sample_id++) {
      global_branch_id = initial_global_branch_id;

      curr_sample_jump_adjoint = {};

      for (int dim = 0; dim < num_inputs; dim++)
        curr_sample_adj_carrier_diff[dim] = std::numeric_limits<uint64_t>::max();

      for (size_t branch_id = 0; branch_id < cond_lt_zero[sample_id].size(); branch_id++) {
        bool curr_cond_lt_zero = cond_lt_zero[sample_id][branch_id];

        auto it = all_branch_data.find(branch_id);
        if (it == all_branch_data.end())
          continue;

        auto &bd = it->second;
        if (bd.num_adj_carriers_true < 1 || bd.num_adj_carriers_false < 1)
          continue;

        for (int dim = 0; dim < num_inputs; dim++) {
          uint64_t adj_carrier_diff = abs((int64_t)(bd.num_adj_carriers_true - bd.num_adj_carriers_false));
          if (curr_sample_jump_adjoint[dim] != 0.0 && adj_carrier_diff >= curr_sample_adj_carrier_diff[dim])
            continue;

          if (curr_cond_lt_zero && (*bd.forward_weight_adjoint)[dim] != 0.0) {
            if (this->debug)
              printf("sample %d, dim %d, branch %lX: setting fw weight adjoint of %.2f\n", sample_id, dim, global_branch_id, (*bd.forward_weight_adjoint)[dim]);
            curr_sample_jump_adjoint[dim] = (*bd.forward_weight_adjoint)[dim];
            curr_sample_adj_carrier_diff[dim] = adj_carrier_diff;
          } else if (!curr_cond_lt_zero && (*bd.backward_weight_adjoint)[dim] != 0.0) {
            if (this->debug)
              printf("sample %d, dim %d, branch %lX: setting bw weight adjoint of %.2f\n", sample_id, dim, global_branch_id, (*bd.backward_weight_adjoint)[dim]);
            curr_sample_jump_adjoint[dim] = (*bd.backward_weight_adjoint)[dim];
            curr_sample_adj_carrier_diff[dim] = adj_carrier_diff;
          }
        }

      }

      jump_adjoints.push_back(curr_sample_jump_adjoint);
    }

  }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->sampling_rng.seed(random_device()());

    double exp = 0.0;
    double der[num_inputs] = { 0.0 };
    for (int rep = 0; rep < this->num_replications; ++rep) {
      // determine seed for this replication
      this->current_seed = this->seed_dist(this->rep_seed_gen); 

      // run (sample) the program
      pass_one(program);
      pass_two(program);

      // collect derivative information
      ym = 0;
      dymdxm = {};
      for (int sample_id = 0; sample_id < num_samples; sample_id++) {
        for (int dim = 0; dim < num_inputs; dim++) {
          if (this->debug)
            printf("sample %d, dim %d, adding %.2f -> %.2f * %.2f + %.2f\n", sample_id, dim, this->parameters[dim].get_val() + deltaxs[sample_id][dim], jump_adjoints[sample_id][dim], ys[sample_id], dydxs[sample_id][dim]);
          dymdxm[dim] += jump_adjoints[sample_id][dim] * ys[sample_id] + dydxs[sample_id][dim];
        }
        exp += ys[sample_id];
      }

      // accumulate value and gradient
      for (int dim = 0; dim < num_inputs; dim++)
        der[dim] += dymdxm[dim];
      
      if (this->num_replications > 1)
        clean_up();
    }
    printf("exp %lf, nsamp %d, reps %d\n", exp, num_samples, this->num_replications);
    this->exp_val = (exp / num_samples) / this->num_replications;
    for (int dim = 0; dim < num_inputs; dim++) {
      this->exp_val.set_adj(dim, (der[dim] / num_samples) / this->num_replications);
    }
  }
};

#else

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {
public:
  default_random_engine sampling_rng;
  normal_distribution<> normal_dist;

  double stddev;
  int num_samples;

  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {
    string path(argv[0]);
    args::ArgParser parser("Usage: " + path + " -s [seed = 1] --nc [#parameter combinations = 1] --nr [#replications = 1] --var [variance = 1] --ns [#samples = 1]");

    parser.option("s");
    parser.option("nc");
    parser.option("nr");
    parser.option("var");
    parser.option("ns");

    parser.parse(argc, argv);

    if (parser.found("s"))
      this->seed_arg = stoi(parser.value("s"));

    if (parser.found("nc"))
      this->num_param_combs = stoi(parser.value("nc"));
    
    if (parser.found("nr"))
      this->num_replications = stoi(parser.value("nr"));

    stddev = 1;
    if (parser.found("var"))
      stddev = sqrt(stof(parser.value("var")));

    num_samples = 1;
    if (parser.found("ns"))
      num_samples = stoi(parser.value("ns"));

    if (debug) {
      printf("variance: %.10g\n", stddev * stddev);
      printf("num_samples: %d\n", num_samples);
    }
  }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->sampling_rng.seed(random_device()());
    array<adouble, num_inputs> pm_perturbed = this->parameters;
    for (int rep = 0; rep < this->num_replications; ++rep) {
      this->current_seed = this->seed_dist(this->rep_seed_gen);

      for (int sample = 0; sample < num_samples; sample++) {
        if (stddev > 0) {
          for (int dim = 0; dim < num_inputs; dim++)
            pm_perturbed[dim] = this->parameters[dim] + normal_dist(sampling_rng);
        }


        this->rng.seed(this->current_seed);

        this->exp_val += program.run(*this, pm_perturbed);
      }
    }
    this->exp_val = this->exp_val / this->num_replications / num_samples;
  }
};
#endif
