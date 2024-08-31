/** Backend library of the sampling-enabled version of DiscoGrad.
 *  Contains classes and functions for enabling a smooth interpretation of C++ programs.
 *
 * Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the “Software”), to deal in the Software without
 * restriction, including without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *  
 *   The above copyright notice and this permission notice shall be included in all copies or
 *   substantial portions of the Software.
 *   
 *   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 *   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 *   PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 *   ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 *   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE. */

#include "ankerl/unordered_dense.h"
#include "boolvector.hpp"

// use defaults if user doesn't override them
#ifndef DGO_NUM_BRANCH_COND
#define DGO_NUM_BRANCH_COND SIZE_MAX
#endif

// choose true/false samples by path similarity
#ifndef DGO_MIN_EXT_PERT
#define DGO_MIN_EXT_PERT true
#endif

#ifndef DGO_MIN_NUM_TANG_CARR
#define DGO_MIN_NUM_TANG_CARR ((size_t)1)
#endif

#ifndef DGO_MAX_NUM_TANG_CARR
#define DGO_MAX_NUM_TANG_CARR ((size_t)10)
#endif

#ifndef DGO_FORK_LIMIT
#define DGO_FORK_LIMIT 0
#endif

#ifndef DGO_MIN_COND_VARIANCE
#define DGO_MIN_COND_VARIANCE 0
#endif

#ifndef DGO_MIN_Y_STEP
#define DGO_MIN_Y_STEP 1e-6
#endif

#ifndef DGO_MIN_Y_STEP_SPREAD
#define DGO_MIN_Y_STEP_SPREAD 1e-6
#endif

#ifndef DGO_MIN_INIT_TANG_CARR
#define DGO_MIN_INIT_TANG_CARR ((size_t)1)
#endif

#ifndef DGO_PREALLOC_BRANCH_DATA
#define DGO_PREALLOC_BRANCH_DATA 10000000
#endif

#include "kde.hpp"

const size_t dgo_fork_limit = DGO_FORK_LIMIT;
using namespace std;

extern uint64_t global_branch_id;
extern uint32_t branch_level;
extern const size_t dgo_fork_limit;

template<int num_inputs>
class DiscoGradBase;

template<int num_inputs>
class DiscoGrad : public DiscoGradBase<num_inputs> {
private:
  static const size_t max_num_branch_conditions = DGO_NUM_BRANCH_COND;

  typedef array<double, num_inputs> tangent_array;

  class smallest_carrier_list {
    public:

    struct carrier_cand {
      adouble cond;
      uint32_t sample_id;

      void become(carrier_cand &&other) {
        cond.become(std::move(other.cond));
        sample_id = std::move(other.sample_id);
      }

      bool operator<(const carrier_cand &other) const { return cond.val < other.cond.val; }
      bool operator<=(const carrier_cand &other) const { return cond.val <= other.cond.val; }
      bool operator>=(const carrier_cand &other) const { return cond.val >= other.cond.val; }
      bool operator>(const carrier_cand &other) const { return cond.val > other.cond.val; }
    };

    static const size_t max_num_carriers = DGO_MAX_NUM_TANG_CARR;

    uint64_t size = 0;

    void add_candidate(adouble& cond, uint32_t sample_id) {
      if (size >= max_num_carriers) {
        if (abs(cond.val) >= items[size - 1].cond.val)
          return;
        
        items[size - 1].cond.clear_tang();
        size--;
      }

      carrier_cand item({ cond, sample_id });
      item.cond.val = abs(cond.val);

      size_t i = size;
      while (i > 0 && items[i - 1] > item) {
        items[i].become(std::move(items[i - 1]));
        i--;
      }

      items[i].become(std::move(item));

      size++;
    }

    bool empty() { return size == 0; }
    carrier_cand& operator[](size_t i) { return items[i]; };

    private:
    carrier_cand items[max_num_carriers];
  };

#ifdef __clang__
  typedef __fp16 flt16;
#else
  typedef _Float16 flt16;
#endif

  class branch_data_ {
  public:
    adouble mean_cond;
    shared_ptr<tangent_array> weight_tangent;
    vector<flt16> branch_conditions;
    size_t num_branch_visits;
    size_t num_true_visits, num_false_visits;
    smallest_carrier_list carriers_true, carriers_false;
    double kde_at_zero;
    double y_step = 0;
    vector<uint32_t> final_carriers_true, final_carriers_false;

    branch_data_() : num_branch_visits(0), num_true_visits(0), num_false_visits(0) { };

    bool has_carriers() {
      return carriers_true.size >= DGO_MIN_NUM_TANG_CARR &&
             carriers_false.size >= DGO_MIN_NUM_TANG_CARR;
    }
  };

  class branch_data_wrapper {
    branch_data_ *bd = nullptr;
    public:
    branch_data_ *get(bool alloc = false) {
      if (bd == nullptr && alloc)
        bd = new branch_data_;

      return bd;
    }
  };

  struct identity_hash {
    using is_avalanching = void;
    auto operator()(uint64_t const& x) const noexcept -> uint64_t { return x; }
  };

  uint64_t sample_branch_pos_visit[_discograd_max_branch_pos + 1];
  ankerl::unordered_dense::map<uint64_t, branch_data_, identity_hash> gid_to_branch_data;
  branch_data_wrapper **pos_to_branch_data;

#if DGO_FORK_LIMIT == 0
  vector<pair<uint64_t, branch_data_ *>> flat_branch_data;
#else
  vector<pair<uint64_t, branch_data_ *>> flat_branch_data;
#endif

  vector<double> ys;
  vector<tangent_array> dydxs;

#if DGO_MIN_EXT_PERT == true
  BoolVector *cond_signs;
#endif

  uint64_t sample_id;

  // source: https://en.wikipedia.org/wiki/Xorshift
  uint64_t xorshift64star(uint64_t x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
  }

  
public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {
    printf("DGO parameters: fork limit %ld, max. branch conditions %.0e,\n", dgo_fork_limit, (double)max_num_branch_conditions);
    printf("                min./max. tangent carriers %lu/%lu, min. initial tangent carriers: %lu\n", DGO_MIN_NUM_TANG_CARR, DGO_MAX_NUM_TANG_CARR, DGO_MIN_INIT_TANG_CARR);
    printf("                minimize external perturbations: %d\n", DGO_MIN_EXT_PERT);
    printf("                min y-step: %.2e, min y-spread: %.2e\n", DGO_MIN_Y_STEP, DGO_MIN_Y_STEP_SPREAD);
  };

  void clean_up() {
    global_branch_id = initial_global_branch_id;
    flat_branch_data.clear();

    ys.clear();
    dydxs.clear();
    
    gid_to_branch_data.clear();
  }

  void advance_global_branch_id(uint64_t branch_pos, bool then) {
    if (dgo_fork_limit == 0)
      return;

    if (this->debug)
      printf("sample %lu advancing from branch %lX\n", sample_id, global_branch_id);

    uint64_t old_global_branch_id = global_branch_id;
    global_branch_id = xorshift64star(old_global_branch_id ^ (then + 2));

    if (gid_to_branch_data.size() >= dgo_fork_limit && gid_to_branch_data.find(global_branch_id) == gid_to_branch_data.end())
      global_branch_id = old_global_branch_id;
  }

  void end_block() {
    if (dgo_fork_limit > 0)
      branch_level--;
  }

  void inc_branch_visit(uint64_t branch_pos) {
    sample_branch_pos_visit[branch_pos]++;

#if DGO_MIN_EXT_PERT == true
    cond_signs[sample_id].inc_offset();
#endif
  }

  void inc_branch_visit(uint64_t branch_pos, bool cond_sign) {
    sample_branch_pos_visit[branch_pos]++;

#if DGO_MIN_EXT_PERT == true
    cond_signs[sample_id].append(cond_sign);
#endif
  }

#if DGO_PREALLOC_BRANCH_DATA > 0
  branch_data_ *get_branch_data(uint64_t branch_pos) {
    uint64_t visit = sample_branch_pos_visit[branch_pos];
    return pos_to_branch_data[branch_pos][visit].get(true);
  }
#else
  branch_data_ *get_branch_data(uint64_t branch_pos) {
    if (pos_to_branch_data.size() < branch_pos + 1)
      pos_to_branch_data.resize(branch_pos + 1, {});

    uint64_t visit = sample_branch_pos_visit[branch_pos];

    if (pos_to_branch_data[branch_pos].size() < visit + 1)
      pos_to_branch_data[branch_pos].resize(visit + 1, {});

    return pos_to_branch_data[branch_pos][visit].get(true);
  }
#endif

  uint64_t compute_merged_gid(uint64_t set_at, uint64_t branch_pos) {
    return xorshift64star(set_at ^ ((branch_pos + 2) << 32) ^ (sample_branch_pos_visit[branch_pos] + 2));
  }

  void prepare_branch(uint64_t branch_pos, adouble& cond) {
    if (dgo_fork_limit > 0)
      branch_level++;

    inc_branch_visit(branch_pos, cond.val >= 0);

    flt16 cv = (float)cond.val;
    if (!cond.has_tang() || isinf((double)cv)) {
      advance_global_branch_id(branch_pos, cond.val < 0);
      return;
    }

#if DGO_FORK_LIMIT == 0
    branch_data_ *bd = get_branch_data(branch_pos);
#else
    branch_data_ *bd;
    bd = &gid_to_branch_data[compute_merged_gid(cond.set_at.first, branch_pos)];
#endif

    cond.val < 0 ? bd->num_true_visits++ : bd->num_false_visits++;

    if (bd->num_true_visits < DGO_MIN_INIT_TANG_CARR ||
        bd->num_false_visits < DGO_MIN_INIT_TANG_CARR)
      return;

    bd->num_branch_visits++;
    if (max_num_branch_conditions == SIZE_MAX || bd->branch_conditions.size() < max_num_branch_conditions)
      bd->branch_conditions.push_back(cv);

    if (cond.val < 0) {
      advance_global_branch_id(branch_pos, true);
      bd->carriers_true.add_candidate(cond, sample_id);
    } else {
      advance_global_branch_id(branch_pos, false);
      bd->carriers_false.add_candidate(cond, sample_id);
    }

  }

  void sample(DiscoGradProgram<num_inputs> &program) {

#if DGO_MIN_EXT_PERT == true
    cond_signs = (BoolVector *)calloc(this->num_samples, sizeof(BoolVector));
#endif

    for (sample_id = 0; sample_id < this->num_samples; sample_id++) {
      global_branch_id = initial_global_branch_id;
      branch_level = 0;
      memset(sample_branch_pos_visit, 0, sizeof(uint64_t) * (_discograd_max_branch_pos + 1));

#if DGO_MIN_EXT_PERT == true
      if (sample_id > 0)
        cond_signs[sample_id].resize(cond_signs[0].bool_size());
#endif

      array<adouble, num_inputs> pm_perturbed = this->parameters;
      tangent_array perturbation = {};
      for (int dim = 0; dim < num_inputs; dim++) {
        if (this->perturbation_dim == -1 || this->perturbation_dim == dim)
          perturbation[dim] = this->normal_dist(this->sampling_rng);

        pm_perturbed[dim] += perturbation[dim];
      }

      if (this->rs_mode)
        this->current_seed = this->seed_dist(this->rep_seed_gen);

      this->rng.seed(this->current_seed);

      adouble r = program.run(pm_perturbed);

      ys.push_back(r.val);

      tangent_array dydx;
      for (int dim = 0; dim < num_inputs; dim++)
        dydx[dim] = r.get_tang(dim);

      dydxs.push_back(dydx);
    }
  }

  void flatten_branch_data() {
#if DGO_FORK_LIMIT == 0
    int flat_branch_id = 0;
    for (uint64_t pi = 0; pi < _discograd_max_branch_pos + 1; pi++) {
      for (uint64_t bdi = 0; bdi < DGO_PREALLOC_BRANCH_DATA; bdi++) {
        auto &item = pos_to_branch_data[pi][bdi];
        branch_data_ *bd = item.get();
        if (bd != nullptr && bd->has_carriers()) {
          flat_branch_data.push_back({flat_branch_id, bd});
          flat_branch_id++;
        }
      }
    }
#else
    flat_branch_data.reserve(gid_to_branch_data.size());
    for (auto &item : gid_to_branch_data) {
      if (item.second.has_carriers())
        flat_branch_data.push_back({item.first, &item.second});
    }
#endif
  }

  void compute_branch_tangents() {
    printf("total number of branches: %ld\n", flat_branch_data.size());

    for (auto &item : flat_branch_data) {
      auto &bd = *item.second;

      assert(bd.has_carriers());

      kdepp::Kde1d<flt16, float> kde(bd.branch_conditions);
      double kde_at_zero = kde.eval(0);

      if (kde_at_zero == 0.0)
        continue;

      bd.kde_at_zero = kde_at_zero;

      size_t num_carriers = min(bd.carriers_true.size, bd.carriers_false.size) * 2;
      for (size_t i = 0; i < min(bd.carriers_true.size, bd.carriers_false.size); i++) {
        bd.mean_cond += (bd.carriers_true[i].cond + bd.carriers_false[i].cond) / num_carriers;
      }

      bd.weight_tangent = shared_ptr<tangent_array>(new tangent_array);
      for (int dim = 0; dim < num_inputs; dim++) {
        (*bd.weight_tangent)[dim] = kde_at_zero * bd.mean_cond.get_tang(dim);
      }
    }
  }

  inline size_t ids_to_key(int i, int j) {
    return (size_t) i << 32 | j;
  }

  void add_branch_tangents(double *der) {

    struct branch_priority {
       double deriv;
       branch_data_* bd;
    };

    uint64_t num_full_cmp = 0;

    unordered_map<size_t, int> dists;
    for (auto &item : flat_branch_data) {
      auto &bd = *item.second;
      if (!bd.weight_tangent)
        continue;

#if DGO_MIN_EXT_PERT == true
      double min_false = ys[bd.carriers_false[0].sample_id], max_false = ys[bd.carriers_false[bd.carriers_false.size - 1].sample_id];
      double min_true = ys[bd.carriers_true[0].sample_id], max_true = ys[bd.carriers_true[bd.carriers_true.size - 1].sample_id];

      //printf("[%.8f, %.8f], [%.8f, %.8f]\n", min_false, max_false, min_true, max_true);
      bool overlap = (min_false <= min_true && max_false >= min_true) || (min_true <= min_false && max_true >= min_false);
      double y_step_lower_bound = overlap ? 0.0 : min(abs(min_false - max_true), abs(max_false - min_true));
      double y_step_upper_bound = max(abs(max_false - min_true), abs(min_false - max_true));
      //printf("y_step_lower_bound: %.8f\n", y_step_lower_bound);
      //printf("y_step_upper_bound: %.8f\n\n", y_step_upper_bound);

      if (y_step_upper_bound < DGO_MIN_Y_STEP) {
        //printf("skipping altogether\n\n");
        bd.y_step = 0.0;
        continue;
      }

      if (y_step_upper_bound - y_step_lower_bound < DGO_MIN_Y_STEP_SPREAD) {
        bd.y_step = ys[bd.carriers_false[0].sample_id] - ys[bd.carriers_true[0].sample_id];
        //printf("would use %.8f\n", bd.y_step);
        bd.final_carriers_true.push_back(bd.carriers_true[0].sample_id);
        bd.final_carriers_false.push_back(bd.carriers_false[0].sample_id);
        continue;
      }

      num_full_cmp++;

      uint64_t min_dist = UINT64_MAX;
      vector<pair<int, int>> min_dist_iss;
      for (uint64_t i = 0; i < bd.carriers_true.size; i++) {
        int true_sample_id = bd.carriers_true[i].sample_id;
        for (uint64_t j = 0; j < bd.carriers_false.size; j++) {
          int false_sample_id = bd.carriers_false[j].sample_id;
          if (true_sample_id == false_sample_id)
            continue;

          assert(cond_signs[true_sample_id].bool_size() == cond_signs[false_sample_id].bool_size());
          size_t key;
          if (true_sample_id < false_sample_id)
            key = ids_to_key(true_sample_id, false_sample_id);
          else
            key = ids_to_key(false_sample_id, true_sample_id);

          auto it = dists.find(key);

          uint32_t dist;
          if (it != dists.end()) {
            dist = it->second;
          } else {
            dist = cond_signs[true_sample_id].abs_dist(cond_signs[false_sample_id]);
            //printf("comparing %d and %d, dist is %u\n", true_sample_id, false_sample_id, dist);
            dists[key] = dist;
          }

          if (dist <= min_dist) {
            if (dist < min_dist) {
              min_dist = dist;
              min_dist_iss.clear();
            }
            min_dist_iss.push_back({i, j});
          }
        }
      }

      if (min_dist == UINT64_MAX)
        continue;
     
      for (auto &is : min_dist_iss) {
        bd.final_carriers_true.push_back(bd.carriers_true[is.first].sample_id);
        bd.final_carriers_false.push_back(bd.carriers_false[is.second].sample_id);
      }

      for (uint64_t i = 0; i < min(bd.final_carriers_true.size(), bd.final_carriers_false.size()); i++)
        bd.y_step += ys[bd.final_carriers_false[i]] - ys[bd.final_carriers_true[i]];
      bd.y_step /= min(bd.final_carriers_true.size(), bd.final_carriers_false.size());

#else
      for (uint64_t i = 0; i < min(bd.carriers_true.size, bd.carriers_false.size); i++) {
        bd.y_step += ys[bd.carriers_false[i].sample_id] - ys[bd.carriers_true[i].sample_id];
        bd.final_carriers_true.push_back(bd.carriers_true[i].sample_id);
        bd.final_carriers_false.push_back(bd.carriers_false[i].sample_id);
      }
      bd.y_step /= bd.carriers_true.size;
#endif
    }
    printf("did %lu full path comparisons\n", num_full_cmp);

    for (int dim = 0; dim < num_inputs; dim++) {
      vector<branch_priority> branch_priorities;

      for (auto &item : flat_branch_data) {
        auto &bd = *item.second;
        if (!bd.weight_tangent || bd.y_step == 0.0)
          continue;

        branch_priority bp;
        bp.bd = &bd;

        assert(bd.has_carriers());

        double prop = (double)(bd.num_branch_visits) / this->num_samples;

        double weight_tangent = (*bd.weight_tangent)[dim];

        double curr_deriv = weight_tangent * prop * bd.y_step;

        bp.deriv = curr_deriv;

        branch_priorities.push_back(bp);
      }

      sort(branch_priorities.begin(), branch_priorities.end(), [](auto &a, auto &b) {
        return abs(a.deriv) > abs(b.deriv);
      });

      vector<bool [2]> is_carrier(this->num_samples);

      for (auto &item : branch_priorities) {
        auto &bd = *item.bd;
        bool jump_sign = bd.mean_cond.get_tang(dim) > 0;
        bool skip_branch = false;
        for (auto &carr : bd.final_carriers_true) {
          if (is_carrier[carr][jump_sign]) {
            skip_branch = true;
            break;
          }
        }
        if (skip_branch) continue;

        for (auto &carr : bd.final_carriers_false) {
          if (is_carrier[carr][1 - jump_sign]) {
            skip_branch = true;
            break;
          }
        }
        if (skip_branch) continue;

        for (auto &carr : bd.final_carriers_true)
          is_carrier[carr][jump_sign] = true;

        for (auto &carr : bd.final_carriers_false)
          is_carrier[carr][1 - jump_sign] = true;

        //printf("adding %.8f\n", item.deriv);
        der[dim] += item.deriv / this->num_replications;
      }
    }
  }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->sampling_rng.seed(random_device()());

    pos_to_branch_data = (branch_data_wrapper **)calloc(_discograd_max_branch_pos + 1, sizeof(branch_data_wrapper *));
    for (int i = 0; i < _discograd_max_branch_pos + 1; i++)
      pos_to_branch_data[i] = (branch_data_wrapper *)calloc(DGO_PREALLOC_BRANCH_DATA, sizeof(branch_data_wrapper));

    double exp = 0.0;
    double der[num_inputs] = {};
    for (uint64_t rep = 0; rep < this->num_replications; ++rep) {

      // determine seed for this replication
      if (!this->rs_mode)
        this->current_seed = this->seed_dist(this->rep_seed_gen); 

      // sample program and collect branch data
      sample(program);

      flatten_branch_data();
      compute_branch_tangents();

      // accumulate value and gradient
      adouble mean_dydxs = 0;
      for (uint64_t sample_id = 0; sample_id < this->num_samples; sample_id++) {
        for (int dim = 0; dim < num_inputs; dim++) {
          der[dim] += dydxs[sample_id][dim] / this->num_samples / this->num_replications;
        }
        exp += ys[sample_id];
      }

      add_branch_tangents(der);

      if (this->num_replications > 1)
        clean_up();
    }
    this->exp_val = (exp / this->num_samples) / this->num_replications;

    for (int dim = 0; dim < num_inputs; dim++)
      this->exp_val.set_tang(dim, der[dim]);
  }
};
