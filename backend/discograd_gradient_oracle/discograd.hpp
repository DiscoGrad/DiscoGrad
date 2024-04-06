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

#include "kde.hpp"
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
        cond.become(move(other.cond));
        sample_id = move(other.sample_id);
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
        if (abs(cond.val) >= items[size - 1].cond)
          return;
        
        items[size - 1].cond.clear_tang();
        size--;
      }

      carrier_cand item({ cond, sample_id });
      item.cond.val = abs(cond.val);

      size_t i = size;
      while (i > 0 && items[i - 1] > item) {
        items[i].become(move(items[i - 1]));
        i--;
      }

      items[i].become(move(item));

      size++;
    }

    bool empty() { return size == 0; }
    carrier_cand& operator[](size_t i) { return items[i]; };

    private:
    carrier_cand items[max_num_carriers];
  };

  class branch_data_ {
  public:
    adouble mean_cond;
    shared_ptr<tangent_array> weight_tangent;
    vector<__fp16> branch_conditions;
    size_t num_branch_visits;
    smallest_carrier_list carriers_true, carriers_false;
    double kde_at_zero;

    branch_data_() : num_branch_visits(0) { };

    bool has_carriers() {
      return carriers_true.size >= DGO_MIN_NUM_TANG_CARR &&
             carriers_false.size >= DGO_MIN_NUM_TANG_CARR;
    }
  };

  class branch_data_wrapper {
    shared_ptr<branch_data_> bd;
    public:
    shared_ptr<branch_data_> get(bool alloc = false) {
      if (bd == nullptr && alloc)
        bd = shared_ptr<branch_data_>(new branch_data_);

      return bd;
    }
  };

  struct identity_hash {
    using is_avalanching = void;
    auto operator()(uint64_t const& x) const noexcept -> uint64_t { return x; }
  };

  vector<uint64_t> sample_branch_pos_visit;
  ankerl::unordered_dense::map<uint64_t, branch_data_, identity_hash> gid_to_branch_data;
  vector<vector<branch_data_wrapper>> pos_to_branch_data;

#if DGO_FORK_LIMIT == 0
  vector<pair<uint64_t, shared_ptr<branch_data_>>> flat_branch_data;
#else
  vector<pair<uint64_t, branch_data_ *>> flat_branch_data;
#endif

  vector<double> ys;
  vector<tangent_array> dydxs;

#if DGO_MIN_EXT_PERT == true
  vector<BoolVector> cond_signs;
#endif

  uint32_t sample_id;

  // source: https://en.wikipedia.org/wiki/Xorshift
  uint64_t xorshift64star(uint64_t x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
  }

  
public:
  DiscoGrad(int argc, char **argv, bool debug=false) : DiscoGradBase<num_inputs>(argc, argv, debug) {
    printf("DGO parameters: fork limit %ld, max. branch conditions %.0e, min./max. tangent carriers %lu/%lu, minimize external perturbations: %d\n", dgo_fork_limit, (double)max_num_branch_conditions, DGO_MIN_NUM_TANG_CARR, DGO_MAX_NUM_TANG_CARR, DGO_MIN_EXT_PERT);
  };

  void clean_up() {
    global_branch_id = initial_global_branch_id;
    flat_branch_data.clear();

    sample_branch_pos_visit.clear();
    ys.clear();
    dydxs.clear();
    
    gid_to_branch_data.clear();

    pos_to_branch_data.clear();
#if DGO_MIN_EXT_PERT == true
    cond_signs.clear();
#endif
  }

  void advance_global_branch_id(uint64_t branch_pos, bool then) {
    if (dgo_fork_limit == 0)
      return;

    if (this->debug)
      printf("sample %d advancing from branch %lX\n", sample_id, global_branch_id);

    uint64_t old_global_branch_id = global_branch_id;
    global_branch_id = xorshift64star(old_global_branch_id ^ (then + 2));

    if (gid_to_branch_data.size() >= dgo_fork_limit && gid_to_branch_data.find(global_branch_id) == gid_to_branch_data.end())
      global_branch_id = old_global_branch_id;
  }

  void end_block() {
    if (dgo_fork_limit > 0)
      branch_level--;
  }

  uint64_t &branch_pos_visit(uint64_t branch_pos) {
    if (sample_branch_pos_visit.size() < branch_pos + 1)
      sample_branch_pos_visit.resize(branch_pos + 1, 0);

    return sample_branch_pos_visit[branch_pos];
  }

  void inc_branch_visit(uint64_t branch_pos, bool cond_sign = false) {
    branch_pos_visit(branch_pos)++;

#if DGO_MIN_EXT_PERT == true
    cond_signs[sample_id].append(cond_sign);
#endif
  }

  shared_ptr<branch_data_> get_branch_data(uint64_t branch_pos) {
    if (pos_to_branch_data.size() < branch_pos + 1)
      pos_to_branch_data.resize(branch_pos + 1, {});
  
    uint64_t visit = branch_pos_visit(branch_pos);

    if (pos_to_branch_data[branch_pos].size() < visit + 1)
      pos_to_branch_data[branch_pos].resize(visit + 1, {});

    return pos_to_branch_data[branch_pos][visit].get(true);
  }

  uint64_t compute_merged_gid(uint64_t set_at, uint64_t branch_pos) {
    return xorshift64star(set_at ^ ((branch_pos + 2) << 32) ^ (branch_pos_visit(branch_pos) + 2));
  }

  void prepare_branch(uint64_t branch_pos, adouble& cond) {
    if (dgo_fork_limit > 0)
      branch_level++;

    inc_branch_visit(branch_pos, cond.val >= 0);

    __fp16 cv = (float)cond.val;
    if (!cond.has_tang() || isinf(cv)) {
      advance_global_branch_id(branch_pos, cond < 0);
      return;
    }

#if DGO_FORK_LIMIT == 0
    shared_ptr<branch_data_> bd;
    bd = get_branch_data(branch_pos);
#else
    branch_data_ *bd;
    bd = &gid_to_branch_data[compute_merged_gid(cond.set_at.first, branch_pos)];
#endif

    bd->num_branch_visits++;
    if (max_num_branch_conditions == SIZE_MAX || bd->branch_conditions.size() < max_num_branch_conditions)
      bd->branch_conditions.push_back(cv);

    if (cond < 0) {
      advance_global_branch_id(branch_pos, true);
      bd->carriers_true.add_candidate(cond, sample_id);
    } else {
      advance_global_branch_id(branch_pos, false);
      bd->carriers_false.add_candidate(cond, sample_id);
    }

  }

  void sample(DiscoGradProgram<num_inputs> &program) {

#if DGO_MIN_EXT_PERT == true
    cond_signs.resize(this->num_samples);
#endif

    for (sample_id = 0; sample_id < this->num_samples; sample_id++) {
      global_branch_id = initial_global_branch_id;
      branch_level = 0;

      array<adouble, num_inputs> pm_perturbed = this->parameters;
      tangent_array perturbation = {};
      for (int dim = 0; dim < num_inputs; dim++) {
        if (this->perturbation_dim == -1 || this->perturbation_dim == dim)
          perturbation[dim] = this->normal_dist(this->sampling_rng);

        pm_perturbed[dim] += perturbation[dim];
      }

      sample_branch_pos_visit.clear();

      if (this->rs_mode)
        this->current_seed = this->seed_dist(this->rep_seed_gen);

      this->rng.seed(this->current_seed);

      adouble r = program.run(*this, pm_perturbed);
      this->lowest_sample_val = min(this->lowest_sample_val, r.val);

      ys.push_back(r.val);

      tangent_array dydx;
      for (int dim = 0; dim < num_inputs; dim++)
        dydx[dim] = r.get_tang(dim);

      dydxs.push_back(dydx);
    }
  }

  void flatten_branch_data() {
#if DGO_FORK_LIMIT == 0
    int branch_pos = 0;
    for (auto &pos_items : pos_to_branch_data) {
      for (auto &item : pos_items) {
        shared_ptr<branch_data_> bd = item.get();
        if (bd != nullptr && bd->has_carriers())
          flat_branch_data.push_back({branch_pos, bd});
      }
      branch_pos++;
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

      kdepp::Kde1d<__fp16, float> kde(bd.branch_conditions);
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

  void add_branch_tangents(double *der) {

    struct branch_priority {
       double deriv;
       vector<uint32_t> carr_true, carr_false;
       branch_data_* bd;
    };

    for (int dim = 0; dim < num_inputs; dim++) {
      vector<branch_priority> branch_priorities;

      for (auto &item : flat_branch_data) {
        auto &bd = *item.second;
        if (!bd.weight_tangent)
          continue;

        branch_priority bp;
        bp.bd = &bd;

        assert(bd.has_carriers());

        double prop = (double)(bd.num_branch_visits) / this->num_samples;
        double y_step = 0;

        double weight_tangent = 0;

#if DGO_MIN_EXT_PERT == true
        uint64_t min_dist = UINT64_MAX;
        pair<int, int> min_dist_is;
        for (int i = 0; i < bd.carriers_true.size; i++) {
          int true_sample_id = bd.carriers_true[i].sample_id;
          for (int j = 0; j < bd.carriers_false.size; j++) {
            int false_sample_id = bd.carriers_false[j].sample_id;
            if (true_sample_id == false_sample_id)
              continue;

            assert(cond_signs[true_sample_id].bool_size() == cond_signs[false_sample_id].bool_size());
            uint32_t dist = cond_signs[true_sample_id].abs_dist(cond_signs[false_sample_id]);
            if (dist < min_dist) {
              min_dist = dist;
              min_dist_is = {i, j};
            }
          }
        }

        if (min_dist == UINT64_MAX)
          continue;
       
        auto &carr_true = bd.carriers_true[min_dist_is.first];
        auto &carr_false = bd.carriers_false[min_dist_is.second];

        bp.carr_true.push_back(bd.carriers_true[min_dist_is.first].sample_id);
        bp.carr_false.push_back(bd.carriers_false[min_dist_is.second].sample_id);

        double y_true = ys[carr_true.sample_id];
        double y_false = ys[carr_false.sample_id];
        assert(carr_true.sample_id != carr_false.sample_id);

        y_step = y_false - y_true;

        weight_tangent = (*bd.weight_tangent)[dim];

        if (weight_tangent == 0.0)
          continue;
#else
        for (int i = 0; i < min(bd.carriers_true.size, bd.carriers_false.size); i++) {
          y_step += ys[bd.carriers_false[i].sample_id] - ys[bd.carriers_true[i].sample_id];
          bp.carr_true.push_back(bd.carriers_true[i].sample_id);
          bp.carr_false.push_back(bd.carriers_false[i].sample_id);
        }
        y_step /= bd.carriers_true.size;
        weight_tangent = (*bd.weight_tangent)[dim];
#endif

        double curr_deriv = weight_tangent * prop * y_step;

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
        for (auto &carr : item.carr_true) {
          if (is_carrier[carr][jump_sign]) {
            skip_branch = true;
            break;
          }
        }
        if (skip_branch) continue;

        for (auto &carr : item.carr_false) {
          if (is_carrier[carr][1 - jump_sign]) {
            skip_branch = true;
            break;
          }
        }
        if (skip_branch) continue;

        for (auto &carr : item.carr_true)
          is_carrier[carr][jump_sign] = true;

        for (auto &carr : item.carr_false)
          is_carrier[carr][1 - jump_sign] = true;

        der[dim] += item.deriv / this->num_replications;
      }
    }
  }

  void estimate_(DiscoGradProgram<num_inputs> &program) {
    this->sampling_rng.seed(random_device()());

    double exp = 0.0;
    double der[num_inputs] = {};
    for (int rep = 0; rep < this->num_replications; ++rep) {

      // determine seed for this replication
      if (!this->rs_mode)
        this->current_seed = this->seed_dist(this->rep_seed_gen); 

      // sample program and collect branch data
      sample(program);

      flatten_branch_data();
      compute_branch_tangents();

      // accumulate value and gradient
      adouble mean_dydxs = 0;
      for (int sample_id = 0; sample_id < this->num_samples; sample_id++) {
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
