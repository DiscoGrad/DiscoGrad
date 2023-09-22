/*
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
#include <unordered_map>
#include <vector>
#include <queue>
#include <array>
#include <iostream>
#include "si_constants.hpp"

/** Internal representation of a single Gaussian distribution ie a tuple (mean, variance).
 *  This class is used to represent a single mixture element of the smooth state (the weights of
 *  which are stored separately).
 *  It provides methods to operate on gaussian disributions in a differentiable manner, i.e. the
 *  mean m and variance v are stored in the differentiable container type adouble. For the variance,
 *  the container type can be chosen by setting @see variance_t. The result of each operation
 *  is approximated by a Gaussian distribution, which is only exact for linear operations.
 *
 */
class SiGaussian {
public:
  adouble m;    /**< The mean of the gaussian distribution. */
  variance_t v; /**< The variance of the gaussian distribution. */

  SiGaussian() : m(false), v(false) {}
  SiGaussian(adouble m, adouble v) : m(m), v(cast_to_variance_t<adouble, variance_t>(v)) {}
  explicit SiGaussian(adouble val) : m(val), v(0.0) {}
  SiGaussian(double val) : m(val), v(0.0) {}

  /* mathematical operators */

  SiGaussian operator-() { return {-m, v}; }

  SiGaussian operator+=(SiGaussian &other) { return *this = {m + other.m, v + other.v}; }
  SiGaussian operator-=(SiGaussian &other) { return *this = {m - other.m, v + other.v}; }
  SiGaussian operator*=(SiGaussian &other) { return *this = {m * other.m, v * other.m * other.m + other.v * m * m}; }
  SiGaussian operator/=(SiGaussian &other) {
    return *this = {m / other.m, v / (other.m * other.m) + other.v * (m / (other.m * other.m)).ipow(2)};
  }

  SiGaussian operator+(const SiGaussian &other) { return {m + other.m, v + other.v}; }
  SiGaussian operator-(const SiGaussian &other) { return {m - other.m, v + other.v}; }
  SiGaussian operator*(const SiGaussian &other) { return {m * other.m, v * other.m * other.m + other.v * m * m}; }
  SiGaussian operator/(const SiGaussian &other) {
    return {m / other.m, v / (other.m * other.m) + other.v * (m / (other.m * other.m)).ipow(2)};
  }

  SiGaussian operator+=(adouble other) { return *this = *this + other; }
  SiGaussian operator-=(adouble other) { return *this = *this - other; }
  SiGaussian operator*=(adouble other) {
    m *= other;
    v *= cast_to_variance_t<adouble, variance_t>(other * other);
    return *this;
  }
  SiGaussian operator/=(adouble other) {
    m /= other;
    v /= cast_to_variance_t<adouble, variance_t>(other * other);
    return *this;
  }

  SiGaussian operator+=(double other) { return *this = *this + other; }
  SiGaussian operator-=(double other) { return *this = *this - other; }
  SiGaussian operator*=(double other) {
    m *= other;
    v *= cast_to_variance_t<double, variance_t>(other * other);
    return *this;
  }
  SiGaussian operator/=(double other) {
    m /= other;
    v /= cast_to_variance_t<double, variance_t>(other * other);
    return *this;
  }

  SiGaussian operator+(adouble other) { return {m + other, v}; }
  SiGaussian operator-(adouble other) { return {m - other, v}; }
  SiGaussian operator*(adouble other) { return {m * other, v * (other * other)}; }
  SiGaussian operator/(adouble other) { return {m / other, v / (other * other)}; }

  SiGaussian operator+(double other) { return {m + other, v}; }
  SiGaussian operator-(double other) { return {m - other, v}; }
  SiGaussian operator*(double other) { return {m * other, v * (other * other)}; }
  SiGaussian operator/(double other) { return {m / other, v / (other * other)}; }

  /** Uses the normal distribution's cdf to calculate the probability of a random variable following
   *  a Gaussian distribution of ({@link m}, {@link v}) being less than zero. Used for the implementation
   *  of the (smooth) comparison operators.
   *  It provides \f$P(X < 0)\f$ for \f$X \sim g\f$.
   *  @param g The Gaussian distribution of the random variable.
   *  @param eq Whether to include the zero value in case g.v==0 (i.e. <= 0).
   *  @returns The probability of g being less than zero.
   */
  adouble prob_lt_zero(SiGaussian g, bool eq = false) {
    adouble v = g.v.val; // uses non-differentiable variance at the moment

    if (si_dea_input_variance != 0.0) {
      v = 0.0;
      for (int i = 0; i < adouble::num_adjoints; i++)
        v += ipow(g.m.get_adj(i), 2);
      v *= si_dea_input_variance;
    }

    // special case to avoid numerical problems (sqrt(0)),
    // as in the case g.v==0, g degenerates to a dirac delta distribution
    if (v == 0.0)
      return eq ? g.m <= 0.0 : g.m < 0.0;

    // this is just the normal distribution's cdf
    adouble r = 1.0 - 0.5 * (1.0 + erf(g.m / (sqrt(v) * sqrt(2))));

    if (isnan(r.val))
      return 0.0;
    //assert(!isnan(r.val));

    return r;
  }

  /* comparison operators. */

  adouble operator<(adouble other) { return prob_lt_zero({m - other, v}); }
  adouble operator<=(adouble other) { return prob_lt_zero({m - other, v}, true); }
  adouble operator==(adouble other) {
    adouble lower = other - 0.5;
    adouble upper = other + 0.5;
    adouble prob = operator<(upper) - operator<(lower);
    return prob;
  }
  adouble operator!=(adouble other) { return 1 - operator==(other); }

  adouble operator<(double other) { return prob_lt_zero({m - other, v}); }
  adouble operator<=(double other) { return prob_lt_zero({m - other, v}, true); }
  adouble operator==(double other) {
    adouble lower = other - 0.5;
    adouble upper = other + 0.5;
    adouble prob = operator<(upper) - operator<(lower);
    return prob;
  }
  adouble operator!=(double other) { return 1 - operator==(other); }
};

/** The state of a single control flow path through the program in particular scope.
 *  It contains the weight (probability of the control flow path being taken) and a
 *  mixture element (see {@link SiGaussian}) for each of the variables that exist on a particular
 *  tracked control flow path and within the scope.
 */
class SiPathState {
private:
  /** A deque is used to store the variable states (gaussian mixture elements) on a single path.
   *  This dynamic data structure does not invalidate pointers to entries upon insertion/deletion.
   *  This is necessary because the user-facing {@link sdouble} needs to retain its mapping to the
   *  correct internal values. They are stored along with their id to identify them across path states.
   *  The variables' offsets are looked up via an unordered_map.
   */
  deque<pair<uint64_t, SiGaussian>> path_state;
  deque<pair<uint64_t, SiGaussian>> tmp_path_state; /**< variables never assigned to */

  uint64_t max_stored_idx = 0;

  /** Cache to locate variables in path_state */
  unordered_map<uint64_t, uint64_t> idx_to_offset;

public:
  /** The weight of this path state. Corresponds to the probability of this control flow path being taken. */
  adouble weight = 1.0;

  /** @returns A forward iterator to the beginning of the internal state dequeue. */
  auto begin() { return path_state.begin(); }
  /** @returns A reverse iterator to the beginning of the internal state dequeue. */
  auto rbegin() { return path_state.rbegin(); }
  /** @returns A forward iterator to the end of the internal state dequeue. */
  auto end() { return path_state.end(); }
  /** @returns A reverse iterator to the end of the internal state dequeue. */
  auto rend() { return path_state.rend(); }

  /** Removes any variables outside size and clears all temporary variables.
   *  Used to clear any scoped variables when exiting a scope, see {@link SiStack}.
   */
  void clean_up(size_t size) {
    path_state.resize(size);
    tmp_path_state.clear();
  }

  /** Method to access the value of a particular variable ({@link SiGaussian}) on this
   *  path state by it's unique idx.
   *  If no value is found for the given idx, a new value is created and returned.
   *  Internally, caching is used to reduce the lookup time for frequently accessed idxs.
   *  @param idx The unique id of the variable to access.
   *  @param set Whether the variable should be assignable or is a temporary value in the user code.
   *  @returns The value of the variable on this path state.
   */
  SiGaussian &access(uint64_t idx, bool set) {
    if (idx <= max_stored_idx) { // otherwise the path state cannot have this idx yet
      // try to find in tmp variables

      auto it = idx_to_offset.find(idx);
      if (it != idx_to_offset.end())
        return path_state[it->second].second;

      for (auto it = tmp_path_state.rbegin(); it != tmp_path_state.rend(); it++) {
        if (it->first == idx) {
          return it->second;
        }
      }

    }


    // when the value does not (yet) exist on this path state, create it
    max_stored_idx = max(max_stored_idx, idx);
    if (set) {
      path_state.push_back({idx, SiGaussian(false)});
      idx_to_offset[idx] = size() - 1;
      return path_state.back().second;
    }

    tmp_path_state.push_back({idx, SiGaussian(false)});
    return tmp_path_state.back().second;
  }

  /** @returns The value of a particular variable on this path. */
  SiGaussian &get(uint64_t idx) { return access(idx, true); }

  /** @returns The value of a particular temporary variable on this path. */
  SiGaussian &get_temp(uint64_t idx) { return access(idx, false); }

  /** @returns The number of variables on this path state. */
  size_t size() { return path_state.size(); }

  /** @returns Whether the weight (probability) of this path state is larger than
   *  the weight of another path state. Used for sorting in Truncate heuristic.
   *  NOTE: we can just change rbegin/end for begin/end there...but this needs testing.
   */
  bool operator<(const SiPathState &other) { return weight.val > other.weight.val; }

  /** Compute the impact on the complete program state's distribution when
   *  merging this path state with other.
   *  This implements the cost measure given in Chaudhuri et. al. and is used
   *  by some of the {@link restrict_mode_t restrict modes}. It iterates all the
   *  corresponding variables of the two path states to determine how their
   *  merged value would deviate from their original values.
   *  @param other The {@link SiPathState} to merge with.
   *  @returns The cost of merging other into this.
   */
  double compute_merge_cost(SiPathState *other) {

    if (other == NULL)
      return DBL_MAX;

    double other_weight = other->weight.val;

    double own_weight = weight.val;
    double sum_weight = weight.val + other_weight;

    double sum_cost = 0;

    for (auto it = begin(); it != end(); it++) {
      double other_mean = other->get_temp(it->first).m.val;
      double new_mean = ((own_weight * it->second.m.val + other_weight * other_mean) / sum_weight);
      if (new_mean == -1.0 / 0.0) {
        return DBL_MAX;
      }

      //if (new_mean

      double cost;
      if (si_restrict_mode == si_merge_chaudhuri)
        cost = abs(own_weight * (it->second.m.val - new_mean)) + abs(other_weight * (other_mean - new_mean));
      else if (si_restrict_mode == si_merge_chaudhuri_ignore_weights)
        cost = abs(it->second.m.val - new_mean) + abs(other_mean - new_mean);
      else
        assert(false);
      sum_cost += cost;
    }

    return sum_cost;
  }

  /** Merges two path states into one by "absorbing" another path state into this one.
   *  This is done by taking the average of each value as the corresponding value of the
   *  merged path state, see Chaudhuri et al.
   *  @param other The {@link SiPathState} to absorb.
   *  @param merge_cost The cost of merging the two path states. Used to check for the
   *  {@link si_max_merge_cost} limit.
   */
  void absorb(SiPathState *other, double merge_cost = 0.0) {
    assert(this != other);

    double sum_weight(weight.val + other->weight.val);

    for (auto it = begin(); it != end(); it++) {

      adouble &own_m = it->second.m;
      adouble &own_v = it->second.v;
      SiGaussian &other_g = other->get_temp(it->first);
      adouble &other_m = other_g.m;
      adouble &other_v = other_g.v;

      if (own_m.val == other_m.val && own_v.val == other_v.val && !other_m.has_adj() && !other_v.has_adj())
        continue;

      adouble new_m = (weight.val * own_m + other->weight.val * other_m) / sum_weight;

      adouble new_m_tmp = (weight.val * own_m + other->weight.val * other_m);

      // correct for numerical errors
      if (own_m.val == other_m.val)
        new_m.val = own_m.val;

      adouble sqrt_own_v = (own_v != 0.0 ? sqrt(own_v) : (adouble)0.0);
      adouble sqrt_other_v = (other_v != 0.0 ? sqrt(other_v) : (adouble)0.0);

      auto new_stddev =
          (weight.val * (sqrt_own_v + 2 * pow((own_m.val - new_m.val), 2)) + other->weight.val * (sqrt_other_v + 2 * pow((other_m.val - new_m.val), 2))) /
          sum_weight; // variance calculation from Chaudhuri paper

      adouble new_variance = new_stddev * new_stddev;

      if (si_max_variance != DBL_MAX && new_variance.val > si_max_variance) {
        new_variance = si_max_variance;
      }

      if (si_max_variance_factor_by_merge != DBL_MAX) {
        double max_variance;
        if (own_v.val == 0.0 && other_v.val == 0.0)
          max_variance = si_max_variance;
        else
          max_variance = max(own_v.val, other_v.val) * si_max_variance_factor_by_merge;

        if (new_variance > max_variance) {
          new_variance = own_v * weight.val + other_v * other->weight.val;
        }
      }


      it->second.m = new_m;
      it->second.v = cast_to_variance_t<adouble, variance_t>(new_variance);

    }
    weight += other->weight;
  }
};

static deque<SiPathState> all_path_states;
static vector<SiPathState *> unused_path_states;

/** Container class to operate on the weights of the {@link SiPathState}s for a particular scope.
 *  It overloads the boolean operators !, || and && to execute boolean operations in branching conditions.
 */
class SiPathWeights {
public:
  /** A weight for each tracked path. */
  vector<adouble> weights;

  SiPathWeights(){};

  explicit SiPathWeights(bool b);

  /** @returns \f$1-w\f$ for all \f$w\f$s in the weights.*/
  SiPathWeights operator!() {
    SiPathWeights r;
    for (auto it = weights.begin(); it != weights.end(); it++)
      r.weights.push_back(1 - *it);
    return r;
  }

  /** Multiplies the weights with the corresponding weights of other to retrieve
   *  the weight (probability) of both states occuring.
   */
  SiPathWeights &operator&&(const SiPathWeights &other) {
    // reuse of "this" is okay since && and || only occur on temporaries
    for (size_t idx = 0; idx < weights.size(); idx++)
      weights[idx] *= other.weights[idx];
    return *this;
  }

  /** Retrieves the weights of either state occuring. */
  SiPathWeights &operator||(const SiPathWeights &other) {
    for (size_t idx = 0; idx < weights.size(); idx++)
      weights[idx] = 1 - (1 - weights[idx]) * (1 - other.weights[idx]);
    return *this;
  }

  SiPathWeights &operator&&(const bool &other) {
    for (size_t idx = 0; idx < weights.size(); idx++)
      weights[idx] *= other;
    return *this;
  }

  SiPathWeights &operator||(const bool &other) {
    for (size_t idx = 0; idx < weights.size(); idx++)
      weights[idx] = 1 - (1 - weights[idx]) * (1 - other);
    return *this;
  }

  operator bool() const {
    assert(false);
    return false;
  } // just to appease clang during source transformation

  void print() {
    for (size_t idx = 0; idx < weights.size(); idx++)
      printf("%.2f ", weights[idx].val);
    printf("\n");
  }
};

/** The smooth program state within a particular scope.
 *  Unifies all {@link SiGaussian}s on all {@link SiPathState}s into a single container.
 *  Provides methods to operate on the state.
 */
class SiState {
public:
  size_t num_parent_variables = 0;

  /** The program state on the tracked control flow paths through the program. */
  vector<SiPathState *> path_states;

  /** All the path's states that have been suspended due to a brake statement. */
  vector<SiPathState *> break_path_states;
  /** All the path's states that have been suspended due to a continue statement. */
  vector<SiPathState *> continue_path_states;
  /** All the path's states that have been suspended due to a return statement. */
  vector<SiPathState *> return_path_states;

  SiState(size_t size) { path_states.resize(size); }
  SiState() : SiState(1) {} // without branching, there is only one possible state

  // note that break_/continue_/return_path_states are not copied, since child states should not inherit them
  SiState(const SiState &other) : num_parent_variables(other.num_parent_variables), path_states(other.path_states){};

  void clear() { path_states.clear(); };

  void push_back(SiPathState *path_state) { path_states.push_back(path_state); }

  /** Print all non-suspended path states.
   *  @returns the sum of their weights.
   */
  double print_path_states(vector<SiPathState *> &ps, string name = "") {
    double sum_weight = 0.0;
    for (auto it = ps.begin(); it != ps.end(); it++) {
      printf((name + "path %ld out of %ld, weight %.5g (").c_str(), it - ps.begin(), ps.size(), (*it)->weight.val);
      for (int i = 0; i < adouble::num_adjoints; i++)
        printf("%.4g ", (*it)->weight.get_adj(i));
      printf("): \n");
      sum_weight += (*it)->weight.val;
      for (auto iit = (*it)->begin(); iit != (*it)->end(); iit++) {
        printf("%ld: (m: %.2f, v: %.2g, sd: %.2f)\n", iit->first, iit->second.m.val, iit->second.v.val, sqrt(iit->second.v.val));
        if (print_adjoints) {
          printf("adjoints: ");
          for (int i = 0; i < adouble::num_adjoints; i++)
            printf("%.4g ", iit->second.m.get_adj(i));
          printf(")\n");
        }
      }
    }
    return sum_weight;
  }

  /** Print all path states (for debugging). */
  void print() {
    double sum_weight = 0.0;
    sum_weight += print_path_states(path_states);
    sum_weight += print_path_states(break_path_states, "break ");
    sum_weight += print_path_states(continue_path_states, "continue ");
    sum_weight += print_path_states(return_path_states, "return ");

    printf("sum of weights: %.2f\n", sum_weight);
  }

  SiPathState *operator[](size_t i) { return path_states[i]; }

  auto begin() { return path_states.begin(); }
  auto end() { return path_states.end(); }
  size_t size() { return path_states.size(); }
  bool empty() { return path_states.empty(); }
  SiPathState *back() { return path_states.back(); }

  /** Utility data structure for the restriction algorithms.
   *  Contains the path weight, selection value (always zero at the moment)
   *  the index of the path state and whether it is a weight for
   *  the "then"-part of the if-statement.
   */
  // weight, select_weight (always zero at the moment), origin path state idx, is_then
  typedef tuple<adouble, double, uint64_t, bool> weight_tuple;
  /** Compare the randomized weights or deviation of a {@link weight_tuple} for sorting. */
  static constexpr auto cmp_weight_tuple = [](weight_tuple const &a, weight_tuple const &b) -> bool { return get<1>(a) < get<1>(b); };

  /** Helper function for cloning: Generates candidate path states for the truncate and wrs
   * {@link restrict_mode_t restrict modes} when encountering a branching statement.
   * @param weights The current weights of the path states.
   * @param[out] weight_tuples The internal {@link weight_tuple} list holding restrict algorithm-
   * related data ("candidate path states").
   * @param[out] target_sum_weight The sum of the weights of the current path states.
   */
  void generate_cand_path_states(const SiPathWeights &weights, vector<weight_tuple> &weight_tuples, adouble &target_sum_weight) {
    assert(!weights.weights.empty());

    // for each existing path state, calculate two new weights for the two resulting path states of the branch
    for (size_t idx = 0; idx < weights.weights.size(); idx++) {
      const adouble &w = weights.weights[idx];
      adouble else_then_weights[] = {(1.0 - w) * path_states[idx]->weight, w * path_states[idx]->weight};

      double sum_weight = else_then_weights[0].val + else_then_weights[1].val;

      assert(path_states[idx]->weight > si_min_weight);
      // for each weight candidate, add the appropriate information for the restrict heuristics
      for (int i = 0; i < 2; i++) {
        adouble &cw = else_then_weights[i];
        double select_weight = 0.0;
        if (cw.val > si_min_weight && cw.val / sum_weight > si_min_branch_prob) {
          weight_tuples.push_back({cw, select_weight, idx, i});
        }
      }

      target_sum_weight += else_then_weights[0] + else_then_weights[1];
    }
  }

  /** Helper function for cloning: Fill new states either with the original path states or copies thereof.
   *  If the generated path states have a weight lower than target_sum_weight, their weights are scaled appropriately.
   *  @param num_new_path_states The index up to which the candidate path states are accepted.
   *  @param cand_path_states The internal {@link weight_tuple} list holding restrict algorithm-
   *  related data ("candidate path states"), as generated by {@link generate_cand_path_states()}.
   *  @param num_copies_needed For each candidate path state the number of copies to create.
   *  @param[out] new_states The new states, generated from the candidates. An array holding two {@link SiState}s
   *  storing the "then"-states and "else"-states, respectively.
   *  @param target_sum_weight The sum of weights of the original path states to reach.
   */
  void fill_cloned_states(uint64_t num_new_path_states, vector<weight_tuple> &cand_path_states, vector<uint64_t> &num_copies_needed,
                          SiState *new_states, adouble target_sum_weight) {
    if (num_new_path_states == 0)
      return;

    adouble final_sum_weight = 0.0;
    for (size_t widx = 0; widx < num_new_path_states; widx++) {
      auto &wt = cand_path_states[widx];
      size_t sidx = get<2>(wt); // origin path state idx
      bool then = get<3>(wt); // is this a weight for a "then"-part of a branch?

      assert(num_copies_needed[sidx] > 0);
      assert(!unused_path_states.empty() || num_copies_needed[sidx] == 1);

      if (num_copies_needed[sidx] == 2) {
        new_states[then].push_back(unused_path_states.back());
        *(new_states[then].back()) = *path_states[sidx];
        unused_path_states.pop_back();
        num_copies_needed[sidx]--;
      } else { // no more copies needed, or no space left for another one
        new_states[then].push_back(path_states[sidx]);
        num_copies_needed[sidx] = 0;
      }

      adouble new_weight = get<0>(wt);
      new_states[then].back()->weight = new_weight;
      final_sum_weight += new_weight;
    }


    // ensure the weights sum up to target_sum_weight (might not be the case for all restrict
    // heuristics at this point)
    adouble weight_factor = target_sum_weight / final_sum_weight;
    for (int then = 0; then < 2; then++) {
      for (auto it = new_states[then].begin(); it != new_states[then].end(); it++) {
        (*it)->weight *= weight_factor;
        assert((*it)->weight > si_min_weight);
      }
    }

    new_states[0].num_parent_variables = new_states[1].num_parent_variables = path_states[0]->size();

    assert(new_states[0].size() + new_states[1].size() <= si_max_path_states && new_states[0].size() + new_states[1].size() > 0);
  }

  /** Truncate heuristic for restricting the state size (see {@link restrict_mode_t}).
   *  Only the candidate path states with the highest weights are retained and others are discarded (truncate).
   *  In wrs-mode, they are selected randomly, but based on their weight.
   */
  void clone_truncate(const SiPathWeights &weights, SiState *new_states) {

    if (size() == 0)
      return;

    adouble target_sum_weight = 0.0;
    vector<weight_tuple> cand_path_states;
    generate_cand_path_states(weights, cand_path_states, target_sum_weight);

    sort(cand_path_states.rbegin(), cand_path_states.rend());

    // get number of copies needed and number of non-zero weights
    vector<uint64_t> num_copies_needed(size());
    uint64_t num_alive_path_states = 0;
    uint64_t num_alive_cand_path_states = 0;

    for (size_t widx = 0; widx < cand_path_states.size(); widx++) {
      auto &wt = cand_path_states[widx];
      adouble weight = get<0>(wt);
      uint64_t sidx = get<2>(wt);

      assert(weight > si_min_weight);

      num_copies_needed[sidx]++;
      if (num_copies_needed[sidx] == 1)
        num_alive_path_states++;
      num_alive_cand_path_states++;
    }

    // free unused states
    for (size_t sidx = 0; sidx < size(); sidx++)
      if (!num_copies_needed[sidx])
        unused_path_states.push_back(path_states[sidx]);

    // identify new path states that fit in the available slots _in weight-order_
    uint64_t num_new_path_states = min(num_alive_path_states + unused_path_states.size(), num_alive_cand_path_states);
    uint64_t slots_left = num_new_path_states;

    for (size_t widx = 0; widx < cand_path_states.size(); widx++) {
      size_t sidx = get<2>(cand_path_states[widx]);

      assert(num_copies_needed[sidx] > 0);
      if (!num_copies_needed[sidx]) // can this happen?
        continue;

      if (slots_left > 0) {
        slots_left--;
      } else {
        num_copies_needed[sidx]--;
      }

      if (num_copies_needed[sidx] == 0)
        unused_path_states.push_back(path_states[sidx]);
    }


    fill_cloned_states(num_new_path_states, cand_path_states, num_copies_needed, new_states, target_sum_weight);
  }

  /** Implementation of the original Restrict algorithm by Chaudhuri et al. Also see {@link restrict_mode_t}. */
  // done before SiPathWeights for a branch is constructed because the branch weights will be different when computed after merging path states
  void merge_chaudhuri() {
    if (size() == 1)
      return;

    uint64_t target_num_input_states = min(size(), (size() + unused_path_states.size()) / 2);
    uint64_t num_merges = size() - target_num_input_states;

    if (num_merges == 0)
      return;

    static double **cost_table;
    if (cost_table == NULL) {
      cost_table = new double *[si_max_path_states];
      for (size_t i = 0; i < si_max_path_states; i++) {
        cost_table[i] = new double[si_max_path_states];
      }
    }

    typedef pair<double, pair<size_t, size_t>> cost_pair;
    typedef priority_queue<cost_pair, vector<cost_pair>, greater<cost_pair>> cost_pq_t;
    static cost_pq_t cost_pq;
    cost_pq = cost_pq_t();

    for (size_t a = 0; a < size(); a++) {
      for (size_t b = a + 1; b < size(); b++) {
        double cost = path_states[a]->compute_merge_cost(path_states[b]);
        cost_table[a][b] = cost;
        cost_pq.push({cost, {a, b}});
      }
    }

    while (num_merges-- > 0) {
      cost_pair cp;
      size_t dst_sidx;
      size_t src_sidx;

      SiPathState *dst;
      SiPathState *src;

      // find first pq element that's not outdated
      do {
        cp = cost_pq.top();
        cost_pq.pop();

        dst_sidx = cp.second.first;
        src_sidx = cp.second.second;

        dst = path_states[dst_sidx];
        src = path_states[src_sidx];

      } while (dst == NULL || src == NULL || cp.first != cost_table[dst_sidx][src_sidx]);


      dst->absorb(src, cost_table[dst_sidx][src_sidx]);

      unused_path_states.push_back(src);
      path_states[src_sidx] = NULL;

      for (size_t a = 0; a < dst_sidx; a++) {
        if (path_states[a] == NULL)
          continue;
        double cost = path_states[a]->compute_merge_cost(dst);
        cost_table[a][dst_sidx] = cost;
        cost_pq.push({cost, {a, dst_sidx}});
      }

      for (size_t b = dst_sidx + 1; b < size(); b++) {
        double cost = dst->compute_merge_cost(path_states[b]);
        cost_table[dst_sidx][b] = cost;
        cost_pq.push({cost, {dst_sidx, b}});
      }
    }

    auto rm = remove_if(begin(), end(), [](SiPathState *p) { return p == NULL; });
    path_states.erase(rm, end());
  }

  /** Modification of the Restrict algorithm by Chaudhuri et al that only considers the weights
   * for calculating the cost of merging.
   */
  void merge_by_weight() {
    if (size() == 1)
      return;

    uint64_t target_num_input_states = min(size(), (size() + unused_path_states.size()) / 2);
    uint64_t num_merges = size() - target_num_input_states;


    if (num_merges == 0)
      return;

    typedef pair<double, size_t> weight_pair;
    typedef priority_queue<weight_pair, vector<weight_pair>, greater<weight_pair>> weight_pq_t;
    static weight_pq_t weight_pq;
    weight_pq = weight_pq_t();

    for (size_t sidx = 0; sidx < size(); sidx++)
      weight_pq.push({path_states[sidx]->weight.val, sidx});

    while (num_merges-- > 0) {
      weight_pair cp;
      size_t dst_src_sidx[2];

      SiPathState *dst_src[2];

      for (int i = 0; i < 2; i++) {
        cp = weight_pq.top();
        weight_pq.pop();
        dst_src_sidx[i] = cp.second;
        dst_src[i] = path_states[cp.second];
      }

      dst_src[0]->absorb(dst_src[1]);

      weight_pq.push({dst_src[0]->weight.val, dst_src_sidx[0]});

      unused_path_states.push_back(dst_src[1]);
      path_states[dst_src_sidx[1]] = NULL;
    }

    auto rm = remove_if(begin(), end(), [](SiPathState *p) { return p == NULL; });
    path_states.erase(rm, end());
  }

  void copy_path_states_from(SiState &src, bool include_active = true) {

    if (include_active) {
      for (auto it = src.begin(); it != src.end(); it++)
        push_back(*it);
    }

    for (auto it = src.break_path_states.begin(); it != src.break_path_states.end(); it++)
      break_path_states.push_back(*it);

    for (auto it = src.continue_path_states.begin(); it != src.continue_path_states.end(); it++)
      continue_path_states.push_back(*it);

    for (auto it = src.return_path_states.begin(); it != src.return_path_states.end(); it++)
      return_path_states.push_back(*it);
  }

  /** Remove all path states with a weight below min_prob.
   *  Together with the definition of {@link SiPathWeights} this can be used to prune any
   *  tracked paths where a certain condition has probability of occuring higher than min_prob.
   *  @param weights The condition to test.
   *  @param min_prob Any tracked paths where the condition has a higher probability of occuring
   *  are removed.
   */
  void si_assert(SiPathWeights weights, double min_prob) {
    //for (auto it = weights.weights.rbegin(); it != weights.weights.rend(); it++)
    //  sum_weight += it->val;
    for (auto it = weights.weights.rbegin(); it != weights.weights.rend(); it++) {
      printf("%.4g, %.4g\n", it->val, min_prob);
      if (it->val < min_prob)
        path_states[weights.weights.size() - 1 - (it - weights.weights.rbegin())] = NULL;
    }
    auto rm = remove_if(begin(), end(), [](SiPathState *p) { return p == NULL; });
    path_states.erase(rm, end());
  }
};

/** Core data structure: Holds the {@link SiState}s for each scope, the top() element is the state
 *  of the current scope.
 *  This class represents the complete smooth program state. It provides functions to execute common
 *  imperative control flow such as branching and looping in a smoothed way.
 */
class SiStack {
public:
  deque<SiState> stack;
  size_t max_instantiated_path_states = 1;

  size_t cumulative_instantiated_path_states = 0;

  size_t cumulative_num_variables = 0;

  uint64_t num_branches = 0;

  /** @copydoc si_max_path_states */
  void set_max_path_states(size_t num_states) { si_max_path_states = num_states; }

  /** @copydoc si_restrict_mode */
  void set_restrict_mode(int restrict_mode) { si_restrict_mode = (restrict_mode_t)restrict_mode; };

  /** @copydoc si_min_branch_prob */
  void set_min_branch_prob(double branch_prob) { si_min_branch_prob = branch_prob; }

  void set_dea_input_variance(double input_variance) { si_dea_input_variance = input_variance; };

  size_t size() { return stack.size(); }

  bool empty() { return stack.empty(); }

  void clear() {
    max_instantiated_path_states = 1;
    cumulative_instantiated_path_states = 0;
    num_branches = 0;
    stack.clear();
    all_path_states.clear();
    unused_path_states.clear();
  }

  /* Methods that handle control-flow in a smooth way. */

  /** When entering a new scope, a copy of the current scopes {@link SiState} is
   *  pushed onto the stack.
   */
  void enter_scope() {
    if (all_path_states.empty()) {

      // exit on nan or inf
      //feenableexcept(FE_INVALID | FE_OVERFLOW);

      all_path_states.resize(si_max_path_states);

      for (size_t i = 1; i < si_max_path_states; i++)
        unused_path_states.push_back(&all_path_states[i]);

      SiState initial_state(0);
      initial_state.push_back(&all_path_states[0]);

      stack.push_back(initial_state);
    } else {
      stack.push_back(top());
      if (stack[size() - 2].size() > 0)
        top().num_parent_variables = stack[size() - 2][0]->size();
      else
        top().num_parent_variables = 0;
    }
  }

  /** Before entering a branching statement (if-else statement), the restrict algorithm needs to be executed
   *  to ensure adherence to the {@link si_max_path_states} limit.
   */
  void prepare_branch() {
    cumulative_instantiated_path_states += top().size();
    if (top().size() > 0)
      cumulative_num_variables += top()[0]->size();
    num_branches++;

    if (si_restrict_mode == si_merge_chaudhuri || si_restrict_mode == si_merge_chaudhuri_ignore_weights) {
      top().merge_chaudhuri();
    } else if (si_restrict_mode == si_merge_by_weights_only) {
      top().merge_by_weight();
    }
  };

  void enter_if(const SiPathWeights &weights) {
    assert(size() > 0);

    SiState new_states[2]{0, 0};

    top().clone_truncate(weights, new_states);

    for (size_t i = 0; i < 2; i++) {
      stack.push_back(new_states[i]);
    }
  }

  void enter_else(const SiPathWeights &weights) {
    for (auto it = top().begin(); it != top().end(); it++)
      (*it)->clean_up(top().num_parent_variables);

    swap_top_two();
  }

  void enter_loop() { enter_scope(); }

  void swap_top_two() {
    SiState then_state = top();
    then_state.copy_path_states_from(top(), false);
    SiState else_state = stack[size() - 2];
    else_state.copy_path_states_from(stack[size() - 2], false);
    pop();
    pop();
    stack.push_back(then_state);
    stack.back().copy_path_states_from(then_state, false);
    stack.push_back(else_state);
    stack.back().copy_path_states_from(else_state, false);
  }

  void break_() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.begin(); it != state.end(); it++)
      state.break_path_states.push_back(*it);
    state.clear();
  }

  void continue_() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.begin(); it != state.end(); it++)
      state.continue_path_states.push_back(*it);
    state.clear();
  }

  void return_() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.begin(); it != state.end(); it++)
      state.return_path_states.push_back(*it);
    state.clear();
  }

  /** @returns The SiState of the current scope. */
  SiState &top() { return stack.back(); }

  /** Remove the topmost element from the stack. */
  void pop() { stack.resize(size() - 1); }

  /** When exiting a branching statement, the two SiStates resulting from the
   *  two branches are the new states of the previous scope and the states of
   *  the branching scopes are removed.
   */
  void exit_if_else() {
    size_t final_size = top().size() + stack[size() - 2].size();

    auto &parent = stack[size() - 3];
    parent.clear();

    for (size_t i = 0; i < 2; i++) {
      auto &child = stack[size() - 1 - i];
      parent.copy_path_states_from(child);
    }

    pop();
    pop();

    assert(top().size() == final_size);

    max_instantiated_path_states = max(final_size, max_instantiated_path_states);
  }

  /** Exiting a scope means removing the current SiState from the stack
   *  along with any scoped variables.
   */
  void exit_scope() {
    for (auto it = top().begin(); it != top().end(); it++)
      (*it)->clean_up(top().num_parent_variables);

    if (size() > 1) {
      auto &parent = stack[size() - 2];
      parent.clear();
      parent.copy_path_states_from(top());
    }

    pop();
  }

  void exit_loop() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.break_path_states.begin(); it != state.break_path_states.end(); it++) {
      state.push_back(*it);
    }
    state.break_path_states.clear();

    exit_scope();
  }

  bool exit_loop_iteration() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.continue_path_states.begin(); it != state.continue_path_states.end(); it++) {
      state.push_back(*it);
    }
    state.continue_path_states.clear();

    for (auto it = top().begin(); it != top().end(); it++)
      (*it)->clean_up(top().num_parent_variables);

    return !top().empty();
  }

  void exit_function() {
    assert(!stack.empty());
    auto &state = top();
    for (auto it = state.return_path_states.begin(); it != state.return_path_states.end(); it++) {
      state.push_back(*it);
    }
    state.return_path_states.clear();
  }

  void print() {
    printf("\n--- printing stack ---\n");
    for (auto it = stack.begin(); it != stack.end(); it++) {
      printf("\nlevel %lu\n", it - stack.begin());
      it->print();
    }
    printf("\n--- end of stack ---\n\n");
  }
};

/** The global state of the Smooth Interpreter. */
SiStack si_stack;

/** @returns The weights of the tracked control flow paths in the current (top-most) scope. */
SiPathWeights::SiPathWeights(bool b) {
  for (size_t i = 0; i < si_stack.top().size(); i++)
    weights.push_back(b);
}

SiPathWeights operator&&(bool lhs, SiPathWeights rhs) {
  for (size_t idx = 0; idx < rhs.weights.size(); idx++)
    rhs.weights[idx] *= lhs;
  return rhs;
}
