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

/* templates for eliminating duplicate code in the operation overloading of sdouble. */
#define ITER_STATE                                                                                                                         \
  assert(!si_stack.empty());                                                                                                               \
  auto &state = si_stack.top();                                                                                                            \
  for (auto it = state.begin(); it != state.end(); it++)

#define SDOUBLE_ASSIGN_OP(OP)                                                                                                              \
  sdouble &operator OP(const sdouble &other) {                                                                                             \
    ITER_STATE (is_temporary ? (*it)->get_temp(idx) : (*it)->get(idx)) OP(*it)->get_temp(other.idx);                                                                               \
    return *this;                                                                                                                          \
  }                                                                                                                                        \
  sdouble &operator OP(double other) {                                                                                                     \
    ITER_STATE (is_temporary ? (*it)->get_temp(idx) : (*it)->get(idx)) OP other;                                                                               \
    return *this;                                                                                                                          \
  }

#define SDOUBLE_BINARY_OP(OP)                                                                                                              \
  sdouble operator OP(const sdouble &other) const {                                                                                        \
    sdouble r(SiGaussian(), true);                                                                                                         \
    ITER_STATE (*it)->get_temp(r.idx) = (*it)->get_temp(idx) OP(*it)->get_temp(other.idx);                                                 \
    return r;                                                                                                                              \
  }                                                                                                                                        \
  sdouble operator OP(double other) const {                                                                                                \
    sdouble r(SiGaussian(), true);                                                                                                         \
    ITER_STATE (*it)->get_temp(r.idx) = (*it)->get_temp(idx) OP other;                                                                     \
    return r;                                                                                                                              \
  }

#define SDOUBLE_CMP_OP(OP, ARG_TYPE)                                                                                                       \
  SiPathWeights operator OP(ARG_TYPE other) {                                                                                              \
    SiPathWeights r;                                                                                                                       \
    ITER_STATE r.weights.push_back((*it)->get_temp(idx) OP other);                                                                         \
    return r;                                                                                                                              \
  }

#define SDOUBLE_UNARY_OP(OP)                                                                                                               \
  sdouble operator OP() const {                                                                                                            \
    sdouble r(SiGaussian(), true);                                                                                                         \
    ITER_STATE (*it)->get_temp(r.idx) = OP(*it)->get_temp(idx);                                                                            \
    return r;                                                                                                                              \
  }

#define SDOUBLE_WEIGHT_ASSIGN_OP(OP)                                                                                                       \
  sdouble &operator OP(const SiPathWeights &other) {                                                                                       \
    ITER_STATE {                                                                                                                           \
      adouble weight = other.weights[it - state.begin()];                                                                                  \
      SiGaussian g(weight, 0.0);                                                                                                           \
      (is_temporary ? (*it)->get_temp(idx) : (*it)->get(idx)) OP g;                                                                                                                \
    }                                                                                                                                      \
    return *this;                                                                                                                          \
  }

#define SDOUBLE_UNARY_FUNC_SAFE(FUNC, DERIV_SQUARED, ASSERTION)                                                                            \
  SiGaussian FUNC(SiGaussian &x) {                                                                                                         \
    assert(ASSERTION);                                                                                                                     \
    return {FUNC(x.m), x.v * DERIV_SQUARED};                                                                                               \
  }                                                                                                                                        \
                                                                                                                                           \
  sdouble FUNC(const sdouble &x) {                                                                                                         \
    sdouble r(SiGaussian(), true);                                                                                                         \
    ITER_STATE (*it)->get(r.idx) = FUNC((*it)->get_temp(x.idx));                                                                           \
    return r;                                                                                                                              \
  }

#define SDOUBLE_UNARY_FUNC(FUNC, DERIV_SQUARED) SDOUBLE_UNARY_FUNC_SAFE(FUNC, DERIV_SQUARED, true)

class sdouble;

/** A smooth double represented by a gaussian mixture distribution.
 *  It uses operator overloading to provide smooth versions of common C++ operators.
 */
class sdouble {
public:
  static size_t next_idx;
  size_t idx = 0;
  bool is_temporary = false;

  SDOUBLE_ASSIGN_OP(=)
  SDOUBLE_ASSIGN_OP(+=)
  SDOUBLE_ASSIGN_OP(-=)
  SDOUBLE_ASSIGN_OP(*=)
  SDOUBLE_ASSIGN_OP(/=)

  SDOUBLE_BINARY_OP(+)
  SDOUBLE_BINARY_OP(-)
  SDOUBLE_BINARY_OP(*)
  SDOUBLE_BINARY_OP(/)

  SDOUBLE_CMP_OP(<, double)
  SDOUBLE_CMP_OP(<, adouble)
  SDOUBLE_CMP_OP(<=, double)
  SDOUBLE_CMP_OP(<=, adouble)
  SDOUBLE_CMP_OP(==, double)
  SDOUBLE_CMP_OP(==, adouble)
  SDOUBLE_CMP_OP(!=, double)
  SDOUBLE_CMP_OP(!=, adouble)

  SDOUBLE_UNARY_OP(-)

  SDOUBLE_WEIGHT_ASSIGN_OP(=)
  SDOUBLE_WEIGHT_ASSIGN_OP(+=)
  SDOUBLE_WEIGHT_ASSIGN_OP(-=)
  SDOUBLE_WEIGHT_ASSIGN_OP(*=)
  SDOUBLE_WEIGHT_ASSIGN_OP(/=)

  SiPathWeights operator<(const sdouble& other) const { return (*this - other < (double)0.0); }
  SiPathWeights operator<=(const sdouble& other) const { return (*this - other <= (double)0.0); }
  SiPathWeights operator>(const sdouble& other) const { return (-*this < -other); }
  SiPathWeights operator>=(const sdouble& other) const { return (-*this <= -other); }

  /** Calculate the expected value of this variable across all tracked control flow paths. */
  adouble expectation() const {
    adouble r = 0.0;
    adouble sum_weight = 0.0;
    ITER_STATE {
      r += (*it)->weight * (*it)->get_temp(idx).m;
      sum_weight += (*it)->weight;
    }
    assert(sum_weight > 0.0 || si_stack.top().size() == 0);

    return (sum_weight > 0.0) ? (r / sum_weight) : (adouble)0.0;
  }

  void print() {
    ITER_STATE {
      // printf("state has length %ld\n", (*it)->size());
      SiGaussian g = (*it)->get_temp(idx);
      printf("(idx: %ld, weight: %.4g, m: %.2f, v: %.2g, sd: %.2f\n", idx, (*it)->weight.val, g.m.val, g.v.val, sqrt(g.v.val));
      if (print_adjoints) {
        printf("adjoints on weight: ");
        for (int i = 0; i < adouble::num_adjoints; i++)
          printf("%.4g ", (*it)->weight.get_adj(i));
        printf("adjoints on mean: ");
        for (int i = 0; i < adouble::num_adjoints; i++)
          printf("%.4g ", g.m.get_adj(i));
        printf(")\n");
      }
    }
  }

  /** Ensure that the value of this sdouble is within a certain range on all
   * tracked control flow paths. This is useful in cases where the smoothing results
   * in states that are not possible under the crisp semantics, but are required for
   * a correct execution.
   * @param max_variance The variance to assign to values on the control flow paths
   * that needed adjustment. Typically zero, because any larger variance would
   * result in invalid values being included.
   */
  void enforce_range(double lower, double upper, double max_variance = DBL_MAX) {
    ITER_STATE {
      SiGaussian &g = (*it)->get_temp(idx);
      if (g.m.val < lower)
        g.m = lower;

      if (g.m.val > upper)
        g.m = upper;

      if (g.v.val > max_variance)
        g.v = max_variance;
      //|| g.m.val > upper || g.v.val > max_variance) {
      //   *it = NULL;
      //}
    }
    // uses "state" from macro ITER_STATE
    //auto rm = remove_if(state.begin(), state.end(), [](SiPathState *p) { return p == NULL; });
    //state.path_states.erase(rm, state.end());
  }

  void enforce_range_hard(double lower, double upper, double max_variance = DBL_MAX) {
    ITER_STATE {
      SiGaussian &g = (*it)->get_temp(idx);
      if (g.m.val < lower || g.m.val > upper || g.v.val > max_variance) {
         *it = NULL;
      }
    }
    // uses "state" from macro ITER_STATE
    auto rm = remove_if(state.begin(), state.end(), [](SiPathState *p) { return p == NULL; });
    state.path_states.erase(rm, state.end());
  }

  /** Creates an "empty" sdouble. */
  sdouble(SiGaussian g, bool is_temporary) {
    this->is_temporary = is_temporary;
    idx = next_idx++;
  }

  /** Creates a new sdouble from g. */
  sdouble(SiGaussian g) {
    idx = next_idx++;
    ITER_STATE (*it)->get(idx) = g;
  }

  /** Allocates the parameter m as a temporary (constant). */
  sdouble(const adouble &m) {
    idx = next_idx++;
    ITER_STATE (*it)->get_temp(idx) = {m, 0.0};
  }

  /** @copydoc sdouble(const adouble &m) */
  sdouble(double m) : sdouble((adouble)m){};

  /** Copy constructor. Called e.g. when passing by-value (!).
   *  If the original was not marked as temporary, allocates a new
   *  sdouble.
   */
  sdouble(const sdouble &other) {
    if (other.is_temporary) {
      idx = other.idx;
      return;
    }
    idx = next_idx++;
    ITER_STATE (*it)->get(idx) = (*it)->get_temp(other.idx);
  }

  sdouble()
      : sdouble((SiGaussian){0.0, 0.0}){}; // calls sdouble(SiGaussian)
};

/** Calculate a (constant) integer power of the given {@link sdouble}.
 *  @param x The base.
 *  @param p The exponent.
 */
sdouble ipow(const sdouble& x, int p) {
  if (p == 0)
    return 1.0;

  sdouble r(SiGaussian(), true);
  r = x;
  while (--p > 0)
    r *= x;
  return r;
}

sdouble operator+(double lhs, sdouble& rhs) { return rhs + lhs; }
sdouble operator-(double lhs, sdouble& rhs) { return -rhs + lhs; }
sdouble operator*(double lhs, sdouble& rhs) { return rhs * lhs; }
sdouble operator/(double lhs, sdouble& rhs) { return sdouble(lhs) / rhs; }

SiPathWeights operator<(double lhs, sdouble& rhs) { return rhs > lhs; }
SiPathWeights operator<(adouble lhs, sdouble& rhs) { return rhs > lhs; }
SiPathWeights operator<=(double lhs, sdouble& rhs) { return rhs >= lhs; }
SiPathWeights operator<=(adouble lhs, sdouble& rhs) { return rhs >= lhs; }

SiPathWeights operator>(double lhs, sdouble& rhs) { return rhs < lhs; }
SiPathWeights operator>(adouble lhs, sdouble& rhs) { return rhs < lhs; }
SiPathWeights operator>=(double lhs, sdouble& rhs) { return rhs <= lhs; }
SiPathWeights operator>=(adouble lhs, sdouble& rhs) { return rhs <= lhs; }

sdouble operator+(double lhs, sdouble&& rhs) { return rhs + lhs; }
sdouble operator-(double lhs, sdouble&& rhs) { return -rhs + lhs; }
sdouble operator*(double lhs, sdouble&& rhs) { return rhs * lhs; }
sdouble operator/(double lhs, sdouble&& rhs) { return sdouble(lhs) / rhs; }

SiPathWeights operator<(double lhs, sdouble&& rhs) { return rhs > lhs; }
SiPathWeights operator<(adouble lhs, sdouble&& rhs) { return rhs > lhs; }
SiPathWeights operator<=(double lhs, sdouble&& rhs) { return rhs >= lhs; }
SiPathWeights operator<=(adouble lhs, sdouble&& rhs) { return rhs >= lhs; }

SiPathWeights operator>(double lhs, sdouble&& rhs) { return rhs < lhs; }
SiPathWeights operator>(adouble lhs, sdouble&& rhs) { return rhs < lhs; }
SiPathWeights operator>=(double lhs, sdouble&& rhs) { return rhs <= lhs; }
SiPathWeights operator>=(adouble lhs, sdouble&& rhs) { return rhs <= lhs; }

SDOUBLE_UNARY_FUNC_SAFE(sqrt, 1.0 / (4.0 * x.m), x.m > (adouble)0.0)
SDOUBLE_UNARY_FUNC_SAFE(log, 1.0 / (x.m * x.m), x.m > (adouble)0.0)
SDOUBLE_UNARY_FUNC(exp, exp(x.m).ipow(2))
SDOUBLE_UNARY_FUNC(sin, cos(x.m).ipow(2))
SDOUBLE_UNARY_FUNC(cos, sin(x.m).ipow(2))
SDOUBLE_UNARY_FUNC(tanh, (1 - tanh(x.m)).ipow(4))

uint64_t sdouble::next_idx = 0;
