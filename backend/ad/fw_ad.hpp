#pragma once

/** Operator-overloading implementation of forward-mode AD.
 *  Requires only a single pass by carrying along the adjoints for all program
 * inputs. Relies on the compiler for vectorization.
 *
 *  Copyright 2023 Philipp Andelfinger, Justin Kreikemeyer
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *    The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *    SOFTWARE.
 */

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <type_traits>
#include <vector>

static constexpr int int_ceil(double x) {
  const int i = x;
  return x > i ? i + 1 : i;
}

template <typename T> T ipow(T x, int p) {
  if (p == 0)
    return (T)1.0;

  T r = x;
  while (--p > 0)
    r *= x;
  return r;
}

#ifdef ENABLE_AD
#define ENABLE_AD_DEFAULT true
#else
#define ENABLE_AD_DEFAULT false
#endif

#define adouble_t fw_adouble<num_adj_, enable_ad_>

template <int num_adj_, bool enable_ad_ = ENABLE_AD_DEFAULT> class fw_adouble {
public:
  static const int num_adjoints = num_adj_; // for external access
  static const int min_adj_pool_size = 1024;
  static std::vector<double *> adj_pool;

  double val;
  double *adj;

  int single_adj_dim = -1;
  double single_adj = 0.0;

  double *alloc_adj(bool init = false) {
    if (adj_pool.empty())
      for (int i = 0; i < min_adj_pool_size; i++)
        adj_pool.push_back(new double[num_adj_]);

    double *r = adj_pool.back();
    adj_pool.pop_back();

    if (init) {
      for (int i = 0; i < num_adj_; i++)
        r[i] = 0.0;
    }

    return r;
  }

  void clear_adj() {
    assert(enable_ad_ || !adj);
    if (adj) {
      adj_pool.push_back(adj);
      adj = nullptr;
    }
    single_adj_dim = -1;
    single_adj = 0.0;
  }

  void init_val(double x) { val = x; };

#define ITER_ADJ(ADJ_EXPR)                                                                                                                 \
  for (int i = 0; enable_ad_ && i < num_adj_; i++) {                                                                                       \
    r.adj[i] = ADJ_EXPR;                                                                                                                   \
  }

  adouble_t &operator=(adouble_t &&other) {
    if (this == &other)
      return *this;

    clear_adj();

    val = other.val;
    adj = other.adj;
    other.adj = nullptr;

    single_adj_dim = other.single_adj_dim;
    single_adj = other.single_adj;

    return *this;
  }

  adouble_t &operator=(const adouble_t &other) {
    val = other.val;

    clear_adj();

    if (other.single_adj_dim != -1) {
      single_adj_dim = other.single_adj_dim;
      single_adj = other.single_adj;
      return *this;
    }

    if (other.adj) {
      init_full_adj();
      fw_adouble &r = *this;
      ITER_ADJ(other.adj[i]);
    }

    return *this;
  }

  fw_adouble(const fw_adouble &other) {
    adj = nullptr;
    single_adj_dim = -1;
    single_adj = 0.0;

    *this = other;
  }

  adouble_t &operator=(double other) {
    val = other;
    clear_adj();

    return *this;
  }

  fw_adouble(double x) {
    val = x;

    adj = nullptr;
    single_adj_dim = -1;
    single_adj = 0.0;
  }

  fw_adouble() : fw_adouble(0.0) {}

  ~fw_adouble() { clear_adj(); }

  void init_full_adj(bool init = false) {
    if (adj || !enable_ad_)
      return;

    adj = alloc_adj(init);

    if (single_adj_dim != -1) {
      adj[single_adj_dim] = single_adj;
      single_adj_dim = -1;
      single_adj = 0.0;
    }
  }

  double get_val() const { return val; }

  double get_adj(int k) const {
    if (!enable_ad_)
      return 0.0;

    if (k == single_adj_dim) {
      return single_adj;
    }

    return adj ? adj[k] : 0.0;
  }

  void set_initial_adj(int k, double a) {
    if (!enable_ad_)
      return;
    single_adj_dim = k;
    single_adj = a;
  }

  void set_adj(int k, double a) {
    if (!enable_ad_)
      return;

    init_full_adj(true);

    adj[k] = a;
  }

  bool has_adj() const {
    if (adj || single_adj_dim != -1)
      return true;

    return false;
  }

  adouble_t ipow(int p) const { return ::ipow(*this, p); }

#define FW_ADOUBLE_BINARY_OP(OP, ADJ_EXPR_A_A, ADJ_EXPR_A_D)                                                                               \
  adouble_t operator OP(const adouble_t &other) const {                                                                                    \
    fw_adouble r;                                                                                                                          \
    r.val = val OP other.val;                                                                                                              \
    if (single_adj_dim != -1 && single_adj_dim == other.single_adj_dim) {                                                                  \
      int i = single_adj_dim;                                                                                                              \
      r.single_adj_dim = single_adj_dim;                                                                                                   \
      r.single_adj = ADJ_EXPR_A_A;                                                                                                         \
      return r;                                                                                                                            \
    }                                                                                                                                      \
    r.init_full_adj(true);                                                                                                                 \
    ITER_ADJ(ADJ_EXPR_A_A);                                                                                                                \
    return r;                                                                                                                              \
  }                                                                                                                                        \
  adouble_t operator OP(double other) const {                                                                                              \
    fw_adouble r;                                                                                                                          \
    r.val = val OP other;                                                                                                                  \
    if (single_adj_dim != -1) {                                                                                                            \
      int i = single_adj_dim;                                                                                                              \
      r.single_adj_dim = single_adj_dim;                                                                                                   \
      r.single_adj = ADJ_EXPR_A_D;                                                                                                         \
      return r;                                                                                                                            \
    }                                                                                                                                      \
    r.init_full_adj(true);                                                                                                                 \
    ITER_ADJ(ADJ_EXPR_A_D);                                                                                                                \
    return r;                                                                                                                              \
  }

#define FW_ADOUBLE_ASSIGN_OP(ASSIGN_OP, BINARY_OP, ADJ_EXPR_A_A, ADJ_EXPR_A_D)                                                             \
  adouble_t operator ASSIGN_OP(const adouble_t &other) {                                                                                   \
    fw_adouble &r = *this;                                                                                                                 \
    if (single_adj_dim != -1 && single_adj_dim == other.single_adj_dim) {                                                                  \
      int i = single_adj_dim;                                                                                                              \
      r.single_adj_dim = single_adj_dim;                                                                                                   \
      r.single_adj = ADJ_EXPR_A_A;                                                                                                         \
      val ASSIGN_OP other.val;                                                                                                             \
      return r;                                                                                                                            \
    }                                                                                                                                      \
    init_full_adj(true);                                                                                                                   \
    ITER_ADJ(ADJ_EXPR_A_A);                                                                                                                \
    val ASSIGN_OP other.val;                                                                                                               \
    return *this;                                                                                                                          \
  }                                                                                                                                        \
  adouble_t operator ASSIGN_OP(double other) {                                                                                             \
    if (!adj && single_adj_dim == -1) {                                                                                                    \
      val ASSIGN_OP other;                                                                                                                 \
      return *this;                                                                                                                        \
    }                                                                                                                                      \
    fw_adouble &r = *this;                                                                                                                 \
    if (single_adj_dim != -1) {                                                                                                            \
      int i = single_adj_dim;                                                                                                              \
      r.single_adj_dim = single_adj_dim;                                                                                                   \
      r.single_adj = ADJ_EXPR_A_D;                                                                                                         \
    }                                                                                                                                      \
    if (adj)                                                                                                                               \
      ITER_ADJ(ADJ_EXPR_A_D);                                                                                                              \
    val ASSIGN_OP other;                                                                                                                   \
    return *this;                                                                                                                          \
  }

#define OWN_ADJ (i == single_adj_dim ? single_adj : (adj ? adj[i] : 0.0))
#define OTHER_ADJ (i == other.single_adj_dim ? other.single_adj : (other.adj ? other.adj[i] : 0.0))

#define ADD_ADJ (OWN_ADJ + OTHER_ADJ)
#define SUB_ADJ (OWN_ADJ - OTHER_ADJ)
#define MUL_ADJ (val * OTHER_ADJ + OWN_ADJ * other.val)
#define DIV_ADJ ((OWN_ADJ * other.val - val * OTHER_ADJ) / (other.val * other.val))

  FW_ADOUBLE_BINARY_OP(+, ADD_ADJ, OWN_ADJ);
  FW_ADOUBLE_BINARY_OP(-, SUB_ADJ, OWN_ADJ);
  FW_ADOUBLE_BINARY_OP(*, MUL_ADJ, OWN_ADJ *other);
  FW_ADOUBLE_BINARY_OP(/, DIV_ADJ, OWN_ADJ / other);

  FW_ADOUBLE_ASSIGN_OP(+=, +, ADD_ADJ, OWN_ADJ);
  FW_ADOUBLE_ASSIGN_OP(-=, -, SUB_ADJ, OWN_ADJ);
  FW_ADOUBLE_ASSIGN_OP(*=, *, MUL_ADJ, OWN_ADJ *other);
  FW_ADOUBLE_ASSIGN_OP(/=, /, DIV_ADJ, OWN_ADJ / other);

  adouble_t operator-() const {
    fw_adouble r;
    r.val = -val;

    if (!has_adj())
      return r;

    if (single_adj_dim != -1) {
      r.single_adj_dim = single_adj_dim;
      r.single_adj = -single_adj;
      return r;
    }

    r.init_full_adj(true);
    ITER_ADJ(-OWN_ADJ);

    return r;
  }

  bool operator<(double other) const { return val < other; };
  bool operator<=(double other) const { return val <= other; };
  bool operator>(double other) const { return val > other; };
  bool operator>=(double other) const { return val >= other; };
  bool operator==(double other) const { return val == other; }
  bool operator!=(double other) const { return val != other; }

  bool operator<(const adouble_t &other) const { return val < other.val; };
  bool operator<=(const adouble_t &other) const { return val <= other.val; };
  bool operator>(const adouble_t &other) const { return val > other.val; };
  bool operator>=(const adouble_t &other) const { return val >= other.val; };
  bool operator==(const adouble_t &other) const { return val == other.val; }
  bool operator!=(const adouble_t &other) const { return val != other.val; }

  explicit operator int() { return (int)val; }
};

template <int num_adj_, bool enable_ad_> bool operator<(double lhs, const adouble_t &rhs) { return rhs > lhs; };
template <int num_adj_, bool enable_ad_> bool operator<=(double lhs, const adouble_t &rhs) { return rhs >= lhs; };
template <int num_adj_, bool enable_ad_> bool operator>(double lhs, const adouble_t &rhs) { return rhs < lhs; };
template <int num_adj_, bool enable_ad_> bool operator>=(double lhs, const adouble_t &rhs) { return rhs <= lhs; };
template <int num_adj_, bool enable_ad_> bool operator==(double lhs, const adouble_t &rhs) { return rhs == lhs; }
template <int num_adj_, bool enable_ad_> bool operator!=(double lhs, const adouble_t &rhs) { return rhs != lhs; }

template <int num_adj_, bool enable_ad_> std::vector<double *> adouble_t::adj_pool;

template <int num_adj_, bool enable_ad_> adouble_t operator+(double lhs, const adouble_t &rhs) { return rhs + lhs; }
template <int num_adj_, bool enable_ad_> adouble_t operator-(double lhs, const adouble_t &rhs) { return -rhs + lhs; }
template <int num_adj_, bool enable_ad_> adouble_t operator*(double lhs, const adouble_t &rhs) { return rhs * lhs; }
template <int num_adj_, bool enable_ad_> adouble_t operator/(double lhs, const adouble_t &rhs) {
  adouble_t r;
  r.val = lhs / rhs.val;

  if (!rhs.has_adj())
    return r;

  if (rhs.single_adj_dim != -1) {
    r.single_adj_dim = rhs.single_adj_dim;
    r.single_adj = -lhs * rhs.single_adj / (rhs.val * rhs.val);
    return r;
  }

  r.init_full_adj();
  ITER_ADJ(-lhs * rhs.adj[i] / (rhs.val * rhs.val));
  return r;
}

#define FW_ADOUBLE_UNARY_OP(FUNC, ADJ_EXPR)                                                                                                \
  template <int num_adj_, bool enable_ad_> adouble_t FUNC(const adouble_t &x) {                                                            \
    adouble_t r;                                                                                                                           \
    r.val = FUNC(x.val);                                                                                                                   \
    if (x.single_adj_dim != -1) {                                                                                                          \
      int i = x.single_adj_dim;                                                                                                            \
      r.single_adj_dim = i;                                                                                                                \
      r.single_adj = ADJ_EXPR;                                                                                                             \
      return r;                                                                                                                            \
    }                                                                                                                                      \
    if (x.adj) {                                                                                                                           \
      r.init_full_adj();                                                                                                                   \
      ITER_ADJ(ADJ_EXPR);                                                                                                                  \
    }                                                                                                                                      \
    return r;                                                                                                                              \
  }

#define UNARY_OWN_ADJ (i == x.single_adj_dim ? x.single_adj : x.adj[i])

FW_ADOUBLE_UNARY_OP(exp, exp(x.val) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(sin, cos(x.val) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(cos, -sin(x.val) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(sqrt, (1.0 / (2.0 * sqrt(x.val))) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(log, (1.0 / x.val) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(erf, (2.0 * exp(-(x.val * x.val)) / sqrt(M_PI)) * UNARY_OWN_ADJ);
FW_ADOUBLE_UNARY_OP(tanh, ipow(1 - tanh(x.val), 2) * UNARY_OWN_ADJ);
