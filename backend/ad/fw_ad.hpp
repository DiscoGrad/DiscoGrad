/* Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer
  
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software
  and associated documentation files (the “Software”), to deal in the Software without
  restriction, including without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:
   
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
    ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE. */

/** Operator-overloading implementation of forward-mode AD with basic sparsity optimizations.
 *  Requires only a single pass by carrying along the tangents for all program inputs.
 *  Relies on the compiler for vectorization.
 */

#pragma once

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <type_traits>
#include <iostream>
#include <unordered_set>
#include "globals.hpp"

#define TANG_POOL_SIZE 10000000

extern uint64_t global_branch_id;
extern bool in_branch;

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

double op_add(const double& a, const double& b) { return a + b; };
double op_sub(const double& a, const double& b) { return a - b; };
double op_mul(const double& a, const double& b) { return a * b; };
double op_div(const double& a, const double& b) { return a / b; };

template <typename T, int N, int M>
struct svec {
  double** items;
  int size = 0;
  svec() {
    items = new double *[N + 1];
    double *flat_pool = new double[N * M];
    for (int i = 0; i < N; i++)
      items[i] = &flat_pool[i * M];

    size = N;
  }

  double *pop_back() {
    assert(size > 0);
    size--;
    return items[size];
  }

  void push_back(double *p) {
    items[size] = p;
    size += !!p; // increment only if p is not nullptr
  }
};

#ifdef ENABLE_AD
#define ENABLE_AD_DEFAULT true
#else
#define ENABLE_AD_DEFAULT false
#endif

#define adouble_t fw_adouble<num_tang_, enable_ad_>

template <int num_tang_, bool enable_ad_ = ENABLE_AD_DEFAULT> class fw_adouble {
public:
  static const int num_tangents = num_tang_; // for external access
  static svec<double *, TANG_POOL_SIZE, num_tang_> tang_pool;

#if DGO_FORK_LIMIT != 0
  static uint64_t set_counter; // for global ordering

  pair<uint64_t, uint64_t> set_at = { initial_global_branch_id, 0 };
#endif

  double val;
  double *tang;

  int single_tang_dim = -1;
  double single_tang = 0.0;

#if DGO_FORK_LIMIT != 0
  void mark_set() {
    if (!enable_ad_ || branch_level == 0)
      return;

    set_at = { global_branch_id, set_counter++ };
  }

  void mark_set(const adouble_t &other) {
    if (!enable_ad_ || global_branch_id == initial_global_branch_id)
      return;

    if (branch_level == 0) {
      set_at.first = set_at.second > other.set_at.second ? set_at.first : other.set_at.first;
      set_at.second = set_counter++;
      return;
    }

    set_at = { global_branch_id, set_counter++ };
  }
#else
void mark_set() { return; }
void mark_set(const adouble_t &other) { return; };
#endif

  double *alloc_tang(bool init = false) {
    double *r = tang_pool.pop_back();

    if (init) {
      for (int i = 0; i < num_tang_; i++)
        r[i] = 0.0;
    }

    return r;
  }

  void clear_tang() {
    if (!enable_ad_)
      return;
    single_tang_dim = -1;
    single_tang = 0.0;

    if (num_tang_ <= 1)
      return;

    tang_pool.push_back(tang); // tang == nullptr is handled in push_back()
    tang = nullptr;
  }

  void init_val(double x) { val = x; };

  void become(adouble_t &&other) {
    val = other.val;

    // XXX set_at not updated here because it's not needed for our use case

    if (other.single_tang_dim != -1) {
      single_tang_dim = other.single_tang_dim;
      single_tang = other.single_tang;
      return;
    }

    tang = other.tang;

    other.tang = nullptr;
  }

  adouble_t &operator=(adouble_t &&other) {
    if (this == &other)
      return *this;

    clear_tang();

    mark_set(other);

    val = other.val;
    tang = other.tang;
    other.tang = nullptr;

    single_tang_dim = other.single_tang_dim;
    single_tang = other.single_tang;

    return *this;
  }

#define ITER_TANG(TANG_EXPR)                                                                                                 \
  for (int i = 0; enable_ad_ && i < num_tang_; i++) {                                                                        \
    r.tang[i] = TANG_EXPR;                                                                                                   \
  }

  adouble_t &operator=(const adouble_t &other) {
    val = other.val;

    if (!enable_ad_)
      return *this;

    clear_tang();

    mark_set(other);

    if (other.single_tang_dim != -1) {
      single_tang_dim = other.single_tang_dim;
      single_tang = other.single_tang;
      return *this;
    }

    if (num_tang_ > 1 && other.tang) {
      init_full_tang();
      fw_adouble &r = *this;
      ITER_TANG(other.tang[i]);
    }

    return *this;
  }

  fw_adouble(const fw_adouble &other) {
    tang = nullptr;
    single_tang_dim = -1;
    single_tang = 0.0;

    *this = other;

    mark_set(other);
  }

  adouble_t &operator=(double other) {
    if (has_tang() || other != val)
      mark_set();

    val = other;
    clear_tang();

    return *this;
  }

  fw_adouble(double x) {
    val = x;

    tang = nullptr;
    single_tang_dim = -1;
    single_tang = 0.0;

    mark_set();
  }

  fw_adouble() : fw_adouble(0.0) { }

  ~fw_adouble() {
    if (!enable_ad_)
      return;
    clear_tang();
  }

  void init_full_tang(bool init = false, bool force = false) {
    if (tang || (!enable_ad_ && !force))
      return;

    tang = alloc_tang(init);

    if (single_tang_dim != -1) {
      tang[single_tang_dim] = single_tang;
      single_tang_dim = -1;
      single_tang = 0.0;
    }
  }

  double get_val() const { return val; }

  double get_tang(int k) const {
    if (k == single_tang_dim) {
      return single_tang;
    }

    return tang ? tang[k] : 0.0;
  }

  void set_initial_tang(int k, double a) {
    if (!enable_ad_)
      return;
    single_tang_dim = k;
    single_tang = a;
  }

  void set_tang(int k, double a) {
    if (!enable_ad_)
      return;

    init_full_tang(true);

    tang[k] = a;
  }

  bool has_tang() const { return tang != nullptr || single_tang_dim != -1; }

  adouble_t ipow(int p) const { return ::ipow(*this, p); }


#define FW_ADOUBLE_BINARY_OP(OP, OP_CALL, TANG_EXPR_A_A, TANG_EXPR_A_D)                                                      \
  adouble_t OP(const adouble_t &other) const {                                                                               \
    fw_adouble r;                                                                                                            \
    r.val = OP_CALL(val, other.val);                                                                                         \
    r.mark_set(other);                                                                                                       \
    if (!has_tang() && !other.has_tang())                                                                                    \
      return r;                                                                                                              \
                                                                                                                             \
    bool only_one_tang_dim = ((!has_tang() && other.single_tang_dim != -1) || (!other.has_tang() && single_tang_dim != -1)); \
    if (only_one_tang_dim || (single_tang_dim != -1 && other.single_tang_dim == single_tang_dim)) {                          \
      int i = (single_tang_dim != -1) ? single_tang_dim : other.single_tang_dim;                                             \
      r.single_tang_dim = i;                                                                                                 \
      r.single_tang = TANG_EXPR_A_A;                                                                                         \
      return r;                                                                                                              \
    }                                                                                                                        \
    r.init_full_tang(true);                                                                                                  \
    ITER_TANG(TANG_EXPR_A_A);                                                                                                \
    return r;                                                                                                                \
  }                                                                                                                          \
  adouble_t OP(double other) const {                                                                                         \
    fw_adouble r;                                                                                                            \
    r.val = OP_CALL(val, other);                                                                                             \
    r.mark_set(*this);                                                                                                       \
                                                                                                                             \
    if (!has_tang())                                                                                                         \
      return r;                                                                                                              \
                                                                                                                             \
    if (single_tang_dim != -1) {                                                                                             \
      int i = single_tang_dim;                                                                                               \
      r.single_tang_dim = single_tang_dim;                                                                                   \
      r.single_tang = TANG_EXPR_A_D;                                                                                         \
      return r;                                                                                                              \
    }                                                                                                                        \
    r.init_full_tang(true);                                                                                                  \
    ITER_TANG(TANG_EXPR_A_D);                                                                                                \
    return r;                                                                                                                \
  }

#define FW_ADOUBLE_ASSIGN_OP(ASSIGN_OP, BINARY_OP, TANG_EXPR_A_A, TANG_EXPR_A_D, ITER_TANG_EXPR_A_D)                         \
  void operator ASSIGN_OP(const adouble_t &other) {                                                                          \
    mark_set(other);                                                                                                         \
    fw_adouble &r = *this;                                                                                                   \
                                                                                                                             \
    if (!has_tang() && !other.has_tang()) {                                                                                  \
      val ASSIGN_OP other.val;                                                                                               \
      return;                                                                                                                \
    }                                                                                                                        \
    bool only_one_tang_dim = ((!has_tang() && other.single_tang_dim != -1) || (!other.has_tang() && single_tang_dim != -1)); \
    if (only_one_tang_dim || (single_tang_dim != -1 && other.single_tang_dim == single_tang_dim)) {                          \
      int i = single_tang_dim != -1 ? single_tang_dim : other.single_tang_dim;                                               \
      r.single_tang_dim = i;                                                                                                 \
      r.single_tang = TANG_EXPR_A_A;                                                                                         \
      val ASSIGN_OP other.val;                                                                                               \
      return;                                                                                                                \
    }                                                                                                                        \
    init_full_tang(true);                                                                                                    \
    ITER_TANG(TANG_EXPR_A_A);                                                                                                \
    val ASSIGN_OP other.val;                                                                                                 \
    return;                                                                                                                  \
  }                                                                                                                          \
  void operator ASSIGN_OP(double other) {                                                                                    \
    mark_set();                                                                                                              \
    if (!has_tang()) {                                                                                                       \
      val ASSIGN_OP other;                                                                                                   \
      return;                                                                                                                \
    }                                                                                                                        \
    fw_adouble &r = *this;                                                                                                   \
    if (single_tang_dim != -1) {                                                                                             \
      int i = single_tang_dim;                                                                                               \
      r.single_tang_dim = single_tang_dim;                                                                                   \
      r.single_tang = TANG_EXPR_A_D;                                                                                         \
    }                                                                                                                        \
    if (tang)                                                                                                                \
      ITER_TANG_EXPR_A_D;                                                                                                    \
    val ASSIGN_OP other;                                                                                                     \
    return;                                                                                                                  \
  }

#define OWN_TANG (i == single_tang_dim ? single_tang : (tang ? tang[i] : 0.0))
#define OTHER_TANG (i == other.single_tang_dim ? other.single_tang : (other.tang ? other.tang[i] : 0.0))

#define ADD_TANG (OWN_TANG + OTHER_TANG)
#define SUB_TANG (OWN_TANG - OTHER_TANG)
#define MUL_TANG (val * OTHER_TANG + OWN_TANG * other.val)
#define DIV_TANG ((OWN_TANG * other.val - val * OTHER_TANG) / (other.val * other.val))
#define ATAN2_TANG ((-OTHER_TANG * val + OWN_TANG * other.val) / ((val * val) + (other.val * other.val)))

  FW_ADOUBLE_BINARY_OP(operator+, op_add, ADD_TANG, OWN_TANG);
  FW_ADOUBLE_BINARY_OP(operator-, op_sub, SUB_TANG, OWN_TANG);
  FW_ADOUBLE_BINARY_OP(operator*, op_mul, MUL_TANG, OWN_TANG * other);
  FW_ADOUBLE_BINARY_OP(operator/, op_div, DIV_TANG, OWN_TANG / other);

  FW_ADOUBLE_BINARY_OP(atan2, std::atan2, ATAN2_TANG, (OWN_TANG * other) / ((val * val) + (other * other)));

  FW_ADOUBLE_BINARY_OP(powc, std::pow, assert(false), OWN_TANG * (other * std::pow(val, other - 1)));

  FW_ADOUBLE_ASSIGN_OP(+=, +, ADD_TANG, OWN_TANG, ); // nothing to do here
  FW_ADOUBLE_ASSIGN_OP(-=, -, SUB_TANG, OWN_TANG, ); // nothing to do here
  FW_ADOUBLE_ASSIGN_OP(*=, *, MUL_TANG, OWN_TANG * other, ITER_TANG(OWN_TANG * other));
  FW_ADOUBLE_ASSIGN_OP(/=, /, DIV_TANG, OWN_TANG / other, ITER_TANG(OWN_TANG / other));

  adouble_t operator-() const {
    fw_adouble r;
    r.val = -val;

    r.mark_set(*this);

    if (!has_tang())
      return r;

    if (single_tang_dim != -1) {
      r.single_tang_dim = single_tang_dim;
      r.single_tang = -single_tang;
      return r;
    }

    r.init_full_tang(true);
    ITER_TANG(-OWN_TANG);

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

template <int num_tang_, bool enable_ad_> bool operator<(double lhs, const adouble_t &rhs) { return rhs > lhs; };
template <int num_tang_, bool enable_ad_> bool operator<=(double lhs, const adouble_t &rhs) { return rhs >= lhs; };
template <int num_tang_, bool enable_ad_> bool operator>(double lhs, const adouble_t &rhs) { return rhs < lhs; };
template <int num_tang_, bool enable_ad_> bool operator>=(double lhs, const adouble_t &rhs) { return rhs <= lhs; };
template <int num_tang_, bool enable_ad_> bool operator==(double lhs, const adouble_t &rhs) { return rhs == lhs; }
template <int num_tang_, bool enable_ad_> bool operator!=(double lhs, const adouble_t &rhs) { return rhs != lhs; }

#if DGO_FORK_LIMIT != 0
template <int num_tang_, bool enable_ad_> uint64_t adouble_t::set_counter = 1;
#endif

template <int num_tang_, bool enable_ad_> adouble_t operator+(double lhs, const adouble_t &rhs) { return rhs + lhs; }
template <int num_tang_, bool enable_ad_> adouble_t operator-(double lhs, const adouble_t &rhs) {
  adouble_t r;
  r.val = lhs - rhs.val;

  if (!rhs.has_tang())
    return r;

  if (rhs.single_tang_dim != -1) {
    r.single_tang_dim = rhs.single_tang_dim;
    r.single_tang = -rhs.single_tang;
    return r;
  }

  r.init_full_tang();
  ITER_TANG(-rhs.tang[i]);
  return r;
}

template <int num_tang_, bool enable_ad_> adouble_t operator*(double lhs, const adouble_t &rhs) { return rhs * lhs; }
template <int num_tang_, bool enable_ad_> adouble_t operator/(double lhs, const adouble_t &rhs) {
  adouble_t r;
  r.val = lhs / rhs.val;

  if (!rhs.has_tang())
    return r;

  if (rhs.single_tang_dim != -1) {
    r.single_tang_dim = rhs.single_tang_dim;
    r.single_tang = -lhs * rhs.single_tang / (rhs.val * rhs.val);
    return r;
  }

  r.init_full_tang();
  ITER_TANG(-lhs * rhs.tang[i] / (rhs.val * rhs.val));
  return r;
}

#define FW_ADOUBLE_UNARY_OP(FUNC, TANG_EXPR)                                                                                 \
  template <int num_tang_, bool enable_ad_> adouble_t FUNC(const adouble_t &x) {                                             \
    adouble_t r;                                                                                                             \
    r.val = FUNC(x.val);                                                                                                     \
    r.mark_set(x);                                                                                                           \
    if (x.single_tang_dim != -1) {                                                                                           \
      int i = x.single_tang_dim;                                                                                             \
      r.single_tang_dim = i;                                                                                                 \
      r.single_tang = TANG_EXPR;                                                                                             \
      return r;                                                                                                              \
    }                                                                                                                        \
    if (x.tang) {                                                                                                            \
      r.init_full_tang();                                                                                                    \
      ITER_TANG(TANG_EXPR);                                                                                                  \
    }                                                                                                                        \
    return r;                                                                                                                \
  }

#define UNARY_OWN_TANG (i == x.single_tang_dim ? x.single_tang : x.tang[i])

FW_ADOUBLE_UNARY_OP(exp, r.val * UNARY_OWN_TANG); // r.val == exp(x.val)
FW_ADOUBLE_UNARY_OP(sin, cos(x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(cos, -sin(x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(sqrt, (1.0 / (2.0 * r.val)) * UNARY_OWN_TANG); // r.val == sqrt(x.val)
FW_ADOUBLE_UNARY_OP(log, (1.0 / x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(erf, (2.0 * exp(-(x.val * x.val)) / sqrt(M_PI)) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(tanh, ipow(1 - r.val, 2) * UNARY_OWN_TANG); // r.val == tanh(x.val)

template <int num_tang_, bool enable_ad_> adouble_t atan2(const adouble_t &a, const adouble_t &b) { return a.atan2(b); }
template <int num_tang_, bool enable_ad_> adouble_t powc(const adouble_t &a, const double b) { return a.powc(b); }
template <int num_tang_, bool enable_ad_> svec<double *, TANG_POOL_SIZE, num_tang_> adouble_t::tang_pool;

typedef fw_adouble<num_inputs> adouble;

