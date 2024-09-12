/* Copyright 2023, 2024 Philipp Andelfinger, Justin Kreikemeyer

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the “Software”), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE. */

/** Operator-overloading implementation of forward-mode AD with basic sparsity
 * optimizations. Requires only a single pass by carrying along the tangents for
 * all program inputs. Relies on the compiler for vectorization.
 */

#pragma once

#include <algorithm>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <type_traits>
#include <unordered_set>
#include <vector>

extern uint64_t global_branch_id;
extern bool in_branch;
const uint64_t initial_global_branch_id = 11061421359639307453UL;

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

#define adouble_t fw_adouble<num_tang_, enable_ad_>

template <int num_tang_, bool enable_ad_ = ENABLE_AD_DEFAULT> class fw_adouble {
public:
  static const int num_tangents = num_tang_; // for external access

#if DGO_FORK_LIMIT != 0
  static uint64_t set_counter; // for global ordering

  pair<uint64_t, uint64_t> set_at = {initial_global_branch_id, 0};
#endif

  double val;
  double tang[num_tang_];

  int tang_dim = -1; // -1: none, INT_MAX: all

#if DGO_FORK_LIMIT != 0
  void mark_set() {
    if (!enable_ad_ || branch_level == 0)
      return;

    set_at = {global_branch_id, set_counter++};
  }

  void mark_set(const adouble_t &other) {
    if (!enable_ad_ || global_branch_id == initial_global_branch_id)
      return;

    if (branch_level == 0) {
      set_at.first = set_at.second > other.set_at.second ? set_at.first
                                                         : other.set_at.first;
      set_at.second = set_counter++;
      return;
    }

    set_at = {global_branch_id, set_counter++};
  }
#else
  void mark_set() { return; }
  void mark_set(const adouble_t &other) { return; };
#endif

  void clear_tang() {
    if (!enable_ad_)
      return;

    tang_dim = -1;
  }

  void init_val(double x) { val = x; };

  void clone_tang(const adouble_t &other) {
    tang_dim = other.tang_dim;
    if (other.has_full_tang()) {
      for (int t = 0; t < num_tang_; t++)
        tang[t] = other.tang[t];
    } else {
      tang[tang_dim] = other.tang[tang_dim];
    }
  }

  void become(const adouble_t &&other) {
    *this = other;
    // XXX set_at not updated here because it's not needed for our use case
  }

  adouble_t &operator=(adouble_t &&other) {
    *this = other;
    mark_set(other);
    return *this;
  }

#define ITER_TANG(TANG_EXPR)                                                   \
  for (int i = 0; enable_ad_ && i < num_tang_; i++) {                          \
    r.tang[i] = TANG_EXPR;                                                     \
  }

  adouble_t &operator=(const adouble_t &other) {
    val = other.val;
    clone_tang(other);
    mark_set(other);
    return *this;
  }

  fw_adouble(const fw_adouble &other) { *this = other; }

  adouble_t &operator=(double other) {
    if (has_tang() || other != val)
      mark_set();

    val = other;
    clear_tang();

    return *this;
  }

  fw_adouble(double x) {
    val = x;
    tang_dim = -1;

    mark_set();
  }

  fw_adouble() : fw_adouble(0.0) {}

  ~fw_adouble() {
    if (!enable_ad_)
      return;
    clear_tang();
  }

  void init_full_tang(bool zero_out = false, bool force = false) {
    if (has_full_tang() || (!enable_ad_ && !force))
      return;

    double sparse_tang_val;
    if (has_sparse_tang())
      sparse_tang_val = tang[tang_dim];

    if (zero_out)
      for (int t = 0; t < num_tang_; t++)
        tang[t] = 0.0;

    if (has_sparse_tang())
      tang[tang_dim] = sparse_tang_val;

    tang_dim = INT_MAX;
  }

  double get_val() const { return val; }

  double get_tang(int k) const {
    if (has_full_tang() || (has_sparse_tang() && k == tang_dim))
      return tang[k];
    return 0.0;
  }

  void set_tang(int k, double a) {
    if (!enable_ad_)
      return;

    if (!has_tang()) {
      tang_dim = k;
      tang[k] = a;
      return;
    }

    init_full_tang(true);

    tang[k] = a;
  }

  bool has_tang() const { return tang_dim != -1; }
  bool has_sparse_tang() const {
    return tang_dim != -1 && tang_dim != INT_MAX;
  };
  bool has_full_tang() const { return tang_dim == INT_MAX; };

  adouble_t ipow(int p) const { return ::ipow(*this, p); }

  double dummy_op(const adouble_t &a, const double &b) const { return a.val; };
  double dummy_op(const adouble_t &a, const adouble_t &b) const { return a.val; };


#define FW_ADOUBLE_BINARY_OP(IS_INFIX, OP_NAME, INFIX_OP, PREFIX_OP, TANG_EXPR_A_A, TANG_EXPR_A_D)        \
  adouble_t OP_NAME(const adouble_t &other) const {                            \
    fw_adouble r;                                                              \
    r.val = IS_INFIX ? val INFIX_OP other.val : PREFIX_OP(val, other.val);     \
    r.mark_set(other);                                                         \
    if (!has_tang() && !other.has_tang())                                      \
      return r;                                                                \
    bool only_one_tang_dim = ((!has_tang() && other.has_sparse_tang()) ||      \
                              (!other.has_tang() && has_sparse_tang()));       \
    if (only_one_tang_dim ||                                                   \
        (has_sparse_tang() && other.tang_dim == tang_dim)) {                   \
      int i = has_sparse_tang() ? tang_dim : other.tang_dim;                   \
      r.tang[i] = TANG_EXPR_A_A;                                               \
      r.tang_dim = i;                                                          \
      return r;                                                                \
    }                                                                          \
    r.init_full_tang(true);                                                    \
    ITER_TANG(TANG_EXPR_A_A);                                                  \
    return r;                                                                  \
  }                                                                            \
  adouble_t OP_NAME(double other) const {                                      \
    fw_adouble r;                                                              \
    r.val = IS_INFIX ? val INFIX_OP other : PREFIX_OP(val, other);             \
    r.mark_set(*this);                                                         \
                                                                               \
    if (!has_tang())                                                           \
      return r;                                                                \
                                                                               \
    if (has_sparse_tang()) {                                                   \
      int i = tang_dim;                                                        \
      r.tang[i] = TANG_EXPR_A_D;                                               \
      r.tang_dim = tang_dim;                                                   \
      return r;                                                                \
    }                                                                          \
    r.init_full_tang(true);                                                    \
    ITER_TANG(TANG_EXPR_A_D);                                                  \
    return r;                                                                  \
  }

#define FW_ADOUBLE_ASSIGN_OP(ASSIGN_OP, BINARY_OP, TANG_EXPR_A_A,              \
                             TANG_EXPR_A_D, ITER_TANG_EXPR_A_D)                \
  void operator ASSIGN_OP(const adouble_t &other) {                            \
    mark_set(other);                                                           \
    fw_adouble &r = *this;                                                     \
                                                                               \
    if (!has_tang() && !other.has_tang()) {                                    \
      val ASSIGN_OP other.val;                                                 \
      return;                                                                  \
    }                                                                          \
    bool only_one_tang_dim = ((!has_tang() && other.has_sparse_tang()) ||      \
                              (!other.has_tang() && has_sparse_tang()));       \
    if (only_one_tang_dim ||                                                   \
        (has_sparse_tang() && other.tang_dim == tang_dim)) {                   \
      int i = has_sparse_tang() ? tang_dim : other.tang_dim;                   \
      r.tang[i] = TANG_EXPR_A_A;                                      \
      r.tang_dim = i;                                                          \
      val ASSIGN_OP other.val;                                                 \
      return;                                                                  \
    }                                                                          \
    init_full_tang(true);                                                      \
    ITER_TANG(TANG_EXPR_A_A);                                                  \
    val ASSIGN_OP other.val;                                                   \
    return;                                                                    \
  }                                                                            \
  void operator ASSIGN_OP(double other) {                                      \
    mark_set();                                                                \
    if (!has_tang()) {                                                         \
      val ASSIGN_OP other;                                                     \
      return;                                                                  \
    }                                                                          \
    fw_adouble &r = *this;                                                     \
    if (has_sparse_tang()) {                                                   \
      int i = tang_dim;                                                        \
      r.tang[i] = TANG_EXPR_A_D;                                      \
      r.tang_dim = tang_dim;                                                   \
    }                                                                          \
    if (has_full_tang())                                                       \
      ITER_TANG_EXPR_A_D;                                                      \
    val ASSIGN_OP other;                                                       \
    return;                                                                    \
  }

#define OWN_TANG get_tang(i)
#define OTHER_TANG other.get_tang(i)

#define ADD_TANG (OWN_TANG + OTHER_TANG)
#define SUB_TANG (OWN_TANG - OTHER_TANG)
#define MUL_TANG (val * OTHER_TANG + OWN_TANG * other.val)
#define DIV_TANG                                                               \
  ((OWN_TANG * other.val - val * OTHER_TANG) / (other.val * other.val))
#define ATAN2_TANG                                                             \
  ((-OTHER_TANG * val + OWN_TANG * other.val) /                                \
   ((val * val) + (other.val * other.val)))

  FW_ADOUBLE_BINARY_OP(true, operator+, +, dummy_op, ADD_TANG, OWN_TANG);
  FW_ADOUBLE_BINARY_OP(true, operator-, -, dummy_op, SUB_TANG, OWN_TANG);
  FW_ADOUBLE_BINARY_OP(true, operator*, *, dummy_op, MUL_TANG, OWN_TANG *other);
  FW_ADOUBLE_BINARY_OP(true, operator/, /, dummy_op, DIV_TANG, OWN_TANG / other);

  FW_ADOUBLE_BINARY_OP(false, atan2, +, std::atan2, ATAN2_TANG,
                       (OWN_TANG * other) / ((val * val) + (other * other)));

  FW_ADOUBLE_BINARY_OP(false, powc, +, std::pow, assert(false),
                       OWN_TANG *(other *std::pow(val, other - 1)));

  FW_ADOUBLE_ASSIGN_OP(+=, +, ADD_TANG, OWN_TANG, ); // nothing to do here
  FW_ADOUBLE_ASSIGN_OP(-=, -, SUB_TANG, OWN_TANG, ); // nothing to do here
  FW_ADOUBLE_ASSIGN_OP(*=, *, MUL_TANG, OWN_TANG *other,
                       ITER_TANG(OWN_TANG *other));
  FW_ADOUBLE_ASSIGN_OP(/=, /, DIV_TANG, OWN_TANG / other,
                       ITER_TANG(OWN_TANG / other));

  adouble_t operator-() const {
    fw_adouble r;
    r.val = -val;

    r.mark_set(*this);

    if (!has_tang())
      return r;

    if (has_sparse_tang()) {
      r.tang_dim = tang_dim;
      r.tang[tang_dim] = -r.tang[tang_dim];
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

template <int num_tang_, bool enable_ad_>
bool operator<(double lhs, const adouble_t &rhs) {
  return rhs > lhs;
};
template <int num_tang_, bool enable_ad_>
bool operator<=(double lhs, const adouble_t &rhs) {
  return rhs >= lhs;
};
template <int num_tang_, bool enable_ad_>
bool operator>(double lhs, const adouble_t &rhs) {
  return rhs < lhs;
};
template <int num_tang_, bool enable_ad_>
bool operator>=(double lhs, const adouble_t &rhs) {
  return rhs <= lhs;
};
template <int num_tang_, bool enable_ad_>
bool operator==(double lhs, const adouble_t &rhs) {
  return rhs == lhs;
}
template <int num_tang_, bool enable_ad_>
bool operator!=(double lhs, const adouble_t &rhs) {
  return rhs != lhs;
}

#if DGO_FORK_LIMIT != 0
template <int num_tang_, bool enable_ad_> uint64_t adouble_t::set_counter = 1;
#endif

template <int num_tang_, bool enable_ad_>
adouble_t operator+(double lhs, const adouble_t &rhs) {
  return rhs + lhs;
}
template <int num_tang_, bool enable_ad_>
adouble_t operator-(double lhs, const adouble_t &rhs) {
  adouble_t r;
  r.val = lhs - rhs.val;

  if (!rhs.has_tang())
    return r;

  if (rhs.has_sparse_tang()) {
    r.tang_dim = rhs.tang_dim;
    r.tang[r.tang_dim] = -rhs.tang[rhs.tang_dim];
    return r;
  }

  r.init_full_tang();
  ITER_TANG(-rhs.tang[i]);
  return r;
}

template <int num_tang_, bool enable_ad_>
adouble_t operator*(double lhs, const adouble_t &rhs) {
  return rhs * lhs;
}
template <int num_tang_, bool enable_ad_>
adouble_t operator/(double lhs, const adouble_t &rhs) {
  adouble_t r;
  r.val = lhs / rhs.val;

  if (!rhs.has_tang())
    return r;

  if (rhs.has_sparse_tang()) {
    r.tang_dim = rhs.tang_dim;
    r.tang[r.tang_dim] = -lhs * rhs.tang[rhs.tang_dim] / (rhs.val * rhs.val);
    return r;
  }

  r.init_full_tang();
  ITER_TANG(-lhs * rhs.tang[i] / (rhs.val * rhs.val));
  return r;
}

#define FW_ADOUBLE_UNARY_OP(FUNC, TANG_EXPR)                                   \
  template <int num_tang_, bool enable_ad_>                                    \
  adouble_t FUNC(const adouble_t &x) {                                         \
    adouble_t r;                                                               \
    r.val = FUNC(x.val);                                                       \
    r.mark_set(x);                                                             \
    if (x.has_sparse_tang()) {                                                 \
      int i = x.tang_dim;                                                      \
      r.tang_dim = i;                                                          \
      r.tang[i] = TANG_EXPR;                                                   \
      return r;                                                                \
    }                                                                          \
    if (x.tang) {                                                              \
      r.init_full_tang();                                                      \
      ITER_TANG(TANG_EXPR);                                                    \
    }                                                                          \
    return r;                                                                  \
  }

#define UNARY_OWN_TANG x.get_tang(i)

FW_ADOUBLE_UNARY_OP(exp, r.val *UNARY_OWN_TANG); // r.val == exp(x.val)
FW_ADOUBLE_UNARY_OP(sin, cos(x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(cos, -sin(x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(sqrt, (1.0 / (2.0 * r.val)) *
                              UNARY_OWN_TANG); // r.val == sqrt(x.val)
FW_ADOUBLE_UNARY_OP(log, (1.0 / x.val) * UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(erf, (2.0 * exp(-(x.val * x.val)) / sqrt(M_PI)) *
                             UNARY_OWN_TANG);
FW_ADOUBLE_UNARY_OP(tanh, ipow(1 - r.val, 2) *
                              UNARY_OWN_TANG); // r.val == tanh(x.val)

template <int num_tang_, bool enable_ad_>
adouble_t atan2(const adouble_t &a, const adouble_t &b) {
  return a.atan2(b);
}
template <int num_tang_, bool enable_ad_>
adouble_t powc(const adouble_t &a, const double b) {
  return a.powc(b);
}

typedef fw_adouble<num_inputs> adouble;
