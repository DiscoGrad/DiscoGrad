#pragma once

#ifdef ENABLE_AD
#define ENABLE_AD_DEFAULT true
#else
#define ENABLE_AD_DEFAULT false
#endif

template <int num_val_, int num_tang_> class avec {
public:
  typedef avec<num_val_, num_tang_> own_t;

  double val[num_val_];
#ifdef ENABLE_AD
  double tang[num_val_ * num_tang_];

  double &acc_tang(int val_idx, int tang_idx) {
    return tang[val_idx * num_tang_ + tang_idx];
  }
  const double &get_tang(int val_idx, int tang_idx) const {
    return tang[val_idx * num_tang_ + tang_idx];
  }
  void set_tang(int val_idx, int tang_idx, double tang_val) {
    get_tang(val_idx, tang_idx) = tang_val;
  }
#endif

  avec() {
    for (int i = 0; i < num_val_; i++)
      val[i] = 0.0;

#ifdef ENABLE_AD
    for (int i = 0; i < num_val_ * num_tang_; i++)
      tang[i] = 0.0;
#endif
  }

  avec(const double x, const double y) {
    val[0] = x;
    val[1] = y;

#ifdef ENABLE_AD
    for (int i = 0; i < num_val_ * num_tang_; i++)
      tang[i] = 0.0;
#endif
  }

  avec(const adouble &x, const adouble &y) {
    const adouble *xs[2] = {&x, &y};

    for (int v = 0; v < num_val_; v++) {
      val[v] = xs[v]->val;

#ifdef ENABLE_AD
      for (int t = 0; t < num_tang_; t++)
        acc_tang(v, t) = xs[v]->get_tang(t);
#endif
    }
  }

  avec(const double x, const double y, const double z) {
    val[0] = x;
    val[1] = y;
    val[2] = y;

#ifdef ENABLE_AD
    for (int i = 0; i < num_val_ * num_tang_; i++)
      tang[i] = 0.0;
#endif
  }

  avec(const adouble &x, const adouble &y, const adouble &z) {
    const adouble *xs[3] = {&x, &y, &z};

    for (int v = 0; v < num_val_; v++) {
      val[v] = xs[v]->val;

#ifdef ENABLE_AD
      for (int t = 0; t < num_tang_; t++)
        acc_tang(v, t) = xs[v]->get_tang(t);
#endif
    }
  }

  adouble squared_norm() const {
    double len = 0.0;
    for (int i = 0; i < num_val_; i++)
      len += val[i] * val[i];

    adouble r;
    r.val = len;

#ifdef ENABLE_AD
    r.init_full_tang(false);
    for (int t = 0; t < num_tang_; t++) {
      double deriv = 0;
      for (int v = 0; v < num_val_; v++)
        deriv += 2 * val[v] * get_tang(v, t);
      r.tang[t] = deriv;
    }
#endif
    return r;
  }

  adouble norm() const {
    double len = 0.0;
    for (int i = 0; i < num_val_; i++)
      len += val[i] * val[i];
    len = sqrt(len);

    adouble r;
    r.val = len;

#ifdef ENABLE_AD
    r.init_full_tang(false);
    for (int t = 0; t < num_tang_; t++) {
      double deriv = 0;
      for (int v = 0; v < num_val_; v++)
        deriv += val[v] * get_tang(v, t);
      deriv /= len;
      r.tang[t] = deriv;
    }
#endif
    return r;
  }

  adouble dot(const own_t &other) const {
    own_t prod = *this * other;

    adouble r = 0.0;
    for (int v = 0; v < num_val_; v++)
      r.val += prod.val[v];

#ifdef ENABLE_AD
    r.init_full_tang(false);
    for (int t = 0; t < num_tang_; t++) {
      double deriv = 0;
      for (int v = 0; v < num_val_; v++)
        deriv += prod.get_tang(v, t);
      r.tang[t] = deriv;
    }
#endif
    return r;

  };

  // expensive, not to be overused
  adouble operator[](size_t v) const {
    adouble r;
    r.val = val[v];
#ifdef ENABLE_AD
    r.init_full_tang(false);
    for (int t = 0; t < num_tang_; t++)
      r.tang[t] = get_tang(v, t);
#endif
    return r;
  }

#ifdef ENABLE_AD

#define BINARY_OP(OP, TANG_EXPR_V_V, TANG_EXPR_V_A, TANG_EXPR_V_D, SWAP_EXPR)  \
  own_t operator OP(const own_t &other) const {                                \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other.val[v];                                       \
                                                                               \
    for (int i = 0; i < num_val_ * num_tang_; i++) {                           \
      int v = i / num_tang_;                                                   \
      (void)v;                                                                 \
      r.tang[i] = TANG_EXPR_V_V;                                               \
    }                                                                          \
                                                                               \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  own_t operator OP(const adouble &other) const {                              \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other.val;                                          \
                                                                               \
    for (int i = 0; i < num_val_ * num_tang_; i++) {                           \
      int v = i / num_tang_;                                                   \
      (void)v;                                                                 \
      int t = i % num_tang_;                                                   \
      r.tang[i] = TANG_EXPR_V_A;                                               \
    }                                                                          \
                                                                               \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  friend own_t operator OP(const adouble &lhs, const own_t &rhs) {             \
    return SWAP_EXPR;                                                          \
  }                                                                            \
                                                                               \
  own_t operator OP(const double &other) const {                               \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other;                                              \
                                                                               \
    for (int i = 0; i < num_val_ * num_tang_; i++)                             \
      r.tang[i] = TANG_EXPR_V_D;                                               \
                                                                               \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  friend own_t operator OP(const double &lhs, const own_t &rhs) {              \
    return SWAP_EXPR;                                                          \
  }                                                                            \
                                                                               \
  void operator OP##=(const own_t &other) {                                    \
    for (int i = 0; i < num_val_ * num_tang_; i++) {                           \
      int v = i / num_tang_;                                                   \
      (void)v;                                                                 \
      tang[i] = TANG_EXPR_V_V;                                                 \
    }                                                                          \
                                                                               \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = other.val[v];                                              \
  }                                                                            \
                                                                               \
  void operator OP##=(const adouble &other) {                                  \
    for (int i = 0; i < num_val_ * num_tang_; i++) {                           \
      int v = i / num_tang_;                                                   \
      (void)v;                                                                 \
      int t = i % num_tang_;                                                   \
      tang[i] = TANG_EXPR_V_A;                                                 \
    }                                                                          \
                                                                               \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = other.val;                                                 \
  }                                                                            \
                                                                               \
  void operator OP##=(const double &other) {                                   \
    for (int i = 0; i < num_val_ * num_tang_; i++)                             \
      tang[i] = TANG_EXPR_V_D;                                                 \
                                                                               \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = val[v] OP other;                                           \
  }

#else

#define BINARY_OP(OP, TANG_EXPR_V_V, TANG_EXPR_V_A, TANG_EXPR_V_D, SWAP_EXPR)  \
  own_t operator OP(const own_t &other) const {                                \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other.val[v];                                       \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  own_t operator OP(const adouble &other) const {                              \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other.val;                                          \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  friend own_t operator OP(const adouble &lhs, const own_t &rhs) {             \
    return SWAP_EXPR;                                                          \
  }                                                                            \
                                                                               \
  own_t operator OP(const double &other) const {                               \
    own_t r;                                                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      r.val[v] = val[v] OP other;                                              \
    return r;                                                                  \
  }                                                                            \
                                                                               \
  friend own_t operator OP(const double &lhs, const own_t &rhs) {              \
    return SWAP_EXPR;                                                          \
  }                                                                            \
                                                                               \
  void operator OP##=(const own_t &other) {                                    \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = other.val[v];                                              \
  }                                                                            \
                                                                               \
  void operator OP##=(const adouble &other) {                                  \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = other.val;                                                 \
  }                                                                            \
                                                                               \
  void operator OP##=(const double &other) {                                   \
    for (int v = 0; v < num_val_; v++)                                         \
      val[v] OP## = val[v] OP other;                                           \
  }

#endif

#define OTHER_ADOUBLE_TANG other.get_tang(t)                                   

  BINARY_OP(+, tang[i] + other.tang[i], tang[i] + OTHER_ADOUBLE_TANG, tang[i],
            rhs + lhs);
  BINARY_OP(-, tang[i] - other.tang[i], tang[i] - OTHER_ADOUBLE_TANG, tang[i],
            -rhs + lhs);
  BINARY_OP(*, val[v] * other.tang[i] + tang[i] * other.val[v],
            val[v] * OTHER_ADOUBLE_TANG + tang[i] * other.val, tang[i] * other,
            rhs *lhs);
  BINARY_OP(/,
            ((tang[i] * other.val[v] - val[v] * other.tang[i]) /
             (other.val[v] * other.val[v])),
            ((tang[i] * other.val - val[v] * OTHER_ADOUBLE_TANG) /
             (other.val * other.val)),
            tang[i] / other, {NAN}); // TODO: divide scalar by vector

  own_t operator-() const {
    own_t r;
    for (int v = 0; v < num_val_; v++)
      r.val[v] = val[v];

#ifdef ENABLE_AD
    for (int i = 0; i < num_val_ * num_tang_; i++)
      r.tang[i] = tang[i];
#endif

    return r;
  }
};

typedef avec<2, num_inputs> adouble2;
typedef avec<3, num_inputs> adouble3;
