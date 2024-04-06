#pragma once

#include <cstdint>

using namespace std;

template<typename T>
struct vec2 {
  T v[2];
  T &operator[](const size_t idx) { return v[idx]; }
  const T &operator[](const size_t idx) const { return v[idx]; }
};

template<typename T> vec2<T> operator+(const vec2<T> &a, const vec2<T> &b) { return { a[0] + b[0], a[1] + b[1] }; }
template<typename T> vec2<T> operator-(const vec2<T> &a, const vec2<T> &b) { return { a[0] - b[0], a[1] - b[1] }; }
template<typename T> vec2<T> operator*(const vec2<T> &a, const vec2<T> &b) { return { a[0] * b[0], a[1] * b[1] }; }
template<typename T> vec2<T> operator/(const vec2<T> &a, const vec2<T> &b) { return { a[0] / b[0], a[1] / b[1] }; }
template<typename T> bool operator==(const vec2<T> &a, const vec2<T> &b) { return a[0] == b[0] && a[1] == b[1]; }

template<typename T, typename O> vec2<T> operator+(const vec2<T> &a, const O &b) { return { a[0] + b, a[1] + b }; }
template<typename T, typename O> vec2<T> operator-(const vec2<T> &a, const O &b) { return { a[0] - b, a[1] - b }; }
template<typename T, typename O> vec2<T> operator*(const vec2<T> &a, const O &b) { return { a[0] * b, a[1] * b }; }
template<typename T, typename O> vec2<T> operator/(const vec2<T> &a, const O &b) { return { a[0] / b, a[1] / b }; }

template<typename T, typename O> vec2<T> operator+(const O &a, const vec2<T> &b) { return { a + b[0], a + b[1] }; }
template<typename T, typename O> vec2<T> operator-(const O &a, const vec2<T> &b) { return { a - b[0], a - b[1] }; }
template<typename T, typename O> vec2<T> operator*(const O &a, const vec2<T> &b) { return { a * b[0], a * b[1] }; }
template<typename T, typename O> vec2<T> operator/(const O &a, const vec2<T> &b) { return { a / b[0], a / b[1] }; }

template<typename T> void operator+=(vec2<T> &a, const vec2<T> &b) { a[0] += b[0]; a[1] += b[1]; }
template<typename T> void operator-=(vec2<T> &a, const vec2<T> &b) { a[0] -= b[0]; a[1] -= b[1]; }
template<typename T> void operator*=(vec2<T> &a, const vec2<T> &b) { a[0] *= b[0]; a[1] *= b[1]; }
template<typename T> void operator/=(vec2<T> &a, const vec2<T> &b) { a[0] /= b[0]; a[1] /= b[1]; }

