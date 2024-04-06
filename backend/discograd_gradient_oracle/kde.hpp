/*
 * MIT License
 * Copyright (c) 2018 duncanmcnae
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "../globals.hpp"

/// @file kde.h
/// @brief Kernel Density Estimation for C++
///
/// @author Duncan McNae

/// @brief kdepp is the namespace for the entire kdepp project.
namespace kdepp {

/// @brief Helper math functions used by kdepp
namespace kdemath {

/// @brief 2D Covariance matrix:
template <typename T>
std::array<typename T::value_type, 4> covariance2d(std::vector<T> const & data)
{
    using Data_type = typename T::value_type;
    std::array<Data_type, 2> mean = {0, 0};
    for (auto i : data) {
        mean[0] += i[0];
        mean[1] += i[1];
    }

    mean[0] = mean[0] / (1.0 * data.size());
    mean[1] = mean[1] / (1.0 * data.size());

    std::array<Data_type, 4> cov = {0, 0, 0, 0};
    for (auto x_j : data) {
        std::array<Data_type, 2> diff = { (x_j[0] - mean[0]), (x_j[1] - mean[1]) };
        cov[0] += diff[0] * diff[0];
        cov[1] += diff[0] * diff[1];
        cov[3] += diff[1] * diff[1];
    }

    cov[2] = cov[1];
    double n = static_cast<Data_type>(data.size());

    cov[0] = (1.0 / (n-1.0)) * cov[0];
    cov[1] = (1.0 / (n-1.0)) * cov[1];
    cov[2] = (1.0 / (n-1.0)) * cov[2];
    cov[3] = (1.0 / (n-1.0)) * cov[3];
    return cov;
}


/// @brief pi as constexpr
template <typename T>
constexpr T pi()
{
    return std::atan(1.0) * 4.0;
}

}  // namespace kdemath

/// @brief Kernel density estimation for one dimensional data.
template <typename T_IN, typename T>
class Kde1d {
 public:
    Kde1d(std::vector<T_IN> data,
          std::string bandwidth_method = "scott")
        : data_(data)
    {
        // Some checks on data, first num elements:
        if (data_.size() < 2) {
            throw std::invalid_argument("Only one data point");
        }

        // All the same is invalid:
        if (std::all_of(data.begin(), data.end(), [&data](T x){return x == data.front();})) {
            invalid_arg = true;
        }
        init_bandwidth(bandwidth_method);
        pre_calculate_terms();
    };

    /// @brief Evaluate the initialized kernel estimator at given point.
    T eval(T point) const
    {
        if (invalid_arg)
          return 0.0;

        T sum = 0;
        for (auto const & i : data_) {
            sum += kernel(point - (T)i);
        }
        T n = data_.size();
        return sum / n;
    };

    /// @brief Manually set bandwidth.
    void set_bandwidth(T h)
    {
        h_ = h;
        pre_calculate_terms();
    };

    T stddev() const { return root_h_; };

    T integrate(T low, T high) {
      T stddev = root_h_;

      T sum = 0;
      for (auto sample : data_)
        sum += norm_cdf((high - (T)sample) / stddev) - norm_cdf((low - (T)sample) / stddev);

      return sum / data_.size();
    }

 private:
    /// @brief Calculates variance of a std::vector<T> of data. 
    T variance(std::vector<T_IN> const & data)
    {
        const T init = 0.0;
        T sum = std::accumulate(data.begin(), data.end(), init);
        T mean = sum / data.size();
    
        T res = 0.0;
        std::for_each(data.begin(), data.end(), [&](T x) {
            res += std::pow(x - mean, 2.0);
        });
    
        double variance = res / (data.size() - 1); 
    
        if (variance < DGO_MIN_COND_VARIANCE)
          invalid_arg = true;
    
        return variance;
    }
    
    /// @brief Calculates standard deviation of a std::vector<T> of data. 
    T std_dev(std::vector<T_IN> const & data)
    {
        T var = variance(data);
        return std::sqrt(var);
    }

    T norm_cdf(T x) { return 0.5 * erfc(-x * M_SQRT1_2); }

    T kernel(T diff) const
    {
        return pow_pi_term_ * h_pow_term_ * std::exp((-1.0/2.0) * diff * h_pow_exp_term_ * diff);
    }

    void init_bandwidth(std::string bandwidth_method)
    {
        if (bandwidth_method == "silverman") {
            root_h_ = std::pow(4.0 / 3.0, 1.0/5.0) * std::pow(data_.size(), (-1.0 / 5.0)) * std_dev(data_);
            h_ = root_h_ * root_h_;
        } else {
            root_h_ = std::pow(data_.size(), (-1.0 / 5.0)) * std_dev(data_);
            h_ = root_h_ * root_h_;
        }
    }

    void pre_calculate_terms()
    {
        // for optimization do some pre calcs of power and exponential constants:
        pow_pi_term_ = std::pow(2 * kdemath::pi<T>(), -1.0/2.0);
        h_pow_term_ = std::pow(h_, -1.0/2.0);
        h_pow_exp_term_ = std::pow(h_, -1.0);
    }

    std::vector<T_IN> const data_;
    T h_;
    T root_h_;

    // Pre calculated terms for efficiency only:
    T pow_pi_term_;
    T h_pow_term_;
    T h_pow_exp_term_;

    bool invalid_arg = false;
};

}  // namespace kdepp
