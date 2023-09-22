/** Backend library for the SI-enabled version of DiscoGrad.
 *  Contains classes and functions for enabling a smooth interpretation of C++ programs.
 *
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

using namespace std;

/** The type to use for storing the variance of the gaussian distributions involved in the smooth state.
 *  When using adouble, the calculation of the variance influences the gradient (default). To disable
 *  the differentiation of variance-calculations, change to double.
 */
typedef adouble variance_t;

/** Selection of algorithms and heuristics that can be used instead of the original
 *  Restrict algorithm described in Chaudhuri et al.
 */
enum restrict_mode_t { si_merge_chaudhuri,                /**< Only retain the highest-weighted paths. */
                       si_merge_by_weights_only,          /**< Like Restrict, but only uses weights as cost. */
                       si_merge_chaudhuri_ignore_weights, /**< Like Restrict, but disregards weights in the cost. */
                       si_discard,                        /**< Only retain the highest-weighted paths. */
                     };

/** The maximum number of control flow paths to track for the smoothing.
 *  Determines the fidelity of the smoothed program output and gradient wrt. the 
 *  convolution. A higher number of tracked paths results in a more accurate result,
 *  but increases the run time.
 */
static size_t si_max_path_states = 128; 
/** The number of initial modes to create for sint parameters. */
static int max_sint_paths = 1;
/** {@link SiPathState}s with a weight below this threshold are discarded upon {@link SiPathState::absorb() absorption}. */
static double si_min_weight = 1e-20;
/** For truncate restrict heuristic: {@link SiPathState}s with a weight below this threshold are discarded upon 
 * {@link SiPathState::generate_cand_path_states() creation}. 
 */
static double si_min_branch_prob = 0;
/** For two {@link SiPathStates} that have a cost of merging higher than this threshold, the absorbed path is completely discared. 
 *  See also {@link SiPathState::absorb(SiPathState*,double)}.
 */
static double si_max_merge_cost = DBL_MAX;

/** For using differential error analysis for a more accurate variance propagation. 
 *  Note: "v" of an {@link SiGaussian} is still the heuristic SI variance. 
 */
static double si_dea_input_variance = 0.0;

/** Factor to limit the increase of variance introduced by the restriction heuristic.
 *  When the heuristic merges two paths (see {@link absorb()}, if the new variance would
 *  be si_max_variance_factor_by_merge times higher than the previous variances, the new
 *  variance is calculated by the weighted sum of the old variances (leading to a much
 *  smaller increase).
 */
static double si_max_variance_factor_by_merge = 2;

static double si_max_variance = 10;

/** Determines the algorithm to use to restrict the number of tracked control 
 *  flow paths to the limit si_max_path_states. 
 */
static restrict_mode_t si_restrict_mode = si_discard;

static bool print_adjoints = true;

/** internal function to cast a double or adouble to the specified {@link variance_t}. */
template <typename T_IN, typename T_OUT> T_OUT cast_to_variance_t(T_IN x);
template <> adouble cast_to_variance_t(double x) { return x; };             /**< @copydoc cast_to_variance_t */
template <> adouble cast_to_variance_t(adouble x) { return x; };            /**< @copydoc cast_to_variance_t */
template <> double cast_to_variance_t(double x) { return x; };              /**< @copydoc cast_to_variance_t */
template <> double cast_to_variance_t(adouble x) { return x.val; };         /**< @copydoc cast_to_variance_t */

struct hash_pair {
  template <class T1, class T2> size_t operator()(const pair<T1, T2> &p) const {
    auto hash1 = hash<T1>{}(p.first);
    auto hash2 = hash<T2>{}(p.second);

    if (hash1 != hash2) {
      return hash1 ^ hash2;
    }

    return hash1;
  }
};
