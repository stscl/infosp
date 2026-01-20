/*
 * infotheo.hpp
 *
 * Copyright (c) 2026-2030 Wenbo Lv
 *
 * Released under the GNU General Public License v3.0 (GPL-3)
 *
 * This program is free software: you can redistribute it and or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------
 * Description
 * --------------------------------------------------------------------
 * Count based information theory utilities.
 *
 * Data layout
 *   mat[variable][category]
 *
 * Each row represents one random variable.
 * Each column index represents the same discrete category across variables.
 *
 * Input model
 *   - Input values are non-negative sample counts (size_t).
 *   - Zero means this category does not occur.
 *   - Probabilities are internally normalized from counts.
 *
 * Joint model
 *   For multiple variables, joint weight on category k is:
 *     w_k = product of counts[var][k]
 *
 *   For numerical stability, computation is performed in log space
 *   using the log-sum-exp trick.
 *
 * Logarithm base
 *   - Default base = 2.0 (bits)
 *   - User can specify any positive base != 1.0
 */

#ifndef INFOTHEO_HPP
#define INFOTHEO_HPP

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstddef>
#include <algorithm>

namespace InfoTheo
{
// ================================================================
// Log utilities
// ================================================================

/*
 * Validate logarithm base.
 */
inline void validate_log_base(double base)
{
  if (!(base > 0.0) || std::abs(base - 1.0) < 1e-12)
  {
    throw std::invalid_argument("Log base must be positive and not equal to 1.");
  }
}

/*
 * Convert natural logarithm value to log with specified base.
 */
inline double ln_to_log_base(double ln_value, double base)
{
  return ln_value / std::log(base);
}

// ================================================================
// Single variable entropy from counts
// ================================================================

/*
 * Compute entropy from a count vector.
 *
 * counts[k] = number of samples in category k.
 *
 * H(X) = - sum_k p_k * log_base(p_k),
 * where p_k = counts[k] / sum_k counts[k].
 *
 * Numerical stability:
 *   Direct normalization is safe here because only one dimension
 *   is involved and no product accumulation exists.
 *
 * Default base = 2.0 (bits).
 */
inline double Entropy(
    const std::vector<size_t>& counts,
    double log_base = 2.0)
{
  validate_log_base(log_base);

  double total = 0.0;
  for (size_t c : counts) {
    total += static_cast<double>(c);
  }

  if (total <= 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  double h_nats = 0.0;
  for (size_t c : counts) {
    if (c == 0) continue;
    double p = static_cast<double>(c) / total;
    h_nats -= p * std::log(p);  // accumulate in nats
  }

  // Convert from nats to desired base
  return h_nats / std::log(log_base);
}

// ================================================================
// Multivariate joint entropy from counts (log-sum-exp stable)
// ================================================================

/*
 * Compute joint entropy for multiple variables using count product model.
 *
 * mat[var][category] = count
 * vars = indices of variables used for joint entropy.
 *
 * For each category k:
 *   ln_w_k = sum_v ln(count[v][k])
 *   Categories with any zero count are ignored.
 *
 * Normalization uses log-sum-exp to avoid overflow.
 */
inline double JE(
    const std::vector<std::vector<size_t>>& mat,
    const std::vector<size_t>& vars,
    double log_base = 2.0)
{
  validate_log_base(log_base);

  if (vars.empty()){
    throw std::invalid_argument("vars must not be empty.");
  }

  for (size_t v : vars){
    if (v >= mat.size()){
      throw std::out_of_range("Variable index out of range.");
    }
  }

  const size_t category_count = mat[vars[0]].size();

  // Store log-weights
  std::vector<double> ln_weights;
  ln_weights.reserve(category_count);

  double max_ln_w = -std::numeric_limits<double>::infinity();

  for (size_t k = 0; k < category_count; ++k){
    double ln_w = 0.0;
    bool valid = true;

    for (size_t v : vars){
      if (k >= mat[v].size()){
        throw std::out_of_range("Category index out of range.");
      }

      size_t c = mat[v][k];
      if (c == 0){
        valid = false;
        break;
      }

      ln_w += std::log(static_cast<double>(c));
    }

    if (!valid){
      continue;
    }

    ln_weights.push_back(ln_w);
    if (ln_w > max_ln_w){
      max_ln_w = ln_w;
    }
  }

  if (ln_weights.empty()){
    return std::numeric_limits<double>::quiet_NaN();
  }

  // Compute normalization denominator in log domain
  double sum_exp = 0.0;
  for (double ln_w : ln_weights){
    sum_exp += std::exp(ln_w - max_ln_w);
  }

  if (sum_exp <= 0.0){
    return std::numeric_limits<double>::quiet_NaN();
  }

  const double inv_log_base = 1.0 / std::log(log_base);

  // Entropy computation
  double h = 0.0;
  for (double ln_w : ln_weights)
  {
    double p = std::exp(ln_w - max_ln_w) / sum_exp;

    // log(p) in chosen base
    double log_p = ( (ln_w - max_ln_w) - std::log(sum_exp) ) * inv_log_base;

    h -= p * log_p;
  }

  return h;
}

// ================================================================
// Conditional entropy
// ================================================================

/*
 * H(X | Y) = H(X, Y) - H(Y)
 */
inline double CE(
    const std::vector<std::vector<size_t>>& mat,
    const std::vector<size_t>& X,
    const std::vector<size_t>& Y,
    double log_base = 2.0)
{
  if (X.empty() || Y.empty()){
    throw std::invalid_argument("X and Y must not be empty.");
  }

  std::vector<size_t> XY = X;
  XY.insert(XY.end(), Y.begin(), Y.end());

  return JE(mat, XY, log_base) - JE(mat, Y, log_base);
}

// ================================================================
// Multivariate mutual information
// ================================================================

/*
 * I(X1; ...; Xn) = sum_i H(Xi) - H(X1, ..., Xn)
 */
inline double MI(
    const std::vector<std::vector<size_t>>& mat,
    const std::vector<size_t>& vars,
    double log_base = 2.0)
{
  if (vars.size() < 2){
    throw std::invalid_argument("At least two variables are required.");
  }

  double sum_h = 0.0;
  for (std::size_t v : vars){
    if (v >= mat.size()){
      throw std::out_of_range("Variable index out of range.");
    }
    sum_h += Entropy(mat[v], log_base);
  }

  double h_joint = JE(mat, vars, log_base);
  return sum_h - h_joint;
}

// ================================================================
// Conditional mutual information
// ================================================================

/*
 * I(X ; Y | Z) =
 *   H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)
 */
inline double CMI(
    const std::vector<std::vector<size_t>>& mat,
    const std::vector<size_t>& X,
    const std::vector<size_t>& Y,
    const std::vector<size_t>& Z,
    double log_base = 2.0)
{
  if (X.empty() || Y.empty() || Z.empty()){
    throw std::invalid_argument("X, Y and Z must not be empty.");
  }

  std::vector<size_t> XZ = X;
  XZ.insert(XZ.end(), Z.begin(), Z.end());

  std::vector<size_t> YZ = Y;
  YZ.insert(YZ.end(), Z.begin(), Z.end());

  std::vector<size_t> XYZ = X;
  XYZ.insert(XYZ.end(), Y.begin(), Y.end());
  XYZ.insert(XYZ.end(), Z.begin(), Z.end());

  return JE(mat, XZ,  log_base)
    + JE(mat, YZ,  log_base)
    - JE(mat, Z,   log_base)
    - JE(mat, XYZ, log_base);
}

} // namespace InfoTheo

#endif // INFOTHEO_HPP
