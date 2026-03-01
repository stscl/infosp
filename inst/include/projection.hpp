#ifndef PROJECTION_HPP
#define PROJECTION_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <utility>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include "numericutils.hpp"

namespace Projection
{
    inline std::vector<double> Simplex(
        const std::vector<std::vector<double>>& embedding,
        const std::vector<double>& target,
        const std::vector<size_t>& lib,
        const std::vector<size_t>& pred,
        size_t num_neighbors = 4,
        std::string method = "euclidean")
    {

    }

}// namespace Projection

#endif // PROJECTION_HPP
