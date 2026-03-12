#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "embed.hpp"
#include "symdync.hpp"
#include "infotheo.hpp"
#include "surd.hpp"

double RcppSPE4Lattice(
    const Rcpp::NumericMatrix& mat,
    const Rcpp::IntegerVector& vars,
    const Rcpp::List& nb,
    int E = 3,
    int tau = 1,
    int style = 1,
    bool relative = true,
    double base = 2.0,
    bool na_rm = true
)