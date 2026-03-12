#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>
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
) {

    //------------------------------------------------------------------
    // Basic matrix dimensions
    //------------------------------------------------------------------

    const size_t n_cols = static_cast<size_t>(std::abs(mat.ncol()));
    const size_t n_obs  = static_cast<size_t>(std::abs(mat.nrow()));

    //------------------------------------------------------------------
    // Convert R variable indices -> C++ (0-based)
    //------------------------------------------------------------------

    std::vector<size_t> v = Rcpp::as<std::vector<size_t>>(vars);

    for (auto& idx : v)
    {
        if (idx < 1 || idx > n_cols)
        {
            Rcpp::stop(
                "Column index %d out of bounds [1, %d]",
                static_cast<int>(idx),
                static_cast<int>(n_cols)
            );
        }

        idx -= 1;  // convert to 0-based indexing
    }

    const size_t n_vars = v.size();

    //------------------------------------------------------------------
    // Convert R neighbor structure -> std::vector
    //------------------------------------------------------------------

    std::vector<std::vector<size_t>> nb_std = nb2std(nb);

    //------------------------------------------------------------------
    // Pattern matrix
    //
    // Layout:
    //   pm[var][obs] -> Pattern (vector<uint8_t>)
    //------------------------------------------------------------------

    std::vector<std::vector<std::vector<uint8_t>>> pm;
    pm.resize(n_vars);

    for (size_t j = 0; j < n_vars; ++j)
    {
        // Reserve space for observations to avoid repeated reallocations
        pm[j].reserve(n_obs);
    }

    //------------------------------------------------------------------
    // Process each selected variable
    //------------------------------------------------------------------

    for (size_t j = 0; j < n_vars; ++j)
    {
        size_t col_id = v[j];

        //--------------------------------------------------------------
        // Extract column vector from R matrix
        //--------------------------------------------------------------

        std::vector<double> vec(n_obs);

        for (size_t r = 0; r < n_obs; ++r)
        {
            vec[r] = mat(r, col_id);
        }

        //--------------------------------------------------------------
        // Generate lattice embedding
        //
        // Output:
        //   [n_obs x embedding_dimension]
        //--------------------------------------------------------------

        std::vector<std::vector<double>> embeddings =
            Embed::GenLatticeEmbedding(
                vec,
                nb_std,
                static_cast<size_t>(std::abs(E)),
                static_cast<size_t>(std::abs(tau)),
                static_cast<size_t>(std::abs(style))
            );

        //--------------------------------------------------------------
        // Convert continuous embedding -> symbolic patterns
        //
        // Encoding:
        //   0 = NA
        //   1 = down
        //   2 = flat
        //   3 = up
        //--------------------------------------------------------------

        pm[j] = SymDync::GenSymbolicPattern(
            embeddings,
            relative,
            na_rm
        );
    }

    //------------------------------------------------------------------
    // Construct variable index vector for JE
    //
    // pm layout:
    //   pm[0], pm[1], ..., pm[n_vars-1]
    //
    // Therefore JE variables must be:
    //   {0,1,...,n_vars-1}
    //------------------------------------------------------------------

    std::vector<size_t> je_vars(n_vars);

    for (size_t i = 0; i < n_vars; ++i)
    {
        je_vars[i] = i;
    }

    //------------------------------------------------------------------
    // Compute Joint Entropy
    //------------------------------------------------------------------

    return InfoTheo::JE(pm, je_vars, base, na_rm);
}
