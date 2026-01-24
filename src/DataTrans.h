#ifndef DataTrans_H
#define DataTrans_H

#include <vector>
#include <Rcpp.h>

// Function to convert Rcpp::List to std::vector<std::vector<size_t>> (the `nb` object)
std::vector<std::vector<size_t>> nb2std(const Rcpp::List& nb);

#endif // DataTrans_H
