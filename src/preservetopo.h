#pragma once
#include <vector>

std::vector<double> convertToRank(const std::vector<double> &X, bool toggleTieHandling = true);
double spearmanCorrelation(const std::vector<double> &raw, const std::vector<double> &processed);
double pearsonCorrelation(const std::vector<double> &X, const std::vector<double> &Y);

double huber(double e, double delta);

double scoreFuncPure(
    double mean_r,
    double median_r,
    double mean_rs,
    double median_rs,
    double knnscore
);

double scoreFunc(
    double mean_r,
    double median_r,
    double mean_rs,
    double median_rs,
    double knnscore
);

std::vector<std::vector<int>> knn(const std::vector<std::vector<double>> &data, int K);
double JaccardIndex(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B);
