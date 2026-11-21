#pragma once
#include <vector>
#include <tuple>

std::vector<double> computeCentroid(const std::vector<std::vector<double>>& points);
std::vector<std::vector<double>> initializeCentroidsKMeansPP(const std::vector<std::vector<double>>& data, int k);
std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeans(const std::vector<std::vector<double>>& data, int k, int max_iters = 100);
double silhouetteScore(const std::vector<std::vector<double>>& X, const std::vector<int>& labels);
