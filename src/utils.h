#pragma once
#include <vector>
#include <Eigen/Dense>

double euclideanDist(const std::vector<double>& a, const std::vector<double>& b);
double mean(const std::vector<double> &data);
double median(const std::vector<double> &data);
std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>>& matrix);
Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>>& data);
std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat);
