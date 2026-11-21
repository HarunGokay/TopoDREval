#pragma once
#include <vector>
#include <Eigen/Dense>

struct PCAResult
{
    std::vector<std::vector<double>> reduced;
    std::vector<std::vector<double>> reconstructed;
    int components;
    double explainedVarianceRatio;
};

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data, double varianceThreshold);
std::vector<std::vector<double>> PCA_dim(const Eigen::MatrixXd &data, int targetDim);
PCAResult computePCA(const Eigen::MatrixXd &data, int targetDim);
