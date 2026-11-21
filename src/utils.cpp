#include "utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

double euclideanDist(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() != b.size())
    {
        std::cerr << "[euclideanDist] Dimension mismatch: a.size() = " << a.size()
                  << ", b.size() = " << b.size() << std::endl;
        exit(1);
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double mean(const std::vector<double> &data)
{
    if (data.empty())
        return 0.0;
    double sum = 0.0;
    for (double v : data)
        sum += v;
    return sum / data.size();
}

double median(const std::vector<double> &data)
{
    if (data.empty())
        return 0.0;
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0)
        return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
    else
        return sorted[n / 2];
}

std::vector<std::vector<double>> transposeData(const std::vector<std::vector<double>>& matrix)
{
    if (matrix.empty()) return {};
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            result[j][i] = matrix[i][j];
        }
    }
    return result;
}

Eigen::MatrixXd vectorToEigenMatrix(const std::vector<std::vector<double>>& data)
{
    if (data.empty()) return Eigen::MatrixXd(0, 0);
    size_t rows = data.size();
    size_t cols = data[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = data[i][j];
    return mat;
}

std::vector<std::vector<double>> matrixToVector2D(const Eigen::MatrixXd& mat)
{
    std::vector<std::vector<double>> result(mat.rows(), std::vector<double>(mat.cols()));
    for (int i = 0; i < mat.rows(); ++i)
        for (int j = 0; j < mat.cols(); ++j)
            result[i][j] = mat(i, j);
    return result;
}
