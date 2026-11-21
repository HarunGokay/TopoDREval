#include "pca.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

PCAResult computePCA(const Eigen::MatrixXd &data, int targetDim)
{
    if (data.rows() == 0 || data.cols() == 0)
    {
        throw std::runtime_error("Input data matrix is empty!");
    }
    if (data.rows() == 1)
    {
        throw std::runtime_error("PCA requires at least 2 rows of data.");
    }

    int k = std::clamp(targetDim, 1, static_cast<int>(data.cols()));

    const Eigen::RowVectorXd meanVec = data.colwise().mean();
    const Eigen::MatrixXd centered = data.rowwise() - meanVec;
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    if (eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    Eigen::VectorXd eigenValues = eig.eigenvalues();
    Eigen::MatrixXd eigenVectors = eig.eigenvectors();

    Eigen::VectorXd eigenValuesDesc = eigenValues.reverse();
    Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols());
    for (int i = 0; i < eigenVectors.cols(); ++i)
    {
        eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);
    }

    double totalVariance = eigenValuesDesc.sum();
    double keptVariance = eigenValuesDesc.head(k).sum();
    double ratio = (totalVariance > 0.0) ? (keptVariance / totalVariance) : 1.0;

    Eigen::MatrixXd Vk = eigenVectorsDesc.leftCols(k);
    Eigen::MatrixXd reduced = centered * Vk;
    Eigen::MatrixXd reconstructed = (reduced * Vk.transpose()).rowwise() + meanVec;

    PCAResult res;
    res.components = k;
    res.explainedVarianceRatio = ratio;

    res.reduced.assign(reduced.rows(), std::vector<double>(reduced.cols()));
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            res.reduced[i][j] = reduced(i, j);

    res.reconstructed.assign(reconstructed.rows(), std::vector<double>(reconstructed.cols()));
    for (int i = 0; i < reconstructed.rows(); ++i)
        for (int j = 0; j < reconstructed.cols(); ++j)
            res.reconstructed[i][j] = reconstructed(i, j);

    return res;
}

std::vector<std::vector<double>> PCA(const Eigen::MatrixXd &data, double varianceThreshold)
{
    if (data.rows() == 0 || data.cols() == 0)
    {
        throw std::runtime_error("Input data matrix is empty!");
    }

    if (data.rows() == 1)
    {
        throw std::runtime_error("PCA requires at least 2 rows of data.");
    }

    varianceThreshold = std::clamp(varianceThreshold, 0.0, 1.0);

    const Eigen::MatrixXd centered = data.rowwise() - data.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);
    if (eig.info() != Eigen::Success)
    {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    Eigen::VectorXd eigenValues = eig.eigenvalues();
    Eigen::MatrixXd eigenVectors = eig.eigenvectors();

    Eigen::VectorXd eigenValuesDesc = eigenValues.reverse();
    Eigen::MatrixXd eigenVectorsDesc(eigenVectors.rows(), eigenVectors.cols());
    for (int i = 0; i < eigenVectors.cols(); ++i)
    {
        eigenVectorsDesc.col(i) = eigenVectors.col(eigenVectors.cols() - 1 - i);
    }

    double totalVariance = eigenValuesDesc.sum();
    double runningSum = 0.0;
    int k = eigenValuesDesc.size();
    for (int i = 0; i < eigenValuesDesc.size(); ++i)
    {
        runningSum += eigenValuesDesc(i);
        if (totalVariance > 0.0 && runningSum / totalVariance >= varianceThreshold)
        {
            k = i + 1;
            break;
        }
    }

    Eigen::MatrixXd reduced = centered * eigenVectorsDesc.leftCols(k);

    std::vector<std::vector<double>> result(reduced.rows(), std::vector<double>(reduced.cols()));
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            result[i][j] = reduced(i, j);

    return result;
}

std::vector<std::vector<double>> PCA_dim(const Eigen::MatrixXd &data, int targetDim)
{
    PCAResult res = computePCA(data, targetDim);
    return res.reduced;
}
