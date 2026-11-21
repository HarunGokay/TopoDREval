#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <set>
#include <algorithm>
#include "utils.h"
#include "preservetopo.h"
#include "pca.h"
#include "kmeans.h"
#include "../data/datagen.h"

void evaluateDimensionalityReduction(const std::string& name, const std::vector<std::vector<double>>& raw_data, int kmeans_k)
{
    std::cout << "\n=========================================================================\n";
    std::cout << " Dataset: " << name << " | Points: " << raw_data.size()
              << " | Ambient Dims: " << raw_data[0].size() << " | k-means k: " << kmeans_k << "\n";
    std::cout << "=========================================================================\n";

    // 1. Raw baseline
    auto [raw_clusters, raw_centroids, raw_sil] = kmeans(raw_data, kmeans_k, 100);
    std::cout << "Baseline Raw (" << raw_data[0].size() << "D) Silhouette Score: "
              << std::fixed << std::setprecision(4) << raw_sil << "\n\n";

    const int kNN_k = 15;
    auto knn_raw = knn(raw_data, kNN_k);
    auto T_raw = transposeData(raw_data);

    int ambient = raw_data[0].size();
    std::vector<int> dims = {ambient, 8, 6, 4, 3, 2};
    std::vector<int> valid_dims;
    for (int d : dims)
    {
        if (d <= ambient && (valid_dims.empty() || valid_dims.back() != d))
            valid_dims.push_back(d);
    }

    std::cout << std::left
              << std::setw(6)  << "Dim"
              << std::setw(12) << "Var Expl."
              << std::setw(10) << "Mean r"
              << std::setw(10) << "Mean rs"
              << std::setw(12) << "kNN Jacc"
              << std::setw(14) << "TrustScore"
              << std::setw(12) << "Silhouette"
              << std::setw(10) << "Sil Delta"
              << "\n";
    std::cout << "-------------------------------------------------------------------------\n";

    int best_reduced_dim = 2;
    double best_reduced_trust = -1e9;
    double best_reduced_sil = -1e9;

    for (int d : valid_dims)
    {
        PCAResult pca_res = computePCA(vectorToEigenMatrix(raw_data), d);

        auto T_recon = transposeData(pca_res.reconstructed);

        std::vector<double> r_list;
        std::vector<double> rs_list;
        r_list.reserve(ambient);
        rs_list.reserve(ambient);

        for (size_t j = 0; j < T_raw.size(); ++j)
        {
            double r = pearsonCorrelation(T_raw[j], T_recon[j]);
            double rs = spearmanCorrelation(T_raw[j], T_recon[j]);
            r_list.push_back(r);
            rs_list.push_back(rs);
        }

        double mean_r = mean(r_list);
        double median_r = median(r_list);
        double mean_rs = mean(rs_list);
        double median_rs = median(rs_list);

        auto knn_reduced = knn(pca_res.reduced, kNN_k);
        double knnscore = JaccardIndex(knn_reduced, knn_raw);

        double trust_score = scoreFuncPure(mean_r, median_r, mean_rs, median_rs, knnscore);

        auto [red_clusters, red_centroids, red_sil] = kmeans(pca_res.reduced, kmeans_k, 100);
        double sil_delta = red_sil - raw_sil;

        if (d < ambient && (trust_score > best_reduced_trust || (trust_score >= best_reduced_trust - 0.1 && red_sil > best_reduced_sil)))
        {
            best_reduced_trust = trust_score;
            best_reduced_dim = d;
            best_reduced_sil = red_sil;
        }

        std::cout << std::left
                  << std::setw(6)  << d
                  << std::setw(12) << (std::to_string(int(pca_res.explainedVarianceRatio * 100.0 + 0.5)) + "%")
                  << std::setw(10) << std::fixed << std::setprecision(4) << mean_r
                  << std::setw(10) << std::fixed << std::setprecision(4) << mean_rs
                  << std::setw(12) << std::fixed << std::setprecision(4) << knnscore
                  << std::setw(14) << std::fixed << std::setprecision(4) << trust_score
                  << std::setw(12) << std::fixed << std::setprecision(4) << red_sil
                  << std::setw(10) << (sil_delta >= 0 ? "+" : "") + std::to_string(sil_delta).substr(0, 6)
                  << "\n";
    }

    std::cout << "-------------------------------------------------------------------------\n";
    std::cout << " Recommended Reduced Dimension: " << best_reduced_dim
              << "D | TrustScore: " << std::fixed << std::setprecision(4) << best_reduced_trust
              << " | Silhouette: " << best_reduced_sil
              << " (Gain: " << (best_reduced_sil - raw_sil >= 0 ? "+" : "") << (best_reduced_sil - raw_sil) << ")\n";
}

int main(int argc, char* argv[])
{
    std::string dataset = "all";
    if (argc > 1)
    {
        dataset = argv[1];
    }

    std::cout << "=========================================================================\n";
    std::cout << " Trustworthiness-Style Scoring for Dimensionality Reduction\n";
    std::cout << " Preserving Pearson, Spearman & kNN Topologies\n";
    std::cout << "=========================================================================\n";

    if (dataset == "clusters" || dataset == "all")
    {
        auto data = generateHighDimClusters(500, 5, 10, 6.0, 0.4);
        evaluateDimensionalityReduction("Gaussian Clusters (10D -> 2D Intrinsic)", data, 5);
    }

    if (dataset == "torus" || dataset == "all")
    {
        auto data = generateHighDimTorus(500, 10, 4.0, 1.0, 0.15);
        evaluateDimensionalityReduction("3D Torus in 10D Ambient Space", data, 4);
    }

    if (dataset == "swissroll" || dataset == "all")
    {
        auto data = generateHighDimSwissRoll(500, 10, 0.15);
        evaluateDimensionalityReduction("3D Swiss Roll in 10D Ambient Space", data, 4);
    }

    if (dataset == "mobius" || dataset == "all")
    {
        auto data = generateHighDimMobius(500, 10, 0.15);
        evaluateDimensionalityReduction("3D Inverted Mobius Strip in 10D Ambient Space", data, 4);
    }

    return 0;
}
