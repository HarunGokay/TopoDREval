#pragma once
#include <vector>

std::vector<std::vector<double>> generateHighDimClusters(
    int num_points = 500, int num_clusters = 5, int num_dims = 10,
    double cluster_spacing = 6.0, double noise_std = 0.5);

std::vector<std::vector<double>> generateHighDimTorus(
    int num_points = 500, int num_dims = 10,
    double R = 4.0, double r = 1.0, double noise_std = 0.2);

std::vector<std::vector<double>> generateHighDimSwissRoll(
    int num_points = 500, int num_dims = 10, double noise_std = 0.2);

std::vector<std::vector<double>> generateHighDimMobius(
    int num_points = 500, int num_dims = 10, double noise_std = 0.2);
