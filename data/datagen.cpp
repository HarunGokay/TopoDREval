#include "datagen.h"
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

std::vector<std::vector<double>> generateHighDimClusters(
    int num_points, int num_clusters, int num_dims,
    double cluster_spacing, double noise_std)
{
    if (num_points <= 0 || num_clusters <= 0 || num_dims < 2)
        throw std::invalid_argument("Invalid cluster parameters");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::mt19937 gen(42);
    std::normal_distribution<> noise(0.0, noise_std);

    std::vector<std::pair<double, double>> centers2D;
    centers2D.reserve(num_clusters);

    const double radius = cluster_spacing * std::max(1.0, std::sqrt(num_clusters) / 2.0);
    const double angle_step = 2.0 * M_PI / num_clusters;

    for (int c = 0; c < num_clusters; ++c)
    {
        double angle = c * angle_step;
        centers2D.emplace_back(radius * std::cos(angle), radius * std::sin(angle));
    }

    std::uniform_int_distribution<> cluster_dist(0, num_clusters - 1);

    for (int i = 0; i < num_points; ++i)
    {
        int cluster_idx = cluster_dist(gen);
        const auto& center = centers2D[cluster_idx];

        std::vector<double> point(num_dims, 0.0);
        point[0] = center.first + noise(gen);
        point[1] = center.second + noise(gen);

        for (int d = 2; d < num_dims; ++d)
            point[d] = noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

std::vector<std::vector<double>> generateHighDimTorus(
    int num_points, int num_dims,
    double R, double r, double noise_std)
{
    if (num_points <= 0 || num_dims < 3)
        throw std::invalid_argument("Invalid torus parameters");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> angle(0.0, 2.0 * M_PI);
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < num_points; ++i)
    {
        double u = angle(gen);
        double v = angle(gen);

        std::vector<double> point(num_dims, 0.0);
        point[0] = (R + r * std::cos(v)) * std::cos(u) + noise(gen);
        point[1] = (R + r * std::cos(v)) * std::sin(u) + noise(gen);
        point[2] = r * std::sin(v) + noise(gen);

        for (int d = 3; d < num_dims; ++d)
            point[d] = noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

std::vector<std::vector<double>> generateHighDimSwissRoll(
    int num_points, int num_dims, double noise_std)
{
    if (num_points <= 0 || num_dims < 3)
        throw std::invalid_argument("Invalid swiss roll parameters");

    std::vector<std::vector<double>> data;
    data.reserve(num_points);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> unit(0.0, 1.0);
    std::uniform_real_distribution<> height(0.0, 20.0);
    std::normal_distribution<> noise(0.0, noise_std);

    for (int i = 0; i < num_points; ++i)
    {
        double t = 1.5 * M_PI * (1.0 + 2.0 * unit(gen));

        std::vector<double> point(num_dims, 0.0);
        point[0] = t * std::cos(t) + noise(gen);
        point[1] = height(gen) + noise(gen);
        point[2] = t * std::sin(t) + noise(gen);

        for (int d = 3; d < num_dims; ++d)
            point[d] = noise(gen);

        data.push_back(std::move(point));
    }

    return data;
}

std::vector<std::vector<double>> generateHighDimMobius(
    int num_points, int num_dims, double noise_std)
{
    if (num_points <= 0 || num_dims < 3)
        throw std::invalid_argument("Invalid mobius parameters");

    std::vector<std::vector<double>> data(num_points, std::vector<double>(num_dims, 0.0));

    std::mt19937 gen(42);
    std::normal_distribution<> noise(0.0, noise_std);
    std::uniform_real_distribution<> unit(-0.5, 0.5);

    double width = 0.4;
    double R = 1.0;
    double tiltAmplitude = 0.35;
    double zWaveAmp = 0.15;
    const double twistSign = -1.0;

    for (int i = 0; i < num_points; ++i)
    {
        double u = 2.0 * M_PI * (double)i / (double)num_points;
        double v = width * unit(gen);

        double rx = std::cos(u), ry = std::sin(u);
        double tx = -std::sin(u), ty = std::cos(u);

        double halfTheta = twistSign * 0.5 * u;
        double c = std::cos(halfTheta);
        double s = std::sin(halfTheta);

        double wx = c * rx + s * tx;
        double wy = c * ry + s * ty;
        double wz = tiltAmplitude * std::sin(3.0 * u);

        double mag = std::sqrt(wx * wx + wy * wy + wz * wz);
        wx /= mag;
        wy /= mag;
        wz /= mag;

        double cx = R * rx;
        double cy = R * ry;
        double cz = zWaveAmp * std::sin(2.0 * u);

        data[i][0] = cx + v * wx + noise(gen);
        data[i][1] = cy + v * wy + noise(gen);
        data[i][2] = cz + v * wz + noise(gen);

        for (int d = 3; d < num_dims; ++d)
            data[i][d] = noise(gen);
    }

    return data;
}
