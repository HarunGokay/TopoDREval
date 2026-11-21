#include "kmeans.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <algorithm>

std::vector<double> computeCentroid(const std::vector<std::vector<double>>& points)
{
    if (points.empty()) return {};
    std::vector<double> centroid(points[0].size(), 0.0);
    for (const auto& p : points)
    {
        for (size_t i = 0; i < p.size(); i++)
        {
            centroid[i] += p[i];
        }
    }
    for (size_t i = 0; i < centroid.size(); i++)
    {
        centroid[i] /= points.size();
    }
    return centroid;
}

std::vector<std::vector<double>> initializeCentroidsKMeansPP(const std::vector<std::vector<double>>& data, int k)
{
    int n = data.size();
    std::vector<std::vector<double>> centroids;
    std::vector<double> distances(n, std::numeric_limits<double>::max());

    std::mt19937 gen(42);
    std::uniform_int_distribution<> distrib(0, n - 1);

    centroids.push_back(data[distrib(gen)]);

    for (int c = 1; c < k; ++c)
    {
        double totalDist = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double dist = euclideanDist(data[i], centroids.back());
            dist *= dist;
            if (dist < distances[i])
            {
                distances[i] = dist;
            }
            totalDist += distances[i];
        }

        std::uniform_real_distribution<> realDist(0.0, totalDist);
        double r = realDist(gen);
        double cumulative = 0.0;
        int selectedIndex = 0;

        for (int i = 0; i < n; ++i)
        {
            cumulative += distances[i];
            if (r <= cumulative)
            {
                selectedIndex = i;
                break;
            }
        }
        centroids.push_back(data[selectedIndex]);
    }

    return centroids;
}

std::tuple<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>, double> kmeans(const std::vector<std::vector<double>>& data, int k, int max_iters)
{
    int n = data.size();
    if (n == 0 || k <= 0)
    {
        return {{}, {}, 0.0};
    }

    std::vector<std::vector<std::vector<double>>> clusters(k);
    std::vector<std::vector<double>> centroids = initializeCentroidsKMeansPP(data, k);
    std::vector<int> labels(n, 0);

    for (int iter = 0; iter < max_iters; iter++)
    {
        bool changed = false;
        for (int c = 0; c < k; ++c)
        {
            clusters[c].clear();
        }

        for (int i = 0; i < n; i++)
        {
            double bestDist = std::numeric_limits<double>::max();
            int bestCluster = 0;

            for (int c = 0; c < k; c++)
            {
                double dist = euclideanDist(data[i], centroids[c]);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestCluster = c;
                }
            }
            if (labels[i] != bestCluster)
            {
                labels[i] = bestCluster;
                changed = true;
            }
        }

        for (int i = 0; i < n; i++)
        {
            clusters[labels[i]].push_back(data[i]);
        }

        for (int c = 0; c < k; c++)
        {
            if (!clusters[c].empty())
            {
                centroids[c] = computeCentroid(clusters[c]);
            }
        }

        if (!changed) break;
    }

    double score = silhouetteScore(data, labels);
    return {clusters, centroids, score};
}

double silhouetteScore(const std::vector<std::vector<double>>& X, const std::vector<int>& labels)
{
    int N = X.size();
    if (N == 0) return 0.0;

    std::unordered_map<int, std::vector<int>> clusters;
    for (int i = 0; i < N; i++)
    {
        clusters[labels[i]].push_back(i);
    }

    if (clusters.size() <= 1) return 0.0;

    double totalScore = 0.0;

    for (int i = 0; i < N; i++)
    {
        int cluster_i = labels[i];
        double a = 0.0;
        int clusterSize = clusters[cluster_i].size();

        if (clusterSize > 1)
        {
            for (int j : clusters[cluster_i])
            {
                if (j == i) continue;
                a += euclideanDist(X[i], X[j]);
            }
            a /= (clusterSize - 1);
        }

        double b = std::numeric_limits<double>::max();
        for (const auto& kv : clusters)
        {
            int cluster_j = kv.first;
            if (cluster_j == cluster_i || kv.second.empty()) continue;

            double distSum = 0.0;
            for (int j : kv.second)
            {
                distSum += euclideanDist(X[i], X[j]);
            }
            double avgDist = distSum / kv.second.size();
            if (avgDist < b) b = avgDist;
        }

        double s = 0.0;
        if (clusterSize > 1 && std::max(a, b) > 0.0)
        {
            s = (b - a) / std::max(a, b);
        }

        if (!std::isnan(s))
        {
            totalScore += s;
        }
    }

    return totalScore / N;
}
