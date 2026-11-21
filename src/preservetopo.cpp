#include "preservetopo.h"
#include "utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <stdexcept>

std::vector<double> convertToRank(const std::vector<double> &X, bool toggleTieHandling)
{
    int n = X.size();
    std::vector<double> ranks(n);

    std::vector<std::pair<double, int>> vp;
    vp.reserve(n);
    for (int i = 0; i < n; i++)
    {
        vp.emplace_back(X[i], i);
    }

    std::sort(vp.begin(), vp.end(),
              [](const auto &a, const auto &b)
              { return a.first < b.first; });

    if (toggleTieHandling)
    {
        int i = 0;
        while (i < n)
        {
            int j = i + 1;
            const double val = vp[i].first;

            while (j < n && vp[j].first == val)
                j++;

            const double avg = ((i + 1) + j) * 0.5;

            for (int k = i; k < j; k++)
                ranks[vp[k].second] = avg;

            i = j;
        }
    }
    else
    {
        for (int r = 0; r < n; r++)
            ranks[vp[r].second] = r + 1;
    }

    return ranks;
}

double spearmanCorrelation(const std::vector<double> &raw, const std::vector<double> &processed)
{
    std::vector<double> A = convertToRank(raw, true);
    std::vector<double> B = convertToRank(processed, true);

    if (A.size() != B.size())
    {
        throw std::invalid_argument("[Spearman] Vectors need to be the same size!");
    }

    double rs = 0.0;
    int n = A.size();
    if (n < 2) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        double di2 = (A[i] - B[i]) * (A[i] - B[i]);
        sum = sum + di2;
    }

    rs = 1.0 - ((6.0 * sum) / (double(n) * ((double(n) * double(n)) - 1.0)));
    return rs;
}

double pearsonCorrelation(const std::vector<double> &X, const std::vector<double> &Y)
{
    if (X.size() < 2 || Y.size() < 2)
    {
        throw std::invalid_argument("[Pearson] X and Y need to be at least size 2!");
    }
    if (X.size() != Y.size())
    {
        throw std::invalid_argument("[Pearson] Vectors need to be the same size!");
    }

    int n = Y.size();
    double meanX = 0.0;
    double meanY = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        meanX += X[i];
        meanY += Y[i];
    }
    meanX /= n;
    meanY /= n;

    double sumXY = 0.0;
    double sumXsqrd = 0.0;
    double sumYsqrd = 0.0;

    for (size_t i = 0; i < n; i++)
    {
        double xdiff = X[i] - meanX;
        double ydiff = Y[i] - meanY;
        sumXY += xdiff * ydiff;
        sumXsqrd += xdiff * xdiff;
        sumYsqrd += ydiff * ydiff;
    }

    double denom = std::sqrt(sumXsqrd * sumYsqrd);
    if (denom <= 1e-12)
    {
        return 0.0;
    }

    return sumXY / denom;
}

double huber(double e, double delta)
{
    if (std::abs(e) <= delta)
        return 0.5 * e * e;
    return delta * (std::abs(e) - 0.5 * delta);
}

double scoreFuncPure(
    double mean_r,
    double median_r,
    double mean_rs,
    double median_rs,
    double knnscore
)
{
    double pr = std::max(mean_r, 0.0);
    double sr = std::max(mean_rs, 0.0);

    double safe_median_r = (std::abs(median_r) < 1e-9) ? 1e-9 : median_r;
    double safe_median_rs = (std::abs(median_rs) < 1e-9) ? 1e-9 : median_rs;

    double p_ratio = mean_r / safe_median_r;
    double s_ratio = mean_rs / safe_median_rs;

    double p_pen = huber(p_ratio - 1.0, 0.2);
    double s_pen = huber(s_ratio - 1.0, 0.2);

    const double w_knn = 0.6;
    const double w_pr = 0.2;
    const double w_sr = 0.2;
    const double w_pen = 0.4;

    double preservation = w_knn * knnscore + w_pr * pr + w_sr * sr;
    double penalty = w_pen * (p_pen + s_pen);

    return preservation - penalty;
}

double scoreFunc(
    double mean_r,
    double median_r,
    double mean_rs,
    double median_rs,
    double knnscore
)
{
    double pr = std::max(mean_r, 0.0);
    double sr = std::max(mean_rs, 0.0);

    double safe_median_r = (std::abs(median_r) < 1e-9) ? 1e-9 : median_r;
    double safe_median_rs = (std::abs(median_rs) < 1e-9) ? 1e-9 : median_rs;

    double p_ratio = mean_r / safe_median_r;
    double s_ratio = mean_rs / safe_median_rs;

    double p_pen = (p_ratio - 1.0) * (p_ratio - 1.0);
    double s_pen = (s_ratio - 1.0) * (s_ratio - 1.0);

    return 2.0 * knnscore + 1.0 * pr + 1.0 * sr - 1.5 * p_pen - 1.5 * s_pen;
}

std::vector<std::vector<int>> knn(const std::vector<std::vector<double>> &data, int K)
{
    const size_t N = data.size();
    std::vector<std::vector<bool>> directed(N, std::vector<bool>(N, false));

    for (size_t i = 0; i < N; i++)
    {
        if (data[i].empty())
            throw std::invalid_argument("Empty vector in knn!");

        std::vector<std::pair<double, int>> buf;
        buf.reserve(N - 1);

        for (size_t j = 0; j < N; j++)
        {
            if (i == j)
                continue;
            buf.emplace_back(euclideanDist(data[i], data[j]), j);
        }

        size_t limit = std::min((size_t)K, buf.size());
        std::nth_element(buf.begin(), buf.begin() + limit, buf.end(),
                         [](const auto &a, const auto &b)
                         {
                             return a.first < b.first;
                         });

        for (size_t k = 0; k < limit; k++)
            directed[i][buf[k].second] = true;
    }

    std::vector<std::vector<int>> mutual(N);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            if (i != j && directed[i][j] && directed[j][i])
            {
                mutual[i].push_back(j);
            }
        }
    }

    return mutual;
}

double JaccardIndex(const std::vector<std::vector<int>>& A,
                    const std::vector<std::vector<int>>& B)
{
    if (A.size() != B.size())
        throw std::invalid_argument("Sizes dont match");

    double total = 0.0;
    size_t n = A.size();

    for (size_t i = 0; i < n; ++i)
    {
        std::unordered_set<int> setA(A[i].begin(), A[i].end());
        int inter = 0;

        for (int x : B[i])
            if (setA.count(x)) inter++;

        int uni = setA.size() + B[i].size() - inter;
        total += (uni == 0 ? 0.0 : double(inter) / double(uni));
    }

    return total / n;
}
