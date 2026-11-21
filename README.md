# Trustworthiness Scoring for Dimensionality Reduction

This is a project I built to solve an annoying problem with standard dimensionality reduction: blindly keeping dimensions based on explained variance often keeps high-dimensional noise that ruins clustering.

Instead of just looking at eigenvalue sums, this codebase evaluates dimensionality reduction using a custom **trustworthiness-style topological preservation score** combining:
1. **Column-wise Pearson correlation ($r$)**: Linear relationship preservation between original and reconstructed dimensions.
2. **Column-wise Spearman rank correlation ($r_s$)**: Monotonic rank topology preservation across features.
3. **Mutual $k$-Nearest Neighbor preservation (Jaccard Index)**: Verifying that local neighborhood graphs are retained without introducing false neighborhood bridges.
4. **Huber-loss ratio distortion penalties**: Penalizing skew between mean and median correlations to prevent dimension collapse.

We then benchmark this against **k-means Silhouette scores** across complex manifold shapes (3D Swiss Roll, 3D Torus, 3D Inverted Möbius Strip, and Gaussian Clusters, each embedded in 10-dimensional noisy ambient space) to prove that optimal dimension reduction improves cluster separation.

---

## The Math

### 1. Feature Correlations (Pearson & Spearman)
Given original data $X \in \mathbb{R}^{N \times D}$ and reconstructed data $\hat{X} \in \mathbb{R}^{N \times D}$ from $k$ principal components ($k < D$):

For each dimension $j \in \{1, \dots, D\}$:

- **Pearson correlation $r_j$:**

  $$
  r_j =
  \frac{
  \sum_{i=1}^{N} (X_{ij} - \bar{X}_j)(\hat{X}_{ij} - \bar{\hat{X}}_j)
  }{
  \sqrt{
  \sum_{i=1}^{N} (X_{ij} - \bar{X}_j)^2
  \sum_{i=1}^{N} (\hat{X}_{ij} - \bar{\hat{X}}_j)^2
  }
  }
  $$

- **Spearman rank correlation $r_{s, j}$**:
$$r_{s, j} = 1 - \frac{6 \sum_{i=1}^N d_i^2}{N(N^2 - 1)}$$
where $d_i = \mathrm{rank}(X_{ij}) - \mathrm{rank}(\hat{X}_{ij})$, with average ranks assigned to ties.

### 2. Mutual $k\text{NN}$ Topology Preservation (Jaccard Index)
We construct symmetric mutual $k$-nearest neighbor graphs for both high-dimensional space $G_{\text{raw}}$ and reduced space $G_{\text{reduced}}$:

$$e_{ij} \in E \iff j \in \mathrm{kNN}(i) \land i \in \mathrm{kNN}(j)$$

The neighborhood retention score is computed using the average Jaccard similarity across all $N$ vertices:

$$J(G_{\text{raw}}, G_{\text{reduced}}) = \frac{1}{N} \sum_{i=1}^N \frac{|\mathcal{N}_{\text{raw}}(i) \cap \mathcal{N}_{\text{reduced}}(i)|}{|\mathcal{N}_{\text{raw}}(i) \cup \mathcal{N}_{\text{reduced}}(i)|}$$

### 3. Huber Distortion Penalty & Final Score
To penalize outlier distortion where a few dimensions collapse while others dominate, we evaluate the ratio of mean to median correlation via a Huber loss penalty:

$$\rho_P = \frac{\bar{r}}{\mathrm{median}(r)}, \quad \rho_S = \frac{\bar{r}_s}{\mathrm{median}(r_s)}$$

$$\mathcal{H}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } |e| \le \delta \\ \delta (|e| - \frac{1}{2}\delta) & \text{otherwise} \end{cases}$$

The composite trustworthiness score balances neighborhood preservation with correlation fidelity while subtracting ratio penalties:

$$\text{Score} = w_{\text{knn}} \cdot J + w_P \cdot \bar{r} + w_S \cdot \bar{r}_s - w_{\text{pen}} \left( \mathcal{H}_\delta(\rho_P - 1) + \mathcal{H}_\delta(\rho_S - 1) \right)$$

---

## Datasets

All datasets are generated in 10-dimensional ambient space ($D = 10$) with Gaussian noise on ambient dimensions:
- **`clusters`**: 5 distinct Gaussian clusters arranged on a circle.
- **`torus`**: 3D Torus ($R = 4.0, r = 1.0$) embedded in 10D with noise.
- **`swissroll`**: 3D Swiss Roll spiral embedded in 10D with noise.
- **`mobius`**: 3D Inverted Möbius strip embedded in 10D with noise.

---

## Build

Requires a C++17 compiler and CMake $\ge 3.14$. Eigen3 is detected automatically (or fetched via CMake).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## Usage

```bash
# Run benchmark on all datasets
./trustworthiness_dim_reduction all

# Run benchmark on a specific shape
./trustworthiness_dim_reduction clusters
./trustworthiness_dim_reduction torus
./trustworthiness_dim_reduction swissroll
./trustworthiness_dim_reduction mobius
```

---

## File Structure

| File | Description |
|---|---|
| `src/preservetopo.cpp / .h` | Spearman, Pearson, Huber loss, mutual kNN Jaccard, and composite score functions |
| `src/pca.cpp / .h` | PCA projection, reconstruction, and explained variance computation |
| `src/kmeans.cpp / .h` | k-means with k-means++ seeding and Silhouette score calculation |
| `src/utils.cpp / .h` | Euclidean distance, vector transposition, mean, median, and Eigen conversions |
| `data/datagen.cpp / .h` | High-dimensional synthetic generators (Clusters, Torus, Swiss Roll, Inverted Möbius) |
| `src/main.cpp` | Benchmark runner comparing raw vs. reduced dimensions and silhouette improvements |

---

## License

MIT
