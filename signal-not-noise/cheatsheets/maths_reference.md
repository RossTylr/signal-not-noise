# Maths Reference

For when you want the formal version. Every formula here is explained in plain language
in the corresponding notebook.

---

## Distances

**Euclidean distance** in d dimensions:

$$d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$$

**Distance concentration**: For random points in [0,1]^d, as d → ∞:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0$$

All pairwise distances converge to the same value.

---

## Volume

**Hypersphere volume** in d dimensions, radius r:

$$V_d(r) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} r^d$$

**Ratio to enclosing hypercube** → 0 as d → ∞.

**Shell concentration**: Fraction of volume in interior of [0,1]^d
that is more than ε from any boundary:

$$f_{\text{interior}} = (1 - 2\varepsilon)^d$$

At d=100, ε=0.01: only ~13% of volume is "interior."

---

## PCA

**Covariance matrix** of centred data X (n × d):

$$C = \frac{1}{n-1} X^T X$$

**Eigendecomposition**: C = QΛQ^T where Q contains eigenvectors (principal directions)
and Λ contains eigenvalues (variance along each direction).

**Projection** onto top k components:

$$X_{\text{reduced}} = X Q_k$$

where Q_k is the d × k matrix of the top k eigenvectors.

**Explained variance ratio** for component i:

$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

---

## SVD

Any matrix X (n × d) decomposes as:

$$X = U \Sigma V^T$$

- U (n × n): left singular vectors
- Σ (n × d): diagonal matrix of singular values
- V (d × d): right singular vectors (these are PCA directions)

**Truncated SVD** (keep top k): X ≈ U_k Σ_k V_k^T — best rank-k approximation.

---

## t-SNE

**High-D similarity** between points i and j (conditional probability):

$$p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2\sigma_i^2)}$$

**Low-D similarity** (Student-t with 1 degree of freedom):

$$q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}$$

**Objective**: Minimise KL divergence between P and Q:

$$\text{KL}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

---

## Mutual Information

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Zero if X and Y are independent. Higher = more information shared.
Unlike correlation, captures nonlinear dependencies.

---

## Autoencoder

**Encoder**: f: ℝ^d → ℝ^k (compress)
**Decoder**: g: ℝ^k → ℝ^d (reconstruct)
**Loss**: ‖x - g(f(x))‖² (reconstruction error)

The bottleneck dimension k forces the network to find the k most
important factors of variation.

**VAE addition**: Encoder outputs μ and σ for each latent dimension.
Latent code z ~ N(μ, σ²). Loss adds KL divergence to standard normal:

$$\mathcal{L} = \|x - g(z)\|^2 + \text{KL}(q(z|x) \| \mathcal{N}(0,I))$$
