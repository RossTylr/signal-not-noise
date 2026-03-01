# Glossary — Plain Language Definitions

**Ambient Dimension**
The number of columns in your spreadsheet. The space your data is *recorded* in, whether or not it needs all of it.

**Autoencoder**
A neural network that learns to compress data through a bottleneck and reconstruct it. The bottleneck layer IS the compressed version.

**Correlation**
When two measurements tend to move together. If tall people tend to be heavier, height and weight are correlated. Correlated features are partially redundant.

**Covariance Matrix**
A table showing how every pair of features moves together. The diagonal tells you each feature's spread. The off-diagonals tell you relationships.

**Curse of Dimensionality**
The collection of problems that emerge in high-dimensional spaces: distances lose meaning, volumes concentrate on boundaries, and you need exponentially more data to cover the space.

**Dimensionality Reduction**
Any technique that takes data with many dimensions and represents it with fewer, while preserving the important structure.

**Distance Concentration**
In high dimensions, the nearest and farthest points become almost equally far away. Distances lose their ability to distinguish "close" from "far."

**Eigenvalue**
A number that tells you how much variance a particular direction (eigenvector) captures. Bigger eigenvalue = more important direction.

**Eigenvector**
A direction in your data space that doesn't change when you apply the covariance matrix — it just gets scaled. PCA uses these as its principal components.

**Explained Variance**
The fraction of your data's total spread captured by a set of components. If 3 components explain 95% of variance, the other dimensions contribute 5% of information.

**Feature Selection**
Choosing which original dimensions to keep and which to discard. Unlike PCA, you don't create new features — you just pick the best existing ones.

**Intrinsic Dimension**
The actual number of independent ways your data varies. A sheet of paper in a room has intrinsic dimension 2, even though it lives in 3D ambient space.

**Latent Space**
The compressed representation space inside an autoencoder or similar model. Points in latent space are the "codes" for your original data.

**Manifold**
A smooth surface that may be curved or twisted, embedded in a higher-dimensional space. The manifold hypothesis says real data lives on such surfaces.

**Manifold Hypothesis**
The claim that real-world high-dimensional data actually lies on (or near) a much lower-dimensional manifold. This is why dimensionality reduction works.

**Mutual Information**
A measure of how much knowing one variable tells you about another. Unlike correlation, it captures nonlinear relationships.

**PCA (Principal Component Analysis)**
A method that finds the directions of greatest variance in your data and projects onto them. It's finding the best "shadows" to cast.

**Perplexity**
t-SNE's key hyperparameter. Roughly: how many neighbours each point should consider. Low perplexity = focus on very local structure. High perplexity = consider broader patterns.

**Projection**
Casting a shadow of high-dimensional data onto a lower-dimensional surface. PCA finds the surface that preserves the most information.

**SVD (Singular Value Decomposition)**
A matrix factorisation that decomposes data into three components. PCA is a special case of SVD. More numerically stable and works on non-square matrices.

**Swiss Roll**
A classic test dataset: a 2D surface rolled up like a Swiss cake roll in 3D. Used to demonstrate why linear methods (PCA) fail on curved manifolds.

**t-SNE**
A nonlinear method that preserves neighbourhood structure: points that are close in high-D stay close in the 2D map. Great for visualisation, not for other tasks.

**UMAP**
Similar to t-SNE but faster, with better global structure preservation, and based on topological ideas. Currently the default choice for nonlinear visualisation.

**Variance**
How spread out a measurement is. High variance = the data varies a lot along that dimension. Zero variance = every point has the same value (useless dimension).

**VAE (Variational Autoencoder)**
An autoencoder that forces the latent space to follow a smooth probability distribution. This makes the latent space navigable — you can interpolate and generate.
