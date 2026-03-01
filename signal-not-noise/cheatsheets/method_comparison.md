# Method Comparison Cheatsheet

## At a Glance

| Method | Type | Linear? | Preserves | Best For | Watch Out |
|--------|------|---------|-----------|----------|-----------|
| **PCA** | Transform | Yes | Global variance | First pass, visualisation, preprocessing | Misses curved structures |
| **SVD** | Transform | Yes | Global variance | Sparse/large data, same as PCA | Same limitations as PCA |
| **Factor Analysis** | Transform | Yes | Latent structure | When you believe in hidden factors | Assumes Gaussian noise |
| **ICA** | Transform | Yes | Independence | Source separation, signals | Needs as many sources as mixtures |
| **t-SNE** | Transform | No | Local neighbourhoods | 2D/3D visualisation | Cluster sizes lie, slow on large data |
| **UMAP** | Transform | No | Local + global shape | Visualisation, clustering | Hyperparameter sensitive |
| **Variance Threshold** | Selection | N/A | Original features | Removing constants | Misses redundancy |
| **Mutual Information** | Selection | N/A | Original features | Nonlinear relevance | Computationally expensive |
| **RFE** | Selection | N/A | Original features | When interpretability matters | Slow, depends on base model |
| **Lasso** | Embedded | Yes | Original features | Sparse linear models | Assumes linear relationships |
| **Tree Importance** | Embedded | No | Original features | Quick feature ranking | Can be biased to high-cardinality |
| **Autoencoder** | Transform | No | Learned compression | Complex nonlinear manifolds | Needs lots of data, harder to tune |
| **VAE** | Transform | No | Smooth latent space | Generation + compression | More complex, may over-smooth |

## Decision Flowchart (Plain Language)

1. **Do you need interpretable features?**
   - Yes → Feature selection (Module 05)
   - No → Continue

2. **Is your data linearly structured?**
   - Yes or unsure → Try PCA first (Module 03)
   - No → Continue

3. **Do you just need a good 2D visualisation?**
   - Yes → UMAP or t-SNE (Module 04)
   - No → Continue

4. **Do you have lots of data and complex structure?**
   - Yes → Autoencoder (Module 06)
   - No → PCA is probably fine

## Rules of Thumb

- **Always try PCA first.** It's fast, deterministic, and interpretable. If it works, stop.
- **t-SNE is for looking, not for learning.** Don't feed t-SNE output into a classifier.
- **UMAP is the better t-SNE** for most purposes, but neither is deterministic.
- **Feature selection when you need to explain** your model to a human.
- **Autoencoders when nothing else works** and you have enough data to train one.
- **Explained variance > 95%** is a common threshold, but domain-dependent.
