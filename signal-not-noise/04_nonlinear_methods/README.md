# 04 — Nonlinear Methods

When the structure is curved, straight lines won't find it.

## Notebooks

| Notebook | Title | Core Question |
|----------|-------|---------------|
| **04a** | Why Linear Fails | What does PCA miss? |
| **04b** | t-SNE Explained | How do you preserve neighbourhoods? |
| **04c** | UMAP Explained | How do you preserve shape *and* neighbourhoods? |
| **04d** | Comparison Arena | Same data, all methods, side by side |

## The Arc

**04a** opens with the Swiss roll — a 2D surface curled up in 3D. PCA sees it and gets confused. The manifold is simple, but it's not flat. This motivates everything that follows.

**04b** explains t-SNE from intuition: "make nearby points stay nearby in the low-dimensional map." We show how it constructs probability distributions over neighbourhoods in both high-D and low-D, then pushes the low-D version to match. Perplexity demystified.

**04c** covers UMAP — similar goals to t-SNE but with better global structure preservation, faster computation, and a topological foundation. We compare the two head-to-head.

**04d** is the arena: PCA, t-SNE, UMAP, and others on the same datasets. When does each win? What do their failure modes look like?

## After This Module You Should Be Able To

- Recognise when PCA is failing because the structure is nonlinear
- Explain what t-SNE is optimising for, without equations
- Choose between t-SNE and UMAP for a given problem
- Avoid the common pitfalls (over-interpreting t-SNE cluster sizes, etc.)
