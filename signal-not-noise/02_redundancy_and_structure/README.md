# 02 — Redundancy and Structure

Most of your dimensions are saying the same thing in different accents.

## Notebooks

| Notebook | Title | Core Question |
|----------|-------|---------------|
| **02a** | Correlation and Redundancy | How many of your features are just echoes? |
| **02b** | Intrinsic vs Ambient Dimension | What's the real shape hiding inside your data? |
| **02c** | The Manifold Hypothesis | Why does dimensionality reduction work at all? |

## The Arc

**02a** shows that real-world data is massively redundant. If height and weight are correlated, carrying both is carrying partial duplicates. We visualise correlation matrices, compute effective dimensionality, and show how 50 measurements often collapse to 5 independent signals.

**02b** introduces the critical distinction: the space your data *lives in* (ambient dimension) vs the space your data *moves through* (intrinsic dimension). A sheet of paper in a room is 2D embedded in 3D. Your data is similar.

**02c** makes the bold claim that sits underneath all of modern ML: real-world high-dimensional data lies on low-dimensional manifolds. This is *why* compression works. Without this, we'd be lost.

## After This Module You Should Be Able To

- Look at a correlation matrix and estimate how many "real" dimensions exist
- Explain the difference between intrinsic and ambient dimensionality
- Articulate why the manifold hypothesis is the foundation of dimensionality reduction
