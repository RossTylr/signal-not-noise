# 03 — Linear Methods

Find the directions that matter. Ignore the rest.

## Notebooks

| Notebook | Title | Core Question |
|----------|-------|---------------|
| **03a** | Projection Intuition | What does it mean to project data onto a line? |
| **03b** | PCA From Scratch | How does PCA actually find the best directions? |
| **03c** | PCA Applied | What does PCA look like on real data? |
| **03d** | SVD and Friends | What are the relatives of PCA and when do they help? |

## The Arc

**03a** builds physical intuition. Projecting data onto a line is casting a shadow. The best shadow preserves the most spread. We draw it, animate it, and feel it before writing a single equation.

**03b** constructs PCA step by step. Compute the covariance matrix. Find its eigenvectors. These *are* the best directions. We build it from numpy before touching sklearn.

**03c** applies PCA to real datasets — digits, patient vitals, sensor data — and shows the explained variance curve. That satisfying elbow where 3 components capture 95% of the information.

**03d** covers SVD (PCA's more general cousin), factor analysis (when you believe in latent variables), and ICA (when you want independence, not just decorrelation).

## After This Module You Should Be Able To

- Explain PCA to someone using only the word "shadow"
- Implement PCA from scratch in 20 lines of numpy
- Read an explained variance plot and decide how many components to keep
- Know when PCA is the wrong tool
