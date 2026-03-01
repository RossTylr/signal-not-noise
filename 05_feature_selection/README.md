# 05 — Feature Selection

Sometimes the best compression is just throwing things away.

## Notebooks

| Notebook | Title | Core Question |
|----------|-------|---------------|
| **05a** | Filter Methods | Which features carry signal on their own? |
| **05b** | Wrapper Methods | Which *combination* of features works best? |
| **05c** | Embedded Methods | Can the model itself tell you what matters? |

## The Arc

**05a** starts simple: rank features by how informative they are. Variance thresholds (constant features are useless), correlation filters (drop duplicates), mutual information (nonlinear relationships).

**05b** treats feature selection as search: forward selection (add the best), backward elimination (remove the worst), recursive feature elimination. More expensive, more thorough.

**05c** shows how some models do selection as a side effect of training. Lasso regression zeros out irrelevant features. Tree-based models rank feature importance. Regularisation is implicit feature selection.

## Key Distinction From Modules 03-04

PCA and t-SNE *transform* dimensions into new ones. Feature selection *keeps the original dimensions* — it just picks which ones matter. This has a huge practical advantage: interpretability. A doctor can understand "blood pressure and age matter most." They can't interpret "principal component 3."

## After This Module You Should Be Able To

- Apply three different feature selection strategies to a dataset
- Explain the trade-off between filter, wrapper, and embedded methods
- Know when to select features vs. transform them
