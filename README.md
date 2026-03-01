# Signal, Not Noise

**A Feynman-style learning repository on dimensionality, its curse, and the art of compression.**

> "Compression is the strategy of whoever actually understands the problem."
> -- paraphrasing Andrew Ng

---

## Overview

An interactive learning repository that builds intuition for **dimensionality** -- what it means, why it breaks things, and how to reduce it -- through plain language, working code, and visualisations. Every concept is explained before it is formalised. Every formula earns its place by solving a problem you have already felt.

This is not a textbook. It is a workshop.

## Audience

- Engineers and researchers who work with data but want deeper intuition for *why* certain methods work
- Anyone who has used PCA or t-SNE but could not explain them to a colleague without jargon
- People who believe understanding matters more than memorisation

## Getting Started

```bash
git clone <repo-url> && cd signal-not-noise
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Notebooks

```bash
jupyter notebook
```

Open any module directory (e.g. `01_what_is_a_dimension/`) and start with `01a_building_intuition.ipynb`. Work through the notebooks in order -- each builds on the last.

### Interactive Explorer

The `app/` directory contains a Streamlit application that covers all seven modules with interactive controls, live plots, and a decision framework for choosing the right dimensionality reduction method.

```bash
streamlit run app/app.py
```

## Learning Path

| Module | Topic | Notebooks |
|--------|-------|-----------|
| **01** | What is a Dimension? | 01a Building Intuition, 01b Curse of Dimensionality, 01c Real-World Examples |
| **02** | Redundancy and Structure | 02a Correlation and Redundancy, 02b Intrinsic vs Ambient, 02c Manifold Hypothesis |
| **03** | Linear Methods | 03a Projection Intuition, 03b PCA from Scratch, 03c PCA Applied, 03d SVD and Friends |
| **04** | Nonlinear Methods | 04a Why Linear Fails, 04b t-SNE Explained, 04c UMAP Explained, 04d Comparison Arena |
| **05** | Feature Selection | 05a Filter Methods, 05b Wrapper Methods, 05c Embedded Methods |
| **06** | Learned Compression | 06a Autoencoder Basics, 06b Variational Autoencoders, 06c Comparing to PCA |
| **07** | Applied Thinking | 07a Decision Framework, 07b Simulation Parameter Spaces, 07c Compression as Philosophy |

Each module contains multiple notebooks (a, b, c) that build on each other. Start at 01a and follow the sequence.

## Repository Structure

```
signal-not-noise/
├── README.md
├── requirements.txt
├── 00_orientation/               # How to use this repository
├── 01_what_is_a_dimension/       # Dimensions, curse, intuition
├── 02_redundancy_and_structure/  # Correlation, manifolds
├── 03_linear_methods/            # PCA, SVD, factor analysis
├── 04_nonlinear_methods/         # t-SNE, UMAP, comparisons
├── 05_feature_selection/         # Filter, wrapper, embedded
├── 06_learned_compression/       # Autoencoders, VAEs
├── 07_applied_thinking/          # Decision frameworks, philosophy
├── app/                          # Streamlit interactive explorer
├── utils/                        # Shared plotting and data generation
├── data/                         # Synthetic and real datasets
└── cheatsheets/                  # Quick reference cards
```

## Principles

1. **Feynman first** -- if you cannot explain it simply, you do not understand it
2. **Visual before formal** -- show it, then explain it
3. **Build before import** -- implement from scratch before reaching for a library
4. **Progressive disclosure** -- plain language, then code, then maths
5. **Connected to reality** -- every technique earns a "why would I care?"

## Licence

MIT
