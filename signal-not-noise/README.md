# Signal, Not Noise

**A Feynman-style learning repository on dimensionality, its curse, and the art of compression.**

> "Compression is the strategy of whoever actually understands the problem."
> — paraphrasing Andrew Ng

---

## What This Is

A hands-on learning repo that builds intuition for **dimensionality** — what it means, why it breaks things, and how to reduce it — using plain language, interactive code, and visuals. Every concept is explained before it's formalised. Every formula earns its place by solving a problem you've already felt.

This isn't a textbook. It's a workshop.

## Who This Is For

- Engineers and researchers who work with data but want deeper intuition for *why* certain methods work
- Anyone who's used PCA or t-SNE but couldn't explain them to a colleague without jargon
- People who believe understanding > memorisation

## Learning Path

```
01  What is a dimension?              ← Build intuition from scratch
02  Redundancy and structure          ← Most dimensions are lying to you
03  Linear methods (PCA, SVD)         ← Find the directions that matter
04  Nonlinear methods (t-SNE, UMAP)   ← When the structure is curved
05  Feature selection                 ← Sometimes just throw them away
06  Learned compression               ← Let a neural net find it
07  Applied thinking                  ← When to use what, and why it matters
```

Each module contains multiple notebooks (a, b, c...) that build on each other. Start at 01a. Don't skip ahead.

### Notebooks (23 total)

| Module | Notebooks |
|--------|-----------|
| **01** What is a Dimension? | 01a Building Intuition, 01b Curse of Dimensionality, 01c Real-World Examples |
| **02** Redundancy & Structure | 02a Correlation & Redundancy, 02b Intrinsic vs Ambient, 02c Manifold Hypothesis |
| **03** Linear Methods | 03a Projection Intuition, 03b PCA from Scratch, 03c PCA Applied, 03d SVD & Friends |
| **04** Nonlinear Methods | 04a Why Linear Fails, 04b t-SNE Explained, 04c UMAP Explained, 04d Comparison Arena |
| **05** Feature Selection | 05a Filter Methods, 05b Wrapper Methods, 05c Embedded Methods |
| **06** Learned Compression | 06a Autoencoder Basics, 06b Variational Autoencoders, 06c Comparing to PCA |
| **07** Applied Thinking | 07a Decision Framework, 07b Simulation Parameter Spaces, 07c Compression as Philosophy |

## Setup

```bash
git clone <repo-url> && cd signal-not-noise
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Notebooks

```bash
jupyter notebook
```

Navigate to any module directory (e.g. `01_what_is_a_dimension/`) and open a notebook. Start with `01a_building_intuition.ipynb` and work through in order.

## Interactive Explorer (Streamlit App)

The `app/` directory contains a **Streamlit application** that lets you interact with the key concepts from every module in one place. It's the capstone — once you've worked through the notebooks, this is where you play.

```bash
streamlit run app/app.py
```

The app covers all 7 modules with interactive sliders, live plots, and a decision framework for choosing the right dimensionality reduction method. Includes a PyTorch autoencoder you can train in the browser.

## Repo Structure

```
signal-not-noise/
├── README.md
├── requirements.txt
├── 00_orientation/               # How to use this repo
├── 01_what_is_a_dimension/       # Dimensions, curse, intuition
├── 02_redundancy_and_structure/  # Correlation, manifolds
├── 03_linear_methods/            # PCA, SVD, factor analysis
├── 04_nonlinear_methods/         # t-SNE, UMAP, comparisons
├── 05_feature_selection/         # Filter, wrapper, embedded
├── 06_learned_compression/       # Autoencoders, VAEs
├── 07_applied_thinking/          # Decision frameworks, philosophy
├── app/                          # Streamlit interactive explorer
├── utils/                        # Shared plotting, data generation
├── data/                         # Synthetic and real datasets
└── cheatsheets/                  # Quick reference cards
```

## Principles

1. **Feynman First** — If you can't explain it simply, you don't understand it
2. **Visual Before Formal** — Show it, then explain it
3. **Build Don't Import** — Implement from scratch before using sklearn
4. **Progressive Disclosure** — Plain language → code → maths (optional)
5. **Connected to Reality** — Every technique gets a "why would I care?"

## Licence

MIT
