# CLAUDE.md

## Project: signal-not-noise

A Feynman-style learning repository on dimensionality and dimensionality reduction. Jupyter notebooks + Streamlit keystone app. Python, numpy, matplotlib, sklearn, PyTorch.

## What This Repo Is

An interactive learning repo that teaches dimensionality — what it is, why it breaks things, and how to reduce it — through plain-language notebooks with code and visuals. Every concept is explained before formalised. Every formula earns its place.

## Repo Structure

```
signal-not-noise/
├── CLAUDE.md                         ← You are here
├── README.md                         ← Repo overview for readers
├── BUILDING.md                       ← Build order and checklist
├── NOTEBOOK_TEMPLATE.md              ← Notebook structure standard
├── requirements.txt
├── .gitignore
│
├── 00_orientation/README.md
├── 01_what_is_a_dimension/
│   ├── README.md                     ← Reader-facing module summary
│   ├── INSTRUCTIONS_01a.md           ← Build spec: what to implement
│   ├── INSTRUCTIONS_01b.md
│   ├── INSTRUCTIONS_01c.md
│   └── visuals/
├── 02_redundancy_and_structure/
│   ├── README.md
│   ├── INSTRUCTIONS_02.md            ← Covers 02a, 02b, 02c
│   └── visuals/
├── 03_linear_methods/
│   ├── README.md
│   ├── INSTRUCTIONS_03.md            ← Covers 03a, 03b, 03c, 03d
│   └── visuals/
├── 04_nonlinear_methods/
│   ├── README.md
│   ├── INSTRUCTIONS_04.md
│   └── visuals/
├── 05_feature_selection/
│   ├── README.md
│   ├── INSTRUCTIONS_05.md
│   └── visuals/
├── 06_learned_compression/
│   ├── README.md
│   ├── INSTRUCTIONS_06.md
│   └── visuals/
├── 07_applied_thinking/
│   ├── README.md
│   ├── INSTRUCTIONS_07.md
│   └── visuals/
│
├── app/
│   ├── app.py                        ← Streamlit keystone (complete)
│   └── README.md
├── utils/
│   ├── __init__.py
│   ├── plotting.py                   ← Shared plot style + helpers
│   └── data_generators.py            ← Synthetic data factories
├── cheatsheets/
│   ├── method_comparison.md
│   ├── glossary.md
│   └── maths_reference.md
└── data/
    ├── synthetic/
    └── real/
```

## What's Done vs What Needs Building

**Done:** All scaffolding, READMEs, INSTRUCTIONS specs, utils, cheatsheets, Streamlit app.

**To build:** 19 Jupyter notebooks (.ipynb files). Each has a detailed INSTRUCTIONS_*.md spec.

## How to Build a Notebook

1. Read `NOTEBOOK_TEMPLATE.md` — this is the structure standard
2. Read the relevant `INSTRUCTIONS_*.md` — this is the exact spec (sections, code, visuals, insights)
3. Create the .ipynb in the module directory following the naming convention: `{module}{sub}_{short_title}.ipynb`
4. Follow the build order in `BUILDING.md`

## Notebook Structure (every notebook, no exceptions)

```
1. Header: title, one-sentence version, "build intuition for" list, prerequisites
2. The Story: plain language, no code, no maths — Feynman layer
3. Interactive Code: code cells with visuals, interleaved with markdown
4. The Maths (collapsible): <details> block, optional formal treatment
5. Where This Matters: real-world connection (healthcare, simulation, ops)
6. Feynman Check: 2-4 questions testing explanation ability
```

## Code Conventions

- `import numpy as np` — always
- `from utils.plotting import apply_style, COLOURS` — top of every notebook
- `apply_style()` — call before any plotting
- Random seeds: `np.random.RandomState(42)` or `seed=42`
- Use `utils.data_generators` functions, don't reinvent synthetic data
- Colour scheme: `COLOURS["signal"]` (blue), `COLOURS["noise"]` (red), `COLOURS["accent"]` (amber), `COLOURS["success"]` (green)
- Figure sizes: (10, 5) single, (12, 5) side-by-side, (12, 8) grids
- Save key figures: `plt.savefig("visuals/filename.png", dpi=150, bbox_inches="tight")`

## Writing Style

- **Feynman first**: explain it like you're talking to a smart friend who works in a different field
- **Visual before formal**: show a plot, then explain what it means, then (optionally) show the maths
- **No jargon without earning it**: every term gets a plain-language introduction before first use
- **Warm and direct**: not a textbook, not a lecture — a workshop
- **Progressive disclosure**: plain language → code → maths (collapsible)

## Engineering Preferences

- DRY — shared utils exist, use them
- Every code cell must run without errors
- "Engineered enough" — not fragile, not over-abstracted
- Explicit over clever
- Minimal diff — reuse existing generators and plotting functions

## Build Order

```
01a → 01b → 01c → 02a → 02b → 02c → 03a → 03b → 03c → 03d →
04a → 04b → 04c → 04d → 05a → 05b → 05c → 06a → 06b → 06c →
07a → 07b → 07c
```

Each depends on the previous. Don't skip ahead.

## Dependencies

numpy>=1.24, matplotlib>=3.7, plotly>=5.15, scipy>=1.11, scikit-learn>=1.3,
umap-learn>=0.5, seaborn>=0.12, ipywidgets>=8.0, jupyter>=1.0, streamlit>=1.30,
pandas>=2.0, torch>=2.0, pillow>=10.0

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Run Streamlit app
cd app && streamlit run app.py

# Run a notebook
jupyter notebook 01_what_is_a_dimension/01a_building_intuition.ipynb
```
