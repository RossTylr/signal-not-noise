# Streamlit Interactive Explorer

The keystone project for the signal-not-noise repo.

## Run

```bash
pip install -r ../requirements.txt
streamlit run app.py
```

## What's In Here

An interactive Streamlit app with tabs covering the key concepts from each module:

| Tab | Module | What You Can Do |
|-----|--------|-----------------|
| Dimensions as Questions | 01 | See patient data go from 1D to 4D |
| The Curse | 01 | Watch distances collapse, spheres vanish, k-NN break |
| Redundancy | 02 | Generate high-D data with low intrinsic dimension, see PCA find it |
| PCA Explorer | 03 | Rotate a projection line, apply PCA to digits |
| Nonlinear Methods | 04 | Compare PCA vs t-SNE on the Swiss roll |
| Feature Selection | 05 | See mutual information separate signal from noise |
| Learned Compression | 06 | Train an autoencoder, compare to PCA reconstruction |
| Decision Framework | 07 | Interactive flowchart: which method for your problem? |

## Design Notes

- Every visualisation uses the same colour scheme as the notebooks
- All random seeds are fixed for reproducibility
- Heavy computations (t-SNE) show progress spinners
- The decision framework tab is the synthesis — use it after completing the notebooks
