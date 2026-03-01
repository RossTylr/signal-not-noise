# 06 — Learned Compression

Let a neural network find the compression function for you.

## Notebooks

| Notebook | Title | Core Question |
|----------|-------|---------------|
| **06a** | Autoencoder Basics | What happens when you squeeze data through a bottleneck? |
| **06b** | Variational Autoencoders | What if the bottleneck had structure? |
| **06c** | Comparing to PCA | When does learned compression beat linear methods? |

## The Arc

**06a** builds the simplest possible autoencoder: input → small hidden layer → reconstruct input. The hidden layer IS the compressed representation. We visualise what it learns and compare it to PCA's output.

**06b** adds structure to the bottleneck. VAEs don't just compress — they learn a smooth, navigable latent space. You can interpolate between data points and generate new ones. This connects dimensionality reduction to generative modelling.

**06c** asks the pragmatic question: when is the neural network overhead worth it? On what kinds of data do autoencoders beat PCA? When is PCA "good enough"?

## After This Module You Should Be Able To

- Build a simple autoencoder in PyTorch
- Explain what the bottleneck layer represents
- Visualise and interpret learned latent spaces
- Make a reasoned choice between PCA and autoencoders for a given problem
