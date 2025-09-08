```@meta
CurrentModule = Copulas
```

# Troubleshooting

A few quick checks and tips when something looks off.

## Plots in the docs are slow or too heavy

- Reduce N: most examples look fine with N ∈ [300, 2000].
- Use thinner grids: 60×60 is often enough for density heatmaps.
- Prefer seriestype=:steppost for ECDFs to keep artifacts small.

## Conditional distortion looks non-monotone

- For Extreme Value copulas, ensure the fixed value u₂ is strictly in (0,1). At endpoints, atoms/kinks may appear and numerical differences can look like non-monotonicity.
- Use cdf/logcdf instead of pdf/logpdf when comparing theory vs ECDF; the latter is noisier.

## Rosenblatt transform not uniform

- Verify input is on [0,1]^d (use `pseudos` on raw data).
- Check the copula family and parameters match how data were generated.
- For heavy tails, sample size needs to be larger to stabilize tails of ECDFs.

## Fitting gives odd parameters

- The built-in `fit` is a convenience MLE; it can land in local optima.
- Try multiple starting points or a different family.
- For IFM workflows, fit marginals first, then transform to uniforms and fit the copula.

## Common code snippets

```julia
# Pseudo-observations (N×d)
U = pseudos(X)

# Rosenblatt sanity check
ts = range(0.0, 1.0; length=401)
S = reduce(hcat, (rosenblatt(C, U[i, :]) for i in 1:size(U,1)))
EC = [StatsBase.ecdf(S[k, :]) for k in 1:size(U,2)]
```

If you run into a reproducible issue, please open an issue with a small code sample and version info.

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
