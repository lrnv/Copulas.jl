```@meta
CurrentModule = Copulas
```

# Quick API tour

This is a quick, practical tour of the public API. It shows how to construct copulas, build Sklar distributions, compute dependence metrics, subset and condition models, use Rosenblatt transforms, and fit models. For background, theory, details, and model descriptions, see the Manual.

## Copulas and Sklar distributions

You can construct a copula object with their respecive constructors. They behave like multivariate distribution from `Distributions.jl` and respect their API: 

```@example 1
using Copulas, Distributions, Random, StatsBase
# A 3-variate Clayton copula
C = ClaytonCopula(3, 2.0)
U = rand(C, 5)
Distributions.loglikelihood(C, U)
```

To build multivariate dsitributions, you can compose a copula with marginals via Sklar’s theorem:

```@example 1
X₁, X₂, X₃ = Gamma(2,3), Beta(1,5), LogNormal(0,1)
C2 = GumbelCopula(3, 1.7)
D  = SklarDist(C2, (X₁, X₂, X₃))
rand(D, 3)
pdf(D, rand(3))
```

## Dependence metrics

You can get scalar dependence metrics at copula level: 

```@example 1
(
    kendall_tau = Copulas.τ(C),
    spearm_rho = Copulas.ρ(C),
    blomqvist_beta = Copulas.β(C),
    gini_gamma = Copulas.γ(C), 
    entropy_iota = Copulas.ι(C), 
    lower_tail_dep = Copulas.λₗ(C), 
    upper_tail_dep = Copulas.λᵤ(C)
)
```

Pairwise matrices of bivarite versions are available through `StatsBase.corkendall(C), StatsBase.corspearman(C),Copulas.corblomqvist(C), Copulas.corgini(C), Copulas.corentropy(C), Copulas.corlowertail(C)`, and  `Copulas.coruppertail(C)` respectively. 

Same functions work passing a dataset instead of the copula for their empirical counterpart. 

## Measure and transforms

The `measure` function measures hypercubes under the distribution fo the copula. You can access rosenbaltt transfromation of a copula (or a sklardist) through the `rosenblatt` and `inverse_rosenblatt` functions: 

```@example 1
Copulas.measure(C, (0.1,0.2,0.3), (0.9,0.8,0.7))
x = rand(D, 100)
u = rosenblatt(D, x)
x2 = inverse_rosenblatt(D, u)
maximum(abs.(x2 .- x))
```

## Subsetting and conditioning

You can subset the dimensions of a model through `subsetdims()`, and you can condition a model on some of its marginals with `condition()`:

```@example 1
S23 = subsetdims(C2, (2,3))
StatsBase.corkendall(S23)
Dj  = condition(C2, 2, 0.3)  # distortion of U₁ | U₂ = 0.3 (d=2)
Distributions.cdf(Dj, 0.95)
Dc  = condition(D, (2,3), (0.3, 0.2))
rand(Dc, 2)
```

## Fitting

Fit both marginals and copula from raw data (Sklar):

```@example 1
X = rand(D, 500)
M = fit(CopulaModel, SklarDist{GumbelCopula, Tuple{Gamma,Beta,LogNormal}}, X; copula_method=:mle)
```

Directly fit a copula from pseudo-observations U:

```@example 1
U = pseudos(X)
Ĉ = fit(GumbelCopula, U; method=:itau)
```

Notes
- `fit` chooses a reasonable default per family; pass `method`/`copula_method` to control it.
- Common methods: copulas `:mle`, `:itau`, `:irho`, `:ibeta`; Sklar `:ifm` (parametric CDFs) and `:ecdf` (pseudo-observations).

!!! info "Where next?"

    The documentation is organized into two main sections: 
    - Manual: Copulas and Sklar Distributions (theory), Dependence measures, Conditioning and subsetting, Fitting.
    - Bestiary: model families and features.
