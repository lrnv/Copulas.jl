```@meta
CurrentModule = Copulas
```

# Public API

This page lists all public docstrings exposed by the package.

## Behavioural contract

The public API consists of documented symbols declared with `export` or
`public`, together with the documented methods that Copulas.jl adds to adopted
interfaces such as Distributions.jl and StatsBase.jl. These behaviours follow
semantic versioning; implementation hooks described in the developer guide do
not.

| Area | Public operations | Guaranteed behaviour |
|:--|:--|:--|
| Construction | `Family{d}(parameters...)`, `Family(d, parameters...)` | Both forms select dimension `d`, validate their inputs and construct equivalent models. A family may document an additional dimension-inferred form. |
| Distribution | `length`, `eltype`, `params`, `cdf`, `logcdf`, `rand` | Every copula is a multivariate distribution with uniform margins and support in the unit hypercube. Vector and matrix sampling preserve dimension and numeric type. |
| Density | `pdf`, `logpdf`, `loglikelihood` | Available for absolutely continuous components. Singular and mixed copulas follow their documented generalized-density semantics and need not possess a Lebesgue density. |
| Marginalization | `subsetdims` | Preserves the requested coordinates and their order. One coordinate yields its univariate marginal. |
| Conditioning | `condition` | Produces the conditional univariate distortion or lower-dimensional distribution, with generalized quantiles where atoms occur. |
| Transforms | `rosenblatt`, `inverse_rosenblatt` | Vector and matrix forms are supported. Round-trip bijectivity is guaranteed only for continuous models without atoms. |
| Dependence | `τ`, `ρ`, `β`, `γ`, `ι`, `λₗ`, `λᵤ`, their documented inverses, `StatsBase.corkendall`, `StatsBase.corspearman` | Results have the documented scalar or pairwise-matrix shape, bounds and symmetry. Closed forms and numerical fallbacks have the same contract. |
| Fitting | `fit`, `CopulaModel` and the StatsBase model interface | Documented family/method pairs return valid fitted models. `CopulaModel` exposes observations, coefficients, covariance when computed, information criteria, residuals and prediction. |
| Composition | `SklarDist` | Distribution operations, marginalization, conditioning and Rosenblatt transforms are expressed on the marginal scales. |
| Utilities | `pseudos`, `measure`, `Nataf` | Rank pseudo-observations, copula rectangle probability, and Nataf correlation correction respectively. |

The mathematical primitives documented for public generators and extreme-value
tails are also stable. Concrete internal wrappers, caches, samplers and fallback
selection are deliberately outside this contract.

```@autodocs
Modules = [Copulas]
Private = false
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
