```@meta
CurrentModule = Copulas
```

# Public API

This page lists all public docstrings exposed by the package.

## Behavioural contract

Docstrings list supported constructor calls, rather than complete concrete type
signatures. For copula families that support both forms, `Family{d}(args...)`
supplies the dimension through the type and `Family(d, args...)` supplies it at
runtime. The former is the inference-friendly path when `d` is known to the
compiler; this is not a blanket inference guarantee for arbitrary argument types.
Additional storage parameters shown by Julia when printing a type are not
constructor arguments or supported extension points unless explicitly documented.

The public API consists of documented symbols declared with `export` or
`public`, together with the documented methods that Copulas.jl adds to adopted
interfaces such as Distributions.jl and StatsBase.jl. These behaviours follow
semantic versioning; implementation hooks described in the developer guide do
not.

| Area | Public operations | Guaranteed behaviour |
|:--|:--|:--|
| Construction | `Family{d}(parameters...)`, `Family(d, parameters...)` | Both forms select dimension `d`, validate their inputs and construct equivalent models of that family. Parameter values may represent simpler limiting copulas without changing the concrete family returned. A family may document an additional dimension-inferred form. |
| Distribution | `length`, `eltype`, `params`, `cdf`, `logcdf`, `rand` | Every copula is a multivariate distribution with uniform margins and support in the unit hypercube. Vector and matrix sampling preserve dimension and numeric type. |
| Density | `pdf`, `logpdf`, `loglikelihood` | Available for absolutely continuous components. Singular and mixed copulas follow their documented generalized-density semantics and need not possess a Lebesgue density. |
| Marginalization | `subsetdims` | Preserves the requested coordinates and their order. One coordinate yields its univariate marginal. |
| Conditioning | `condition` | Produces the conditional univariate distortion or lower-dimensional distribution, with generalized quantiles where atoms occur. |
| Transforms | `rosenblatt`, `inverse_rosenblatt` | Vector and matrix forms are supported. Round-trip bijectivity is guaranteed only for continuous models without atoms. |
| Dependence | `τ`, `ρ`, `β`, `γ`, `ι`, `λₗ`, `λᵤ`, their documented inverses, `StatsBase.corkendall`, `StatsBase.corspearman` | Results have the documented scalar or pairwise-matrix shape, bounds and symmetry. Closed forms and numerical fallbacks have the same contract. |
| Fitting | `fit`, `CopulaModel` and the StatsBase model interface | Documented family/method pairs return valid fitted models. `CopulaModel` exposes observations, coefficients, covariance when computed, information criteria, residuals and prediction. |
| Composition | `SklarDist` | Distribution operations, marginalization, conditioning and Rosenblatt transforms are expressed on the marginal scales. |
| Utilities | `pseudos`, `measure`, `Nataf` | Rank pseudo-observations, copula rectangle probability, and Nataf correlation correction respectively. |

Public component constructors guarantee their documented mathematical semantics
and supported constructor forms. Public status does not expose undocumented fields,
storage type parameters, intermediate subtype hierarchies, caches, AD backends,
or numerical algorithms as extension contracts. Explicitly documented public
relationships and constructor syntax remain part of the contract.

The public mathematical operations on components are `ϕ`, `max_monotony`, `A`,
`ℓ`, and `𝒲₋₁`. Their public status describes evaluation of supported components;
it does not guarantee that implementing one of them automatically supplies every
copula operation. In particular, sampling, density and conditioning have their
own mathematical and implementation requirements.

Derivative and inverse helpers (`ϕ⁻¹`, `ϕ⁽¹⁾`, `ϕ⁻¹⁽¹⁾`, `ϕ⁽ᵏ⁾`,
`ϕ⁽ᵏ⁾⁻¹`, `dA`, `d²A`, `ellpartial`), the `Distortion` protocol and
the `MGenerator`/`WGenerator` limit representations are internal. See the
[developer guide](@ref developer_fitting) for their current implementation role;
their qualified availability is not a compatibility guarantee.

```@autodocs
Modules = [Copulas]
Private = false
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
