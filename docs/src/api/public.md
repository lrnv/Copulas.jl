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
constructor arguments or supported extension points unless explicitly
documented. The deliberate exception is
`SklarDist{CopulaType,Tuple{MarginTypes...}}`, whose two family selectors are a
supported fitting-target syntax; this does not expose any further representation
details of `SklarDist`.

The public API consists of documented symbols declared with `export` or
`public`, together with the documented methods that Copulas.jl adds to adopted
interfaces such as Distributions.jl and StatsBase.jl. These behaviours follow
semantic versioning; implementation hooks described in the developer guide do
not.

A public copula object has dimension `d ≥ 2`. The unique mathematical
one-dimensional copula `C(u)=u` is represented by the ordinary univariate
`Uniform()` distribution when marginalization or conditioning leaves one
coordinate; Copulas.jl does not expose a `Copula{1}` model. Families may impose
stronger dimension restrictions.

| Area | Public operations | Guaranteed behaviour |
|:--|:--|:--|
| Construction | `Family{d}(parameters...)`, `Family(d, parameters...)` | Both forms select dimension `d ≥ 2`, validate their inputs and construct equivalent models of that family. Parameter values may represent simpler limiting copulas without changing the concrete family returned. A family may document an additional dimension-inferred form or a stronger dimension restriction. |
| Distribution | `length`, `eltype`, `params`, `cdf`, `logcdf`, `rand` | Every copula is a multivariate distribution with uniform margins (except the [`EmpiricalCopula`](@ref)) and support in the unit hypercube. Vector and matrix sampling preserve dimension and numeric type. |
| Density | `pdf`, `logpdf`, `loglikelihood` | Available for absolutely continuous components. Singular and mixed copulas follow their documented generalized-density semantics and need not possess a Lebesgue density. A family-defined boundary value is preserved; if its formula is indeterminate (`NaN`) on the boundary of the unit hypercube, the public interface uses the valid density representative `pdf = 0` (`logpdf = -Inf`). |
| Marginalization | `subsetdims` | Preserves the requested coordinates and their order. One coordinate yields its univariate marginal. |
| Conditioning | `condition` | Produces the conditional univariate distortion or lower-dimensional distribution, given points or intervals of the conditioned coordinates, with generalized quantiles where atoms occur. A discrete observation of a `SklarDist` conditions on its latent interval. |
| Transforms | `rosenblatt`, `inverse_rosenblatt` | Vector and matrix forms are supported. Round trips hold almost surely when successive conditional CDFs are continuous and invertible on their supports; atomic conditionals of a copula need not give a bijection or a uniform forward transform. A `SklarDist` with discrete margins takes the distributional transform of each atom, so `rosenblatt(rng, X, x)` is random on atoms and `inverse_rosenblatt(X, s)` inverts it in law. |
| Dependence | `τ`, `ρ`, `β`, `γ`, `ι`, `λₗ`, `λᵤ`, `StatsBase.corkendall`, `StatsBase.corspearman` | Copula-level scalar summaries operate on public copulas, hence `d ≥ 2`. Data-level scalar `τ`, `ρ`, `β`, and `γ` likewise require at least two rows; their multivariate normalizations are degenerate at `d=1`. Results otherwise have the documented scalar or pairwise-matrix shape, bounds and symmetry. Closed forms and numerical fallbacks have the same contract. Parameter inversions used by fitting are internal. |
| Fitting, inference, and selection | `fit`, `CopulaModel`, `fitted_distribution`, `infer`, `selected_model`, and `selection_table` | Documented family/method pairs return valid point estimates. `CopulaModel` retains only the fitted result, original data, fitted likelihood, and replay recipe. Inference owns covariance state, while automatic selection keeps candidate comparisons separate from the winning model. Estimator registries, execution hooks, result storage, and parameter reconstruction remain internal; see the [fitting interface](@ref fitting_interface). |
| Hypothesis testing | `IndependenceCopulaTest`, `ExchangeabilityCopulaTest`, `RadialSymmetryCopulaTest`, `ExtremeValueCopulaTest`, `GOFCopulaTest`, `pvalue`, `teststatistic` | Each procedure applies its documented statistic and calibration under its stated assumptions and returns a `CopulaTest`; see [hypothesis testing](@ref hypothesis_testing). |
| Composition | `SklarDist` | Distribution operations, marginalization, conditioning and Rosenblatt transforms are expressed on the marginal scales. `SklarDist{CopulaType,Tuple{MarginTypes...}}` is additionally a supported fitting target. |
| Generator extension | `Generator`, `ϕ`, `max_monotony`, `Distributions.params` | Subtyping `Generator` and implementing these three mathematical operations is a supported way to define a custom Archimedean generator. Optional derivative, inverse, radial, fitting, cache, and dispatch hooks remain internal. |
| Utilities | `pseudos`, `measure`, `Nataf` | Rank pseudo-observations, copula rectangle probability, and Nataf correlation correction respectively. |

For continuous distributions, `eltype` follows the Distributions.jl convention:
it is the default numeric type allocated by `rand`. Parameterized copulas
propagate the numeric representation of their parameters, composite copulas
promote their components, and parameter-free copulas default to `Float64`.
`rand!` may instead target any compatible real-valued buffer type.

Public component constructors guarantee their documented mathematical semantics
and supported constructor forms. Public status does not expose undocumented fields,
storage type parameters, intermediate subtype hierarchies, caches, AD backends,
or numerical algorithms as extension contracts. Explicitly documented public
relationships and constructor syntax remain part of the contract.

The public mathematical operations on components are documented below. Their
public status describes evaluation of supported components; it does not imply
that defining one operation supplies every copula capability. The current
implementation machinery is described separately in the
[developer guide](@ref developer_fitting) and is not part of this contract.

```@autodocs
Modules = [Copulas]
Private = false
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
