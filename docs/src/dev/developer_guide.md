```@meta
CurrentModule = Copulas
```

# [Developer Guide: current contributor architecture](@id developer_fitting)

This page describes the current internal machinery used by package contributors
to add and maintain copula families in this repository.
It focuses on what must be defined for a new copula to work consistently
with the main interfaces (`cdf`, `pdf`, `rand`, `fit`, etc.), without going into
mathematical details.

::: info Target audience

This page is intended for package contributors and advanced users who want to extend 
`Copulas.jl` with new copula families, internal optimizations, or additional features.

:::

!!! warning "Internal interfaces are not covered by SemVer"
    This is architectural documentation, not a supported downstream extension API.
    Every hook, abstract subtype and dispatch pattern shown here may change without
    deprecation unless its behavior is separately documented in the public API.
    Exported/public mathematical objects retain only their public documented
    semantics; this guide does not enlarge that compatibility promise.


# 1. Implementing the public behaviour internally

## 1.1 Overview

The stable user-facing contract is defined by the [Public API](@ref), not by
the implementation recipes on this page. It includes documented methods owned
by `Copulas` as well as documented extensions of ecosystem interfaces:

| Public operation                 | Owning interface                    | Availability |
| -------------------------------- | ----------------------------------- | ----------- |
| `length`                          | `Base`                              | all copulas |
| `cdf`, `params`                   | `Distributions.jl`                  | all copulas |
| `pdf`, `logpdf`, `loglikelihood` | `Distributions.jl`                  | when the documented measure semantics permit it |
| `rand`                            | `Random` / `Distributions.jl`       | all copulas |
| `fit`                             | `Distributions.jl` / `StatsBase.jl` | declared family/method pairs |
| automatic family selection       | `Copulas` / `StatsBase.jl`          | explicit candidate collections |
| copula hypothesis tests          | `Copulas` / `StatsAPI.jl`           | documented procedures and assumptions |
| `corkendall`, `corspearman`       | `StatsBase.jl`                      | all copulas |
| dependence measures, subsetting, conditioning and transforms | `Copulas` | according to their public mathematical preconditions |


However, directly implementing these methods is not always the best way to
fulfil the contract. When implementing a new copula, this document identifies
the internal methods that need to be provided. It is also useful to read the
implementation of an existing copula from the same family alongside this guide.

The table summarizes adopted user-facing interfaces; it does not make their
internal hooks stable. Generic fallbacks provide many operations. Singular and
mixed copulas do not acquire a Lebesgue density or a bijective Rosenblatt
transform merely to satisfy an interface; their documented mathematical
semantics take precedence.


## 1.2 Probability interface (`cdf`, `pdf`, `rand`)

All copulas have a joint `cdf()` over the hypercube. Absolutely continuous
copulas also provide `pdf()` and `logpdf()`; these are not promised for purely
singular copulas, and entropy-based dependence is consequently restricted to
models with an ordinary density.
The `rand(C, n)` method should generate a `d × n` matrix of samples from the copula.

The corresponding public behavior is documented on the [Public API](@ref) page.
Inside this repository, it is currently supplied by the following internal
methods; these hooks may change independently of that behavior:

```julia
struct MyCopula{d, P} <: Copula{d} # Note that the size of the copula must be part of the type. 
    θ::P  # Copula parameter
    MyCopula{d}(θ) where {d} = new{d, typeof(θ)}(θ)
end
MyCopula(d, θ) = MyCopula{d}(θ) # Runtime-dimension convenience constructor
function Distributions.params(C::MyCopula) 
    # It will be assumed that `MyCopula{d}(params(C)...)` reproduces `C`.
    # Keep `MyCopula(d, ...)` as a thin forwarder to this canonical constructor.
    # The return value should be a NamedTuple. 
    return (θ = C.θ,) # Return a named tuple containing the parameters.
end
function Copulas._cdf(C::MyCopula, u)
     # You can safely assume u to be an abstract vector of the right length and inside the hypercube.
     # Return the cdf value on u
end
function Distributions._logpdf(C::MyCopula, u)
    # You can safely assume u to be an abstract vector of the right length and inside the hypercube.
    # Return the logpdf value on u
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::MyCopula, U::AbstractMatrix{<:Real})
    # Fill the d × n matrix U with n samples, stored column-wise, and return it.
    return U
end 
```

Every public copula family provides both `MyCopula{d}(parameters...)`, the
canonical type-stable path, and the thin runtime-dimension convenience form
`MyCopula(d, parameters...)`. When `params(C)` describes an ordinary parametric
instance, `typeof(C)(values(params(C))...)` reconstructs it. Structural models
may expose additional explicitly documented constructors, but must still provide
the two dimension spellings above.

Once defined, the public wrappers integrate the type with the documented
`Distributions.jl` and `Random` operations.

::: info Sampling contract

The matrix `_rand!` method is the sampling primitive. The generic
`Distributions.jl` machinery handles the one-sample/vector interface by
delegating to the matrix sampler, so copula implementations should define
only the matrix method. Before `_rand!` is called, the public interface
validates that the output has `length(C)` rows; implementations may assume
that the matrix has size `d × n` and should not repeat this check. When
several sampling algorithms are available, select among them with ordinary
Julia dispatch on the copula/tail type rather than with a separate routing
trait.

:::

## 1.3 Dependence metrics

Dependence measures — such as Kendall’s τ, Spearman’s ρ, and others listed in [this section](@ref dep_metrics) — are not mandatory.
The package provides default implementations that will work with your copula out-of-the-box. 
However, if some of them can be derived theoretically or numerically with a specific algorithm, 
providing specific methods (with analytical forms when possible) is highly recommended.

| Function         | Description                 | Default behavior            |
| ---------------- | --------------------------- | --------------------------- |
| `Copulas.τ(C)`   | Kendall’s tau               | Default numerical estimator |
| `Copulas.ρ(C)`   | Spearman’s rho              | Default numerical estimator |
| `Copulas.β(C)`   | Blomqvist's beta            | Default numerical estimator |
| `Copulas.γ(C)`   | Gini's gamma                | Default numerical estimator |
| `Copulas.ι(C)`   | Copula entropy              | Default numerical estimator |
| `Copulas.λₗ(C)`  | Lower tail dependence        | Default extrapolation-based |
| `Copulas.λᵤ(C)`  | Upper tail dependence       | Default extrapolation-based |

If your copula provides closed-form expressions for any of these, overriding the default
methods will improve both accuracy and performance.

```julia
Copulas.τ(C::MyCopula) = ...
Copulas.ρ(C::MyCopula) = ...
...
```

## 1.4 Conditioning and subsetting

The conditioning framework works by default, and you can already use
`condition(C::MyCopula, dims, us)`. No additional public method is required.

Inside Copulas.jl, specialized families currently optimize this path through the
following internal hooks:

```julia
Copulas.conditional_copula(C::MyCopula, dims, us) = ...
Copulas.distortion(C::MyCopula, dims, us, i) = ...
```

These bindings are documented for contributors working on Copulas.jl itself. They
are not public extension points and are not covered by SemVer. Downstream packages
should prefer the generic `condition` interface; if a missing fast path matters,
please coordinate its implementation upstream. If the hooks are not defined,
conditioning falls back to the generic path.

* The first binding returns the copula of the conditional random vector.
  The conditioning framework combines it with the conditional marginals in a
  `SklarDist` when more than one coordinate remains.
* The second binding returns the `i`th conditional marginal. It must
  return a `<:Distortion`, itself a
  `Distributions.ContinuousUnivariateDistribution` supported on `[0, 1]`.
  Implement its `cdf` and either `pdf` or `logpdf`; implementing `quantile` is
  also recommended. The returned object is used as a functor to distort
  marginals as follows:

```julia
(D::Distortion)(::Distributions.Uniform) = D # Already provided by the framework.
(D::Distortion)(X::Distributions.UnivariateDistribution) = DistortedDist(D, X) # Default.
```

This is how we enable conditioning on the SklarDist level.

!!! tip "Look at existing distortions"
    Take a look in the `src/UnivariateDistributions/Distortions` folder for examples, there are plenty. 



## 1.5 Fitting interface

The fitting interface allows your copula to work with `fit(::Type{CopulaModel}, ...)`
and the general estimation framework.

### Implementing a custom fitting method

These hooks are contributor-facing internals. A custom estimator does not need
parameter reparameterizations merely to return its fitted copula.

| Method                              | Purpose                                                       |
| ----------------------------------- | ------------------------------------------------------------- |
| `_available_fitting_methods(CT, d)` | Declares supported methods (`:mle`, `:itau`, `:ibeta`, etc.)  |
| `_fit(CT, U, ::Val{:method})`       | Core fitting routine returning `(copula, meta)`               |

Minimal skeleton for a custom fitting method:

```julia
_available_fitting_methods(::Type{MyCopula}, d) = (:mymethod,)

function _fit(::Type{MyCopula}, U, ::Val{:mymethod})
    θ̂ = .... # do things.
    return MyCopula(size(U, 1), θ̂), (; θ̂,)
end
```

Alternatively, reuse a generic fitting engine rather than defining `_fit`.

### Opting into generic fitting methods

| Method                              | Purpose                                                       |
| ----------------------------------- | ------------------------------------------------------------- |
| `_example(CT, d)`                   | Returns a representative instance used for defaults           |
| `_unbound_params(CT, d, params)`    | Maps parameter tuple → unconstrained vector                   |
| `_rebound_params(CT, d, α)`         | Inverse map for optimizer results                             |

Example minimal skeleton:

```julia
_example(::Type{MyCopula}, d) = MyCopula(d, default_parameters...)
_unbound_params(::Type{MyCopula}, d, params) = [log(params.θ)]
_rebound_params(::Type{MyCopula}, d, α) = (; θ = exp(α[1]))
_available_fitting_methods(::Type{MyCopula}, d) = (:mle, :itau, :ibeta,) # or others...

# No _fit definition: the generic engine consumes these hooks.
```

Each fitting method is dispatched on `Val{:method}` for performance and clarity.

Declare only methods supported by the model: MLE needs an evaluable likelihood,
and rank-based fitting needs the corresponding dependence measures and an
identifiable parameterization. Covariance and confidence intervals require
additional regularity and a supported covariance procedure; returning a fitted
copula does not guarantee their availability. Data have shape `d × n`, so the
dimension is `size(U, 1)`, not the number of observations.

## 1.6 Hypothesis tests

The public procedures and their assumptions are documented in
[Hypothesis testing](@ref hypothesis_testing). They return a common
`CopulaTest <: StatsAPI.HypothesisTest`, with `pvalue`, `teststatistic`
and `StatsBase.nobs` accessors.

To contribute a new procedure, add an internal hypothesis description and
implement its statistic and calibration in `src/CopulaTest.jl`. A public function
validates procedure-specific options and calls `_run_copula_test`, which handles
common data validation and result assembly. Dispatch is on the hypothesis:
there is no registry of hypothetical statistic/calibration combinations.
The two multiplier procedures share the numerical resampling loop.

Add a small independent statistic/process oracle and reproducibility checks to
`test/operations/hypothesis_testing.jl`, then document the precise procedure,
reference, applicability and finite-sample conventions. Composite GOF must replay
the estimator specification for every bootstrap sample.
These implementation details are internal and are not covered by SemVer.

# 2. Specific sub-APIs
Some families of copulas in `Copulas.jl` have additional internal structures or specific mathematical representations.
This section summarizes the bindings required for the most common ones: **Archimedean** and **Extreme Value** copulas.

Each sub-API is based on the general interface described above (`cdf`, `logpdf`, `rand`, `fit`, etc.); however, in these cases, the requirements are different.

## 2.1 Archimedean copulas

Archimedean copulas are defined by a generator function ϕ. To implement a new Archimedean family, define a subtype of
[`Generator`](@ref) and implement the following:

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
const MyArchimedeanCopula{d,T} = ArchimedeanCopula{d, MyGenerator{T}}
ϕ(G::MyGenerator, t) = ...
max_monotony(G::MyGenerator) = ...
Distributions.params(G::MyGenerator) = (θ = G.θ,)
```

### Required methods for a generator `G`

| Method                              | Purpose                                                            | Required    |
| ------------------------------------| ------------------------------------------------------------------ | ----------- |
| `max_monotony(G)`                   | Maximum degree of monotonicity (controls validity in d dimensions) | ✅          |
| `Distributions.params(G)`           | Return parameters as a `NamedTuple`                                | ✅          |
| `ϕ(G, t)`                           | Generator function                                                 | ✅          |
| `ϕ⁻¹(G, t)`                         | Generator function inverse                                         | ⚙️ Optional |
| `ϕ⁽¹⁾(G, t)`                        | Generator function derivative                                      | ⚙️ Optional |
| `ϕ⁻¹⁽¹⁾(G, t)`                      | Generator function derivative of the inverse                       | ⚙️ Optional |
| `ϕ⁽ᵏ⁾(G, k::Int, t)`                | Generator function kth derivative                                  | ⚙️ Optional |
| `ϕ⁽ᵏ⁾⁻¹(G, k::Int, t; start_at=t)`  | Generator function kth derivative's inverse                        | ⚙️ Optional |
| `𝒲₋₁(G, d::Real)`                  | Inverse Williamson transform; integer specializations are preserved | ⚙️ Optional |


The generator definition enables the generic Archimedean construction in valid
dimensions. CDF evaluation also needs a working inverse, supplied analytically
or numerically. Density requires appropriate derivatives and measure semantics;
radial sampling requires an evaluable and sampleable inverse Williamson law.
Do not infer that every singular or numerically difficult generator automatically
supports every operation. Check the paths exercised by the proposed model before
adding specializations, and benchmark numerical alternatives when relevant.

Only fitting routines or dependence metrics need to be added if the defaults are insufficient.

::: info Other generator interfaces

1) In-package one-parameter families can use the internal `AbstractUnivariateGenerator` hierarchy.
2) If your generator is a Frailty, then there is `FrailtyGenerator`
3) If you know the radial part, use `𝒲 === WilliamsonGenerator` directly. 
4) If you are lost, just open an issue ;)

:::

## 2.2 Extreme-Value copulas

Extreme-value copulas are represented by an [`ExtremeValueCopula`](@ref)
containing a stable tail dependence function object, [`Tail`](@ref). The
dimension-free mathematical identity is

```math
C(\boldsymbol u)=\exp\{-\ell(-\log\boldsymbol u)\}.
```

The EV API deliberately separates the mathematical family from computational
capabilities.

### `Tail`: the mathematical STDF interface

A multivariate EV tail should subtype `Tail` and implement its STDF:

```julia
struct MyTail{T} <: Copulas.Tail
    θ::T
end

Copulas.ℓ(tail::MyTail, x) = ...
Distributions.params(tail::MyTail) = (; θ = tail.θ)
```

`Tail` is valid by default for every `d >= 2`. Override
`_is_valid_in_dim(tail, d)` only when the mathematical family has additional
dimensional restrictions.

`ExtremeValueCopula(d, tail)` checks `_is_valid_in_dim(tail, d)` at
construction time.

### `BivariatePickandsTail`: the scalar bivariate Pickands capability

`BivariatePickandsTail <: Tail` means that the tail provides the native scalar
bivariate Pickands representation `A(t)`, and therefore can use the specialized
Pickands derivative, density, conditioning, and sampling machinery:

```julia
struct MyTail{T} <: Copulas.BivariatePickandsTail
    θ::T
end

Copulas.A(tail::MyTail, t::Real) = ...
```

Its default validity is `d == 2`. If the same mathematical family also has a
valid STDF in higher dimension, opt in explicitly:

```julia
Copulas.ℓ(tail::MyTail, x) = ...
Copulas._is_valid_in_dim(::MyTail, d::Int) = d >= 2
```

`_is_valid_in_dim` is an internal validity hook. Defining a multivariate `ℓ`
alone does not override the bivariate restriction or provide a sampler.

This is the pattern used by families such as Logistic, Galambos,
Hüsler-Reiss, Mixed, extremal-``t``, and Cuadras-Augé.

::: info Why keep `BivariatePickandsTail`?

A multivariate family can still have exceptionally good analytic formulas
in dimension two. `BivariatePickandsTail` lets the package retain `A`, `dA`, `d²A`,
conditional distortions, and the Ghoudi sampler without pretending that the
mathematical family stops at ``d=2``.

:::

### Constructor convention

The canonical EV constructor encodes the dimension in the type:

```julia
FamilyCopula{d}(params...)
```

The runtime-dimension form is only syntactic sugar:

```julia
FamilyCopula(d, params...)
```

Scalar and exchangeable families do **not** infer an implicit bivariate
dimension. For example, use `GalambosCopula{2}(2.3)` (or the runtime sugar
`GalambosCopula(2, 2.3)`), not `GalambosCopula(2.3)`.

Structured parameterizations follow the same rule. Their canonical forms are,
for example,

```julia
HuslerReissCopula{d}(Γ)
tEVCopula{d}(ν, R)
TawnCopula{d}(α, weights)
AsymGalambosCopula{d}(α, weights)
BC2Copula{d}(a)
MOCopula{d}(λ)
EmpiricalEVCopula{d}(U)
```

An inferred-dimension constructor may additionally be provided only when a
single parameter determines `d` immediately and unambiguously, such as
`HuslerReissCopula(Γ)` or `MOCopula(λ)`. It must validate to the same
mathematical copula as the canonical `{d}` constructor. Do not add inference
machinery merely to support a shorter spelling.

Full subset parameterizations obey the same contract:

```julia
TawnCopula{d}(dep, asy)
AsymGalambosCopula{d}(dep, asy)
```

with `FamilyCopula(d, ...)` as runtime sugar. These multi-parameter forms do
not infer `d`.

A public family may store scalar and matrix parameterizations in the same tail
type and use the parameter type for dispatch. A matrix parameterization can
still select a specialized bivariate kernel without being converted to a
different representation. Do not use the concrete stored parameter type as
the public family identity.

### Density interface

For ``x_i=-\log u_i``, an absolutely continuous EV density can be written

```math
c(\boldsymbol u)
=
\frac{\exp\{-\ell(\boldsymbol x)\}}{\prod_i u_i}
\sum_{\pi\in\Pi_d}
(-1)^{d+|\pi|}
\prod_{B\in\pi}\partial_B\ell(\boldsymbol x),
```

where ``\Pi_d`` is the set of partitions of ``\{1,\ldots,d\}``.

The generic multivariate path needs mixed STDF partials, but these are not an
additional requirement for a new family. The common `_mixed_partial` utility
computes mixed derivatives with `ForwardDiff` and is shared by the EV density
machinery and generic conditioning.

| Method | Meaning | Required |
|---|---|---|
| `ℓ(tail, x)` | stable tail dependence function | ✅ |
| `_ellpartial_signlog(tail, x, I)` | stable sign/log-absolute mixed partial | ⚙️ Optional |
| `A`, `dA`, `d²A` | native bivariate Pickands kernel | ⚙️ Optional |

By default, `_ellpartial_signlog` is obtained from `ℓ` through the shared
automatic-differentiation helper, and `ellpartial(tail, x, I)` is reconstructed
from that sign/log representation. A new multivariate EV tail therefore needs
to implement **only `ℓ`** for the generic density path when its mixed derivatives
exist and its implementation supports the required AD inputs. This does not
represent singular mass. Override
`_ellpartial_signlog` only when an analytic expression is materially more
stable or faster.

Density selection itself uses ordinary Julia dispatch. In ``d=2``, a `BivariatePickandsTail`
uses the native Pickands derivative kernel. The generic `ExtremeValueCopula{d}`
method uses the partition formula above, so a family-specific `_logpdf` method
is only needed when the family provides a genuinely different numerical
algorithm.


### Conditioning and Rosenblatt in higher dimensions

No separate extreme-value Rosenblatt algorithm is required. The generic
conditioning framework is dimension-agnostic: `distortion` obtains
conditional marginals from mixed derivatives of the copula CDF, while
`rosenblatt` and `inverse_rosenblatt` build the usual sequence of conditional
distributions from that interface.

Consequently, smooth multivariate EV families whose numerical CDF/STDF path is
compatible with automatic differentiation inherit `condition`, `rosenblatt`,
and `inverse_rosenblatt` in `d > 2`. The Logistic and Galambos families are
covered explicitly by the architecture tests. In `d = 2`,
`BivariatePickandsTail` retains the faster native `BivEVDistortion` path.

This generic guarantee is computational rather than purely mathematical.
Families whose multivariate STDF relies on numerical probability routines that
materialize `Float64` values may require specialized derivatives or a distortion
instead of the ForwardDiff fallback. Check the current family implementation
rather than inferring compatibility from its mathematical formula. Discrete
spectral EV models can contain singular components, so a global Lebesgue
density and the ordinary smooth conditional-derivative construction need not
exist in general.

### Sampling interface and dispatch

The required public behavior is simply

```julia
rand(C, n)
```

Extreme-value sampling is selected directly through Julia dispatch on the
copula dimension and the concrete tail type. There is no separate sampling
backend trait or routing layer.

For a `BivariatePickandsTail` in ``d=2``, the generic extreme-value method uses the native
Ghoudi/Pickands sampler:

```julia
function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2,<:BivariatePickandsTail},
    X::AbstractMatrix{T},
) where {T<:Real}
    # Ghoudi/Pickands algorithm
end
```

A family with its own exact multivariate sampler implements `_rand!` directly
for its concrete tail type. There is no universal multivariate EV sampler
provided by the STDF definition alone:

```julia
function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:MyTail},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    return _my_exact_rand!(rng, C.tail, X)
end
```

Because the fallback leaves `d` generic, a method on a concrete `MyTail` is
naturally more specific and needs no intersection-resolving specialization.
This keeps algorithm selection entirely in Julia's dispatch system. Logistic
retains its native bivariate Ghoudi/Pickands route, while Galambos,
Hüsler-Reiss, Mixed, and extremal-``t`` use their exact family samplers.

Algorithm-specific helpers such as `_discrete_spectral_rand!` or family
spectral samplers may be used internally
when they represent a reusable numerical algorithm rather than a routing
layer.

!!! warning "Internal, non-stable API"
    Algorithm-specific sampling helpers and `_ellpartial_signlog` are
    contributor-facing internals. Public user code should call `rand`, `cdf`,
    `pdf`, etc.

### Source organization

One source file corresponds to one mathematical family. A family can contain
multiple internal representations in that file; for example an optimized
bivariate tail and a general matrix/subset tail. This keeps family semantics
together while allowing dispatch to specialize the computational backend.

See [Extreme Value family](@ref Extreme_theory) for the user-facing theory,
constructor table, bivariate Ghoudi development, and model documentation.

## 2.3 Elliptical copulas (internal architecture)

Elliptical copulas arise from the dependence structure of multivariate **elliptical distributions** such as the Gaussian or Student-t.
The current implementation shares code through

```julia
Copulas.EllipticalCopula{d, MT}
```

where `MT` describes the stored correlation-matrix representation, not the
associated distribution. This is the current internal hierarchy, not a stable
public storage contract.

Elliptical copulas are characterized by a correlation matrix `Σ` and, optionally, additional shape parameters (e.g. degrees of freedom `ν` for the t-copula).


### Required methods

| Method                    | Purpose                                                | Required       |
| ------------------------- | ------------------------------------------------------ | -------------- |
| `U(C)`                   | Return the standardized univariate distribution instance | ✅            |
| `N(C)`                   | Return a callable constructing the multivariate distribution from `Σ` | ✅ |
| `Distributions.params(C)` | Return parameters as a `NamedTuple`                    | ✅              |

Minimal outline:

```julia
struct MyEllipticalCopula{d,MT} <: Copulas.EllipticalCopula{d,MT}
    Σ::MT
    function MyEllipticalCopula{d}(Σ) where {d}
        size(Σ) == (d, d) || throw(DimensionMismatch("expected a $d×$d matrix"))
        matrix = Matrix{Float64}(Σ)
        Copulas.make_cor!(matrix)  # normalize a copy; validate the family as needed
        return new{d,typeof(matrix)}(matrix)
    end
end
MyEllipticalCopula(d, Σ) = MyEllipticalCopula{d}(Σ)

# Required bindings
Copulas.U(C::MyEllipticalCopula) = Normal()
Copulas.N(C::MyEllipticalCopula) = Σ -> MvNormal(Σ)
Distributions.params(C::MyEllipticalCopula) = (Σ = C.Σ,)
```

The example uses Gaussian distributions. For runtime shape parameters, the
object-based hooks are essential: a Student family uses `TDist(C.df)` for `U(C)`
and `Σ -> MvTDist(C.df, Σ)` for `N(C)`. Type-based hooks remain convenient when
the distributions do not depend on runtime fields, through the existing adapters.

The generic evaluation and sampling paths require the corresponding operations
on those distributions (including univariate quantiles and multivariate density
or sampling). Defining these hooks does not manufacture a missing multivariate
CDF or guarantee compatibility with AD-based conditioning.

Most elliptical families (Gaussian, t, Laplace, power-exponential, GED) can be implemented 
simply by changing their `U` and `N` definitions, reusing the same generic machinery. 
We only have gaussian and student, but you could propose other ones. 


!!! note "Analytical and numerical stability"
    Although most elliptical copulas work out-of-the-box through numerical evaluation of multivariate CDFs and densities,
    it is **highly recommended** to provide analytical or semi-analytical forms for the following when possible:

    * Tail coefficients (`λ_L`, `λ_U`)
    * Dependence measures (`τ`, `ρ`)
    * Specialized `logpdf` or `rand` implementations (e.g. variance-mixture sampling for Laplace or generalized t families)

    Such implementations significantly improve numerical stability and performance of the overall package.


# 3. Complete Examples
This section provides practical examples of complete copula implementations.  
Each example illustrates how to make a new family compatible with the main API of `Copulas.jl`.


## 3.1 Generic copula example — *MardiaCopula*

The `MardiaCopula` is a simple **bivariate** copula that mixes the Fréchet upper, lower, and independent copulas using a single parameter θ ∈ [−1, 1].  
It serves as a minimal example of how to implement a copula *from scratch* without relying on the `Generator` or `Tail` sub-APIs.

```@example generic_copula_example
using Copulas, Distributions, Random

struct MardiaCopula{P} <: Copulas.Copula{2}
    θ::P
    function MardiaCopula(θ)
        -1 <= θ <= 1 || throw(ArgumentError("θ must be in [-1,1]"))
        θf = float(θ)
        return new{typeof(θf)}(θf)
    end
end
MardiaCopula(d, θ) = d == 2 ? MardiaCopula(θ) :
    throw(DimensionMismatch("MardiaCopula is bivariate"))
Distributions.params(C::MardiaCopula) = (; θ = C.θ,)
function Copulas._cdf(C::MardiaCopula, u)
    # The joint CDF follows Mardia’s formulation:
    θ = C.θ
    u1, u2 = u
    term1 = (θ^2 * (1 + θ) / 2) * min(u1, u2)
    term2 = (1 - θ^2) * u1 * u2
    term3 = (θ^2 * (1 - θ) / 2) * max(u1 + u2 - 1, 0)
    return term1 + term2 + term3
end
```

Boundary parameters must not make a constructor return another copula type.
Keeping `MardiaCopula(0)`, `MardiaCopula(1)`, and `MardiaCopula(-1)` in the
`MardiaCopula` family makes inference independent of runtime values. Handle
equivalent independence or Fréchet-bound cases inside numerical methods when a
generic formula is undefined or a dedicated path is materially better.


### Defining the PDF and Random Generation

This copula has no analytical density.
Instead, we define a sampling rule that randomly selects between three dependence structures with probabilities determined by θ:

```@example generic_copula_example
Distributions._logpdf(C::MardiaCopula, u) = NaN

function Distributions._rand!(rng::Distributions.AbstractRNG, C::MardiaCopula, X::AbstractMatrix{T}) where {T<:Real}
    θ = C.θ
    p = [θ^2 * (1 + θ) / 2, 1 - θ^2, θ^2 * (1 - θ) / 2]
    for j in axes(X, 2)
        u1, u2 = rand(rng, Distributions.Uniform(0,1), 2)
        z = rand(rng, Distributions.Categorical(p))
        if z == 1
            u = min(u1, u2)
            X[1, j] = u; X[2, j] = u
        elseif z == 2
            X[1, j] = u1; X[2, j] = u2
        else
            u = max(u1 + u2 - 1, 0)
            X[1, j] = u; X[2, j] = 1 - u
        end
    end
    return X
end
```

### Usage

```@example generic_copula_example
Random.seed!(123)
C = MardiaCopula(2, 0.8)
U = rand(C, 2000)
```

The copula now works seamlessly with all standard methods:

```@example generic_copula_example
cdf(C, [0.3, 0.7])
pdf(C, [0.3, 0.7])
D = condition(C, 1, 0.3)
rand(D, 10)
```

### Fitting interface and integration

To make the copula compatible with `Distributions.fit` and the unified `CopulaModel` interface,
we provide a minimal `_fit` definition using a dependence-based measure — in this case, **Gini’s γ**.

```@example generic_copula_example

Copulas._available_fitting_methods(::Type{<:MardiaCopula}, d::Int) = (:igamma,)

function Copulas._fit(::Type{<:MardiaCopula}, U::AbstractMatrix, ::Val{:igamma})
    γ̂ = Copulas.corgini(U')[1, 2]
    θ  = sign(γ̂) * abs(γ̂)^(1/3)
    θ  = clamp(θ, -1.0, 1.0)
    Ĉ = MardiaCopula(2, θ)
    return Ĉ, (; θ̂ = (; θ = θ), γ̂ = γ̂, method = :igamma)
end
```

This approach bypasses the need for a log-likelihood function (since the copula lacks a Lebesgue density)
while maintaining compatibility with all higher-level fitting utilities.

Remark that we could also opt-in the default moment matching methods, but for that we need to specify parameter relaxations through the following: 

```@example generic_copula_example
Copulas._unbound_params(::Type{MardiaCopula}, d, params) = [atanh(clamp(params.θ, -1 + eps(), 1 - eps()))]
Copulas._rebound_params(::Type{MardiaCopula}, d, α) = (; θ = tanh(α[1]) )
Copulas._example(::Type{<:MardiaCopula}, d::Int) = MardiaCopula(2, 0.5)
```

And we need to change our availiable methods: 
```@example generic_copula_example
Copulas._available_fitting_methods(::Type{<:MardiaCopula}, d::Int) = (:igamma, :itau, :irho, :ibeta)
```



### Example: fitting and model summary

```@example generic_copula_example
using StatsBase

# Short syntax, leveraging the generics: 
println(fit(MardiaCopula, U, :ibeta))

# Long syntax, using our new method: 
M = fit(CopulaModel, MardiaCopula, U; method = :igamma, vcov = false)
println(M)
```



!!! note "Example purpose"
    This example illustrates a fully functional copula defined *from scratch*.
    Once these minimal methods are implemented,
    the family automatically integrates with the `Distributions.jl` and `StatsBase` ecosystems.


## 3.2 Archimedean example — a positive Clayton generator

For a small contributor example, reproduce the positive-parameter Clayton
generator. This is not a new Nelsen family; production code should use
`Copulas.ClaytonGenerator`. Restricting the example to finite `θ > 0` keeps
its mathematical domain and numerical formulas consistent.

```@example generator_contributor
using Copulas, Distributions

struct ExampleClaytonGenerator{T} <: Copulas.Generator
    θ::T
    function ExampleClaytonGenerator(θ::Real)
        isfinite(θ) && θ > 0 || throw(ArgumentError("θ must be finite and positive"))
        value = float(θ)
        new{typeof(value)}(value)
    end
end

Copulas.max_monotony(::ExampleClaytonGenerator) = Inf
Distributions.params(G::ExampleClaytonGenerator) = (; θ=G.θ)
Copulas.ϕ(G::ExampleClaytonGenerator, t) = exp(-log1p(G.θ * t) / G.θ)

G = ExampleClaytonGenerator(2.0)
C = ArchimedeanCopula{2}(G)
reference = ClaytonCopula{2}(2.0)
u = [0.3, 0.8]
@assert isapprox(cdf(C, u), cdf(reference, u))
cdf(C, u)
```

This checks evaluation through the generic inverse rather than defining a
second inverse solely for the example. Optional derivative, inverse and radial
specializations are internal numerical hooks, not requirements for this CDF
example. Density and sampling need separate checks of the generic numerical
paths; fitting requires the opt-in described in Section 1.5.

A family that accepts boundary parameters must preserve its family in the
constructor and implement its mathematical behavior. Do not return a different
generator type from the constructor. In-package families use the internal
`limit_kind` mechanism where appropriate; that mechanism is not a stable
downstream extension protocol. This example rejects boundaries instead.

## 3.3 Extreme-value example — a bivariate logistic tail

The following contributor example implements the smooth, finite-parameter
interior of the logistic model. Production code should use `Copulas.LogTail`.
Its internal Pickands capability restricts it to dimension two by default.

```@example tail_contributor
using Copulas, Distributions, LogExpFunctions

struct ExampleLogTail{T} <: Copulas.BivariatePickandsTail
    θ::T
    function ExampleLogTail(θ::Real)
        isfinite(θ) && θ > 1 || throw(ArgumentError("θ must be finite and greater than one"))
        value = float(θ)
        new{typeof(value)}(value)
    end
end

function Copulas.A(tail::ExampleLogTail, t::Real)
    return exp(LogExpFunctions.logaddexp(tail.θ * log(t),
        tail.θ * log1p(-t)) / tail.θ)
end
Distributions.params(tail::ExampleLogTail) = (; θ=tail.θ)

C = ExtremeValueCopula{2}(ExampleLogTail(2.5))
reference = LogCopula{2}(2.5)
u = [0.4, 0.7]
@assert isapprox(cdf(C, u), cdf(reference, u))
cdf(C, u), pdf(C, u)
```

For this smooth bivariate model, the Pickands machinery supplies derivatives
and the Ghoudi sampler. That conclusion does not extend to every convex
Pickands function: kinks may encode singular mass and require specialized
density or conditional methods.

To extend a family to higher dimensions, supply a valid multivariate STDF,
opt into those dimensions through `_is_valid_in_dim`, and provide an appropriate
sampler. Convexity and the usual bounds alone are not sufficient to characterize
a multivariate STDF. Fitting is a separate opt-in; use either the custom-method
or generic-engine pattern in Section 1.5.


# 4. Testing architecture

The test suite proves the documented public API while avoiding the repetition of
expensive numerical checks for every copula family.

## 4.1 Current organization

The SemVer-stable surface consists of symbols exported or declared `public` by
`Copulas`, together with documented methods extending interfaces such as
`Distributions`, `StatsBase`, and `Random`. Undocumented internal hooks are
not covered by this guarantee.

Tests are organized as follows:

- `test/bestiary.jl` is the central registry of public copula representatives;
- `test/api/` checks constructors, components, compositions, and standalone
  public utilities;
- `test/operations/` checks each public copula operation;
- `test/correctness/` contains independent mathematical, statistical, and
  numerical references that span operations or describe a family;
- `test/extensions/` contains optional-extension regressions.

- Every bestiary representative receives the applicable public-operation contracts.
- Expensive mathematical or numerical mechanisms are validated once against an
  independent oracle; `which` may be used locally to avoid repeating the same
  proof for several families selecting the same implementation.
- Parameter- or dimension-dependent branches invisible to method dispatch receive
  focused regressions. Historical constructor reductions additionally receive
  direct source/target equivalence tests.

## 4.2 Adding a copula family

After implementing and documenting `MyCopula`:

1. Add the cheapest ordinary representative to `ALL_COPULA_CASES` in
   `test/bestiary.jl`:

   ```julia
   copula_case(MyCopula, d, parameters...)
   ```

   This automatically checks the public `MyCopula{d}(...)` and
   `MyCopula(d, ...)` constructors and subjects every representative to the
   family-wide operation contracts. Constructors must infer their concrete
   family without a return union. Constructor keywords and exceptional
   numerical tolerances remain optional metadata.
2. Add another bestiary entry whenever another dimension, representation or
    parameter regime exercises materially different code. Every such entry receives
    the applicable public-operation contracts.
3. If the copula can be singular or mixed, implement the appropriate internal
   measure-style trait. This determines whether density and invertible
   Rosenblatt requirements apply; do not duplicate that classification in the
   tests.
4. If the family exposes a new generic numerical mechanism, add one independent
   oracle under `test/correctness/`. If it adds a specialization of an existing
   operation, add its generic-equivalence check to the corresponding
   `test/operations/` file. A specialization without a valid generic fallback
   needs an independent identity instead.
5. Add a focused family regression only for information not implied by those
   proofs, such as a published value, atom mass, or reproduced bug. Record
   value-dependent equivalences in `test/correctness/reduction_graph.jl`; its
   source constructors are automatically added to the bestiary and therefore
   receive every public operation contract. Fitting methods advertised by
   package dispatch are discovered automatically.

Archimedean generators and extreme-value tails used by bestiary copulas are
extracted automatically for their component contracts. Add a separate copula
representative when a new public generator or tail would otherwise be absent.

## 4.3 Adding a public feature

When adding or changing a public operation `newstuff(C::Copula)`:

1. Declare and document the public interface, including applicability,
   return shape, bounds, and errors. For an adopted external interface, document
   the supported methods without redeclaring its symbol.
2. Create or extend `test/operations/newstuff.jl`. Apply a cheap contract helper
   to every applicable entry in `COPULA_FIXTURES`.
3. Test every generic implementation mechanism once against an independent
   mathematical or statistical oracle.

Use `which` locally to identify distinct implementations reached by the bestiary,
and validate each distinct implementation once against the generic method or an
independent oracle. Add focused regressions for important value-dependent branches
that share the same Julia method.
