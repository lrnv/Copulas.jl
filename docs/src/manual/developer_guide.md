```@meta
CurrentModule = Copulas
```

# [Developer Guide: Extending the Copulas.jl API](@id developer_fitting)

This page provides a complete overview of the internal developer API required
to add new copula families to `Copulas.jl`.
It focuses on what must be defined for a new copula to work consistently
with the main interfaces (`cdf`, `pdf`, `rand`, `fit`, etc.), without going into
mathematical details.

!!! info "Target audience"
    This page is intended for package contributors and advanced users who want to extend 
    `Copulas.jl` with new copula families, internal optimizations, or additional features.


# 1. The main API

## 1.1 Overview

Every copula type in `Copulas.jl` provides an extensive set of methods, to integrate correctly with the ecosystem: (non-exhaustive table)

| Method                           | Purpose                             | Required    |
| -------------------------------- | ----------------------------------- | ----------- |
| `length(C)`                      | Dimension d of the copula           | ✅          |
| `cdf(C, u)`                      | Cumulative distribution function    | ✅          |
| `pdf(C, u)`                      | Density                             | ✅          |
| `logpdf(C, u)`                   | Joint log density                   | ✅          |
| `rand(C, n)`                     | Random generation                   | ✅          |
| `params(C)`                      | Return parameters as a `NamedTuple` | ✅          |
| `fit(::Type{<:MyCopula}, C, u)`  | Model fitting interface             | ⚙️ Optional |
| `τ(C)`, `ρ(C)`, etc...           | Dependence metrics                  | ⚙️ Optional |
| `λₗ(C)`, `λᵤ(C)`                  | Tail dependence coefficients        | ⚙️ Optional |
| `condition(C, dims, us)`         | Conditional copula                  | ⚙️ Optional |
| `subsetdims(C, dims)`            | Conditional copula                  | ⚙️ Optional |
| `rosenblatt(C, u)`               | Rosenblatt transformation           | ⚙️ Optional |
| `inverse_rosenblatt(C, u)`       | Inverse Rosenblatt transformation   | ⚙️ Optional |


However, direct implementation of these methods is not always the best way to fullfill the contract. 
If you want to implement a new copula, this document will quide you into the right methods that you need to implement.
The easiest way is probably to look at another copula's code, choosing a copula *from the same family as yours* if possible, and then 
reading this code in parralell to this doucment. 


## 1.2 Probability interface (`cdf`, `pdf`, `rand`)

All copulas have a joint `cdf()` over the hypercube, and they might have a `pdf()` too (optional but highly recomended).
The `rand(C, n)` method should generate an `d × n` matrix of samples from the copula. 

Public API : `rand(C, n)`, `cdf(C, u)`, `pdf(C, u )`, `logpdf(C, u )`, `loglikelihood(C, u )`. 
For these methodes to work corectly, you need to overwrite a few internal methods, as in the following minimal example: 

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
    return (θ = C.θ,) # Return a named tuple with the parametrisation. 
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

Once defined, these automatically integrate with the `Copulas.jl` and `Distributions.jl` interface.

!!! info "Sampling contract"
    The matrix `_rand!` method is the sampling primitive. The generic
    `Distributions.jl` machinery handles the one-sample/vector interface by
    delegating to the matrix sampler, so copula implementations should define
    only the matrix method. Before `_rand!` is called, the public interface
    validates that the output has `length(C)` rows; implementations may assume
    that the matrix has size `d × n` and should not repeat this check. When
    several sampling algorithms are available, select among them with ordinary
    Julia dispatch on the copula/tail type rather than with a separate routing
    trait.


## 1.3 Dependence metrics

Dependence measures — such as Kendall’s τ, Spearman’s ρ, and others listed in [this section](@ref dep_metrics) — are not mandatory.
The package provides default implementations that will work with your copula out-of-the-box. 
However, if some of them can be derived theoretically or numerically with a specific algorithm, 
providing specific methods (with analytical forms when possible) is highly recommended.

| Function         | Description                 | Default behavior            |
| ---------------- | --------------------------- | --------------------------- |
| `Copulas.τ(C)`   | Kendall’s tau               | Default numerical estimator |
| `Copulas.ρ(C)`   | Spearman’s rho              | Default numerical estimator |
| `Copulas.β(C)`   | Kendall’s tau               | Default numerical estimator |
| `Copulas.γ(C)`   | Spearman’s rho              | Default numerical estimator |
| `Copulas.ι(C)`   | Kendall’s tau               | Default numerical estimator |
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

The conditining framework works by default, and you can already use `condition(C::MyCopula, dims, us)`.
You don’t need to override anything else unless your copula has a closed form for conditional distributions 
(univariate or multivariate), or a semi-closed-form that is better than our generics.  
If it does, then it is **highly recomended** that you overwrite these two bindings: 

```julia
ConditionalCopula(C::MyCopula, dims, us) = ...
DistortionFromCop(C::MyCopula, dims, us, i) = ...
```

These allow `Copulas.jl` to build conditional distributions internally.
If not defined, conditioning will fall back to a generic (and thus slower) path.

* The first binding returns a `SklarDist`, containing the conditional copula as a copula, 
  and conditional marginals as the marginals. This literally represent the conditional 
  distribution of the random vector, but already splitted by the Sklar's theorem. 
* The second binding corresponds to the ith marginal of the first. It must return an object
  `<:Distortion`, which itself subtypes `Distributions.ContinuousUnivariateDistribution` supported on [0,1],
  corresponding to the distortion. You need to implement its `cdf`, `pdf` or `logpdf` (as you want), 
  and eventually (recomended) the `quantile` function. The returned object will be used as a 
  functor to distord marginals as follows (already implemented):  

```julia
(D::Distortion)(::Distributions.Uniform) = D # Always, no need to implement its already there. 
(D::Distortion)(X::Distributions.UnivariateDistribution) = DistortedDist(D, X) # the default. 
```

This is how we enable conditioning on the SklarDist level.

!!! tip "Look at existing distortions"
    Take a look in the `src/UnivariateDistributions/Distortions` folder for examples, there are plenty. 



## 1.5 Fitting interface

The fitting interface allows your copula to work with `fit(::Type{CopulaModel}, ...)`
and the general estimation framework.

### Required internal methods

| Method                              | Purpose                                                       |
| ----------------------------------- | ------------------------------------------------------------- |
| `_example(CT, d)`                   | Returns a representative instance used for defaults           |
| `_unbound_params(CT, d, params)`    | Maps parameter tuple → unconstrained vector                   |
| `_rebound_params(CT, d, α)`         | Inverse map for optimizer results                             |
| `_available_fitting_methods(CT, d)` | Declares supported methods (`:mle`, `:itau`, `:ibeta`, etc.)  |
| `_fit(CT, U, ::Val{:method})`       | Core fitting routine returning `(copula, meta)`               |

Example minimal skeleton:

```julia
_example(::Type{MyCopula}, d) = MyCopula(d, default_parameters...)
_unbound_params(::Type{MyCopula}, d, params) = [log(params.θ)]
_rebound_params(::Type{MyCopula}, d, α) = (; θ = exp(α[1]))
_available_fitting_methods(::Type{MyCopula}, d) = (:mle, :itau)

function _fit(::Type{MyCopula}, U, ::Val{:mle})
    θ̂ = optimize_mle(U)
    return MyCopula(size(U, 2), θ̂), (; θ̂, optimizer = :lbfgs, converged = true)
end
```

Each fitting method is dispatched on `Val{:method}` for performance and clarity.

**Automatic compatibility**  
Once the above methods are implemented, your family becomes automatically compatible with:

- `fit`, `CopulaModel`
- `StatsBase.vcov`, `StatsBase.confint`
- `Distributions.loglikelihood`
- `StatsBase.aic`, `StatsBase.bic`, `Copulas.aicc`, `Copulas.hqc`




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


Once the generator defines `ϕ`, and `max_monotony`, all functions such as
`cdf`, `logpdf`, and `rand` become available automatically through
`ArchimedeanCopula`’s generic implementation. The default we have for the rest of the methods are pretty efficient, so, even if a theoretical version exists, time it against our generics it might be slower. 

Only fitting routines or dependence metrics need to be added if the defaults are insufficient.

!!! info "Other generator interfaces"
    1) If you generator has only a one-dimensional parametrisation, then you might look at the `UnivariateGenerator<:Generator` interface that is a bit easier. 
    2) If your generator is a Frailty, then there is `FrailtyGenerator`
    3) If you know the radial part, use `𝒲 === WilliamsonGenerator` directly. 
    4) If you are lost, just open an issue ;)




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
```

This is the pattern used by families such as Logistic, Galambos,
Hüsler-Reiss, Mixed, extremal-``t``, and Cuadras-Augé.

!!! info "Why keep `BivariatePickandsTail`?"
    A multivariate family can still have exceptionally good analytic formulas
    in dimension two. `BivariatePickandsTail` lets the package retain `A`, `dA`, `d²A`,
    conditional distortions, and the Ghoudi sampler without pretending that the
    mathematical family stops at ``d=2``.

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

A public constructor may map a ``2\times2`` matrix to a specialized scalar tail
and a larger matrix to a general tail. Do not use the concrete stored tail type
as the public family identity.

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
to implement **only `ℓ`** for the generic density path. Override
`_ellpartial_signlog` only when an analytic expression is materially more
stable or faster.

Density selection itself uses ordinary Julia dispatch. In ``d=2``, a `BivariatePickandsTail`
uses the native Pickands derivative kernel. The generic `ExtremeValueCopula{d}`
method uses the partition formula above, so a family-specific `_logpdf` method
is only needed when the family provides a genuinely different numerical
algorithm.


### Conditioning and Rosenblatt in higher dimensions

No separate extreme-value Rosenblatt algorithm is required. The generic
conditioning framework is dimension-agnostic: `DistortionFromCop` obtains
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
materialize `Float64` values may require a specialized distortion instead of
the ForwardDiff fallback; the current multivariate Hüsler--Reiss and
extremal-``t`` numerical kernels fall in this category. Likewise, discrete
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
    C::ExtremeValueCopula{d,<:BivariatePickandsTail},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    # Ghoudi/Pickands algorithm
end
```

A family with its own exact multivariate sampler implements `_rand!` directly
for its concrete tail type:

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

## 2.3 Elliptical copulas

Elliptical copulas arise from the dependence structure of multivariate **elliptical distributions** such as the Gaussian or Student-t.
In `Copulas.jl`, every elliptical family is represented as

```julia
EllipticalCopula{d, D}
```

where `D` is the associated multivariate distribution type (for instance, `MvNormal` or `MvTDist`).

Elliptical copulas are characterized by a correlation matrix `Σ` and, optionally, additional shape parameters (e.g. degrees of freedom `ν` for the t-copula).


### Required methods

| Method                    | Purpose                                                | Required       |
| ------------------------- | ------------------------------------------------------ | -------------- |
| `U(::Type{CT})`           | Return the univariate elliptical distribution          | ✅              |
| `N(::Type{CT})`           | Return the multivariate elliptical distribution        | ✅              |
| `Distributions.params(C)` | Return parameters as a `NamedTuple`                    | ✅              |

Minimal outline:

```julia
struct MyEllipticalCopula{d,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function MyEllipticalCopula(Σ)
        if LinearAlgebra.isdiag(Σ)
            return IndependentCopula(size(Σ,1))
        end
        make_cor!(Σ)  # normalize to correlation matrix
        return new{size(Σ,1), typeof(Σ)}(Σ)
    end
end

# Required bindings
U(::Type{<:MyEllipticalCopula}) = UnivariateDistribution
N(::Type{<:MyEllipticalCopula}) = MultivariateDistribution
Distributions.params(C::MyEllipticalCopula) = (Σ = C.Σ,)
```

Once these bindings are defined, all core functionality —
`cdf`, `logpdf`, and `rand` —
is automatically available through the generic `EllipticalCopula` implementation in `Copulas.jl`.

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
    function MardiaCopula(d, θ)
        @assert d ==2
        if !(-1 <= θ <= 1)
            throw(ArgumentError("θ must be in [-1,1]"))
        elseif θ == 0
            return IndependentCopula(2)
        elseif θ == 1
            return MCopula(2)
        elseif θ == -1
            return WCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end
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


## 3.2 Archimedean example — *Nelsen2Copula*

The `Nelsen2Copula` is a simple **Archimedean** copula defined by the generator

$$\varphi(t) = (1 + θ * t)^{\frac{-1}{θ}}, \quad θ > 0.$$

This example demonstrates how to define a new Archimedean copula family using the `Generator` sub-API.
Once the generator is defined, all the usual functions (`cdf`, `pdf`, `rand`, `fit`, etc.)
are automatically inherited from the generic `ArchimedeanCopula` implementation.

### Defining the generator

Every Archimedean copula in `Copulas.jl` is built from a subtype of `Generator` that defines
the core functional behavior of the family.

```@example generic_copula_example
struct Nelsen2Generator{T} <: Copulas.AbstractUnivariateGenerator # subtype of Generator
    θ::T
    function Nelsen2Generator(θ)
        if θ < 1
            throw(ArgumentError("θ must be ≥ 1"))
        elseif θ == 1
            return Copulas.WGenerator()
        elseif θ == Inf
            return Copulas.MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end

# Validity and parameters
Copulas.max_monotony(G::Nelsen2Generator) = Inf
Distributions.params(G::Nelsen2Generator) = (; θ = G.θ,)

# Generator and its inverse
Copulas.ϕ(G::Nelsen2Generator, s) = (1 + G.θ * s)^(-1 / G.θ)
Copulas.ϕ⁻¹(G::Nelsen2Generator, t) = (t^(-G.θ) - 1) / G.θ # This is not mandatory

# Nice alias: 
const Nelsen2Copula{d, T} = ArchimedeanCopula{d, Nelsen2Generator{T}}
```

### Building the Archimedean copula

With our alias, we can directly construct the copula through: 

```@example generic_copula_example
C = Nelsen2Copula(2, 3.5)
```

The resulting object already supports all standard functionality from the general API:

```@example generic_copula_example
u = [0.3, 0.8]
cdf(C, u)
pdf(C, u)
```

### Fitting interface

Because the `ArchimedeanCopula` type already provides estimation routines for Kendall’s τ and Spearman’s ρ,
no explicit `_fit` definition is needed unless you wish to override the defaults.

To verify:

```@example generic_copula_example
Copulas._unbound_params(::Type{<:Nelsen2Generator}, d, θ) = [log(θ.θ - 1)]
Copulas._rebound_params(::Type{<:Nelsen2Generator}, d, α) = (; θ = exp(α[1]) + 1)
Copulas._available_fitting_methods(::Type{Nelsen2Copula}, d) = (:ibeta, :mle)
Copulas._example(::Type{Nelsen2Copula}, d) = Nelsen2Copula(d, 2.5)
Copulas._θ_bounds(::Type{<:Nelsen2Generator}, d) = (1, Inf) # specific to the fitting methods of one-parameter archimedean copulas. 
```

### Example: quick γ-based fit

```@example generic_copula_example
Random.seed!(123)
U = rand(C, 250)
Fit = fit(Nelsen2Copula, U; method=:ibeta) # igamma is the default method
Fit
```

For completeness, you can also use the model-based interface using the `:mle` default fit
for one-parameter Archimedean copulas:

```@example generic_copula_example
FitModel = fit(CopulaModel, Nelsen2Copula, U; method=:mle, start = 1.5)
FitModel
```

!!! tip "About the `:mle` method for Archimedean copulas"
    The `:mle` fitting method is automatically available for all
    `ArchimedeanCopula{d, GT}` types where the generator `GT <: UnivariateGenerator`.
    It performs maximum-likelihood estimation over the parameter bounds
    defined by `_θ_bounds(GT, d)`, using an adaptive LBFGS optimizer within a
    box-constrained (`Fminbox`) setup.


## 3.3 Extreme-Value example — *GumbelEVCopula*

The `GumbelEVCopula` (also known as the *logistic model*) is one of the most common
**Extreme-Value (EV)** copulas. It is defined through its **Pickands dependence function**:

$$A(t) = \bigl(t^{θ} + (1-t)^{θ}\bigr)^{1/θ}, \quad θ \ge 1.$$

### Defining the tail function

All bivariate EV copulas in `Copulas.jl` are defined via a subtype of `Tail`,
which specifies the Pickands function `A(t)` and its parameterization.

```@example generic_copula_example
using LogExpFunctions

struct GumbelTail{T} <: Copulas.OneParameterPickandsTail # subtype of Tail
    θ::T
    function GumbelTail(θ)
        !(1 <= θ) && throw(ArgumentError("θ must be in [1, ∞)"))
        θ == 1 && return NoTail()
        isinf(θ) && return MTail()
        θ, _ = promote(θ, 1.0)
        return new{typeof(θ)}(θ)
    end
end

# Pickands dependence function
function Copulas.A(tail::GumbelTail, t::Real)
    θ = tail.θ
    logB = LogExpFunctions.logaddexp(θ * log(t), θ * log1p(-t))
    return exp(logB / θ)
end

# Parameters and bounds
Distributions.params(tail::GumbelTail) = (; θ = tail.θ,)
Copulas._unbound_params(::Type{<:GumbelTail}, d, θ) = [log(θ.θ - 1)]      # θ ≥ 1
Copulas._rebound_params(::Type{<:GumbelTail}, d, α) = (; θ = exp(α[1]) + 1)
Copulas._θ_bounds(::Type{<:GumbelTail}, d) = (1, Inf)
```

### Building the EV copula

Once the tail is defined, constructing the copula is immediate:

```@example generic_copula_example
const GumbelEVCopula{d,T} = Copulas.ExtremeValueCopula{d, GumbelTail{T}}
C = GumbelEVCopula{2}(2.5)
C_runtime = GumbelEVCopula(2, 2.5)
@assert typeof(C_runtime) == typeof(C)
```

All standard API methods (`cdf`, `pdf`, `rand`, `fit`, etc.) are automatically inherited
from `ExtremeValueCopula`, with internal numerical integration based on the Pickands function.

```@example generic_copula_example
u = [0.4, 0.7]
cdf(C, u)
pdf(C, u)
```

### Fitting interface

The fitting API for EV copulas relies on dependence-based estimators (`:itau`, `:irho`, `:igamma`),
since likelihood evaluation involves non-smooth densities.

For the `GumbelEVCopula`, we define the available methods and optional parameter reparameterizations:

```@example generic_copula_example
Copulas._available_fitting_methods(::Type{GumbelEVCopula}, d) = (:iupper, :mle)
Copulas._example(::Type{GumbelEVCopula}, d) = GumbelEVCopula{d}(2.5)
```

#### Closed-form estimator from upper-tail dependence

The Gumbel EV copula has a closed-form expression for the upper-tail coefficient:

$$\lambda_U = 2 - 2^{1/θ}.$$

This can be inverted to obtain a simple plug-in estimator for (θ):

$$\hat{θ} = 1 / \log_2(2 - \hat{\lambda}_U).$$

Hence, the `:iupper` method can be implemented as:

```@example generic_copula_example
function Copulas._fit(::Type{CT}, U, ::Val{:iupper}) where {CT<:GumbelEVCopula}
    d = size(U, 1)
    λ̂ = Copulas.λᵤ(U)                # empirical upper-tail dependence
    θ  = 1 / log2(2 - λ̂)
    θ  = clamp(θ, 1.0, 50.0)
    Ĉ = Copulas._construct_from_params(CT, d, θ)
    return Ĉ, (; θ̂ = (; θ = θ), λ̂ = λ̂, method = :iupper)
end
```

### Example: sampling and fitting

```@example generic_copula_example
Random.seed!(123)
U = rand(GumbelEVCopula{2}(4.5), 300)
M = fit(CopulaModel, GumbelEVCopula, U)
M
```


!!! note "Automatic inheritance"
    For all `ExtremeValueCopula` types, once the `A(t)` function is defined and satisfies the convexity
    and boundary conditions, the generic API automatically provides:
    `cdf`, `pdf`, `rand`, and dependence measures (`τ`, `ρ`, `λ_L`, `λ_U`).

!!! tip "Analytical inversion and custom estimators"
    EV copulas with known analytical relationships between parameters and tail coefficients
    can provide fast and numerically stable estimators (e.g., the `:iupper` method shown above),
    which can complement or replace likelihood-based methods.

!!! tip "Recommended practice"
    EV copulas usually lack smooth closed-form densities.
    Analytical forms are optional but highly recommended to improve numerical stability.
    Otherwise, `Copulas.jl` will fall back to numerical integration based on the Pickands function.
