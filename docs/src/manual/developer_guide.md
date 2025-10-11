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
This page is intended for package contributors and advanced users who want to extend `Copulas.jl` with new copula families, internal optimizations, or additional features.

---

# 1. The main API

## 1.1 Overview

Every copula type in `Copulas.jl` must implement a minimal set of methods
to integrate correctly with the ecosystem.
Most features will automatically work once these methods are in place.

| Method                           | Purpose                             | Required    |
| -------------------------------- | ----------------------------------- | ----------- |
| `cdf(C, u)`                      | Cumulative distribution function    | ✅           |
| `logpdf(C, u)`                   | Joint log density                   | ✅           |
| `rand(C, n)`                     | Random generation                   | ✅           |
| `params(C)`                      | Return parameters as a `NamedTuple` | ✅           |
| `fit(::Type{CopulaModel}, C, U)` | Model fitting interface             | ⚙️ Optional |
| `τ(C)`, `ρ(C)`, others           | Dependence metrics                  | ⚙️ Optional |
| `λ_L(C)`, `λ_U(C)`               | Tail dependence coefficients        | ⚙️ Optional |
| `condition(C, i, u)`             | Conditional copula                  | ⚙️ Optional |

---

## 1.2 Probability interface (`cdf`, `pdf`, `rand`)

All copulas must define their joint CDF and PDF over the unit hypercube.
If the density is not analytically available, the implementation may raise
an error or rely on a numerical approximation.

The `rand(C, n)` method should generate an `d × n` matrix of samples
from the copula.

Minimal example:

```julia
struct MyCopula{d, P} <: Copula{d}
    θ::P  # Copula parameter
    MyCopula{d}(θ) where {d} = new{d, typeof(θ)}(θ)
end
Distributions.cdf(C::MyCopula{P}, u) where {P} = ...
Distributions._logpdf(C::MyCopula{P}, u) where {P} = ...
Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:MyCopula} = ...
Distributions.params(C::MyCopula) = (θ = C.θ,)
```

Once defined, these automatically integrate with the `Distributions.jl` interface.

---

## 1.3 Dependence metrics

Dependence measures — such as Kendall’s τ, Spearman’s ρ, and others listed in [this section](@ref dep_metrics) — are not mandatory.  
Most can be derived theoretically, empirically, or numerically through the existing API, but providing analytical forms when possible is highly recommended.

| Function           | Description                 | Default behavior            |
| ------------------ | --------------------------- | --------------------------- |
| `τ(C)`             | Kendall’s tau               | Default numerical estimator |
| `ρ(C)`             | Spearman’s rho              | Default numerical estimator |
| `λ_L(C)`, `λ_U(C)` | Lower/upper tail dependence | Default extrapolation-based |

If your copula provides closed-form expressions for any of these, overriding the default
methods will improve both accuracy and performance.

```julia
τ(C::MyCopula) = ...
ρ(C::MyCopula) = ...
```
---

## 1.4 Conditioning and subsetting

To make `condition(C, i, u)` available on your copula type, you only need to define
two bindings:

```julia
DistortionFromCop(::Type{<:MyCopula}) = ...
ConditionalCopula(::Type{<:MyCopula}, args...) = ...
```

These allow `Copulas.jl` to build conditional distributions internally.
If not defined, conditioning will fall back to a generic empirical approximation.

You don’t need to override anything else unless your copula has a closed form
for conditional dependence.

---

## 1.5 Fitting interface

The fitting interface allows your copula to work with `fit(::Type{CopulaModel}, ...)`
and the general estimation framework.

### Required internal methods

| Method                              | Purpose                                                       |
| ----------------------------------- | ------------------------------------------------------------- |
| `_example(CT, d)`                   | Returns a sample instance used for defaults                   |
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

---

# 2. Specific sub-APIs
Some families of copulas in `Copulas.jl` have additional internal structures or specific mathematical representations.
This section summarizes the bindings required for the most common ones: **Archimedean** and **Extreme Value** copulas.

Each sub-API is based on the general interface described above (`cdf`, `logpdf`, `rand`, `fit`, etc.); however, in these cases, the requirements are different.

## 2.1 Archimedean copulas

Archimedean copulas are defined by a generator function ϕ. To implement a new Archimedean family, define a subtype of
[`Generator`](@ref) and implement the following:

```julia
ArchimedeanCopula{d,G}
```

### Required methods for a generator `G`

| Method                    | Purpose                                                            | Required |
| ------------------------- | ------------------------------------------------------------------ | -------- |
| `ϕ(G, t)`                 | Generator function                                                 | ✅        |
| `max_monotony(G)`         | Maximum degree of monotonicity (controls validity in d dimensions) | ✅        |
| `Distributions.params(G)` | Return parameters as a `NamedTuple`                                | ✅        |

Minimal sketch:

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
ϕ(G::MyGenerator, t) = ...
max_monotony(G::MyGenerator) = ...
Distributions.params(G::MyGenerator) = (θ = G.θ,)
```

An Archimedean copula is then built with:

```julia
const MyArchimedeanCopula{d,T} = ArchimedeanCopula{d, MyGenerator{T}}
```

Once the generator defines `ϕ`, and `max_monotony`, all functions such as
`cdf`, `logpdf`, and `rand` become available automatically through
`ArchimedeanCopula`’s generic implementation.
Only fitting routines or dependence metrics need to be added if the defaults
are insufficient.

---

## 2.2 Extreme-Value copulas

Bivariate Extreme-Value (EV) copulas are defined by a Pickands function A. 
To implement a new bivariate Extreme-Value family, define a subtype of [`Tail`](@ref).

All EV copulas in `Copulas.jl` are subtypes of `ExtremeValueCopula{d}`.

### Required methods

| Method      | Purpose                                                      | Required    |
| ----------- | ------------------------------------------------------------ | ----------- |
| `A(T, t)`   | Pickands dependence function                                 | ✅           |
| `Distributions.params(T)` | Return parameters as a `NamedTuple`            | ✅           |

Minimal outline:

```julia
struct MyTail{T} <: Tail
    θ::T
end

A(T::MyTail, t) = ...
Distributions.params(T::MyTail) = (θ = T.θ,)
```
An Extreme-Value copula is then built with:

```julia
const MyEVCopula{d,T} = ExtremeValueCopula{d, MyTail{T}}
```

If these conditions hold, the resulting copula is guaranteed to be valid.
Once `A` and `cdf` are provided, `Copulas.jl` automatically handles random generation,
dependence measures, and integration with the general API.

!!! note "Inherited interfaces in structured families"
For structured copula families such as **Archimedean** and **Extreme-Value**,  
most of the general interface (`cdf`, `logpdf`, `rand`, `fit`, etc.) is already implemented internally in `Copulas.jl`.  

Therefore, these methods are **not mandatory** for each new subtype.  
Defining the corresponding *core component* — the `Generator` (for Archimedean) or the `Tail` (for Extreme-Value) —  
is sufficient to automatically enable the entire probability interface, fitting routines, and dependence measures.

In other words:
- The only **mandatory** definitions are those listed in each sub-API table (`ϕ`, `max_monotony` for Archimedean; `A` for Extreme-Value).  
- All other methods become **optional overrides**, recommended only when analytical or more stable forms are available.

Perfecto — aquí tienes la subsección **2.3 Elliptical copulas**, escrita con el mismo tono, formato y estilo que tus secciones de Archimedean y Extreme-Value, siguiendo la estructura del desarrollador (“qué definir” → “qué es opcional”).

---

## 2.3 Elliptical copulas

Elliptical copulas arise from the dependence structure of multivariate **elliptical distributions** such as the Gaussian or Student-t.
In `Copulas.jl`, every elliptical family is represented as

```julia
EllipticalCopula{d, D}
```

where `D` is the associated multivariate distribution type (for instance, `MvNormal` or `MvTDist`).

Elliptical copulas are characterized by a correlation matrix `Σ` and, optionally, additional shape parameters (e.g. degrees of freedom `ν` for the t-copula).

---

### Required methods

| Method                    | Purpose                                                | Required       |
| ------------------------- | ------------------------------------------------------ | -------------- |
| `U(::Type{CT})`           | Return the univariate marginal distribution            | ✅              |
| `N(::Type{CT})`           | Return the multivariate elliptical distribution        | ✅              |
| `Distributions.params(C)` | Return parameters as a `NamedTuple`                    | ✅              |
| `make_cor!(Σ)`            | Ensure `Σ` is a valid correlation matrix (SPD, diag=1) | ⚙️ Recommended |

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

---

### Optional (recommended) methods

| Method                          | Purpose                                       | Comment                          |
| ------------------------------- | --------------------------------------------- | -------------------------------- |
| `make_cor!(Σ)`                  | Normalize arbitrary covariance to correlation | Helps ensure numerical stability |
| `τ(C)`, `ρ(C)`                  | Closed-form dependence measures               | Defaults provided theoretically  |
| `λ_L(C)`, `λ_U(C)`              | Tail coefficients                             | Optional (computed from family)  |
| `fit(::Type{CopulaModel}, ...)` | Specialized fitting routine                   | Optional override                |

Most elliptical families (Gaussian, t, Laplace, power-exponential, GED) can be implemented simply by changing their `U` and `N` definitions, reusing the same generic machinery.

---

!!! note "Analytical and numerical stability"
Although most elliptical copulas work out-of-the-box through numerical evaluation of multivariate CDFs and densities,
it is **highly recommended** to provide analytical or semi-analytical forms for the following when possible:

* Tail coefficients (`λ_L`, `λ_U`)
* Dependence measures (`τ`, `ρ`)
* Specialized `logpdf` or `rand` implementations (e.g. variance-mixture sampling for Laplace or generalized t families)

Such implementations significantly improve numerical stability and performance of the overall package.

---

# 3. Complete Examples
This section provides practical examples of complete copula implementations.  
Each example illustrates how to make a new family compatible with the main API of `Copulas.jl`.

---

## 3.1 Generic copula example — *MardiaCopula*

The `MardiaCopula` is a simple **bivariate** copula that mixes the Fréchet upper, lower, and independent copulas using a single parameter θ ∈ [−1, 1].  
It serves as a minimal example of how to implement a copula *from scratch* without relying on the `Generator` or `Tail` sub-APIs.

```@example generic_copula_example
using Copulas, Distributions, Random

struct MardiaCopula{P} <: Copulas.Copula{2}
    θ::P
    function MardiaCopula(θ)
        if !(-1 <= θ <= 1)
            throw(ArgumentError("θ must be in [-1,1]"))
        elseif θ == 0
            return IndependentCopula()
        elseif θ == 1
            return MCopula()
        elseif θ == -1
            return WCopula()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
```

The joint CDF follows Mardia’s formulation:

```@example generic_copula_example
function Distributions.cdf(C::MardiaCopula, u)
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

function Distributions._rand!(rng::Distributions.AbstractRNG, C::MardiaCopula, x::AbstractVector{T}) where {T<:Real}
    θ = C.θ
    u1, u2 = rand(rng, Distributions.Uniform(0,1), 2)
    p = [θ^2 * (1 + θ) / 2, 1 - θ^2, θ^2 * (1 - θ) / 2]
    z = rand(rng, Distributions.Categorical(p))
    if z == 1
        u = min(u1, u2)
        x[1] = u; x[2] = u
    elseif z == 2
        x[1] = u1; x[2] = u2
    else
        u = max(u1 + u2 - 1, 0)
        x[1] = u; x[2] = 1 - u
    end
    return x
end
```

### Parameters and usage

```@example generic_copula_example
Distributions.params(C::MardiaCopula) = (; θ = C.θ,)

Random.seed!(123)
C = MardiaCopula(0.8)
U = rand(C, 300)
```

The copula now works seamlessly with all standard methods:

```@example generic_copula_example
cdf(C, [0.3, 0.7])
pdf(C, [0.3, 0.7])
```

### Fitting interface and integration

To make the copula compatible with `Distributions.fit` and the unified `CopulaModel` interface,
we provide a minimal `_fit` definition using a dependence-based measure — in this case, **Gini’s γ**.

```@example generic_copula_example
_unbound_params(::Type{MardiaCopula}, d, params) = [atanh(clamp(params.θ, -1 + eps(), 1 - eps()))]
_rebound_params(::Type{MardiaCopula}, d, α) = (; θ = tanh(α[1]) )
_example(::Type{<:MardiaCopula}, d::Int) = MardiaCopula(0.5)
_available_fitting_methods(::Type{<:MardiaCopula}, d::Int) = (:igamma,)

function _fit(::Type{<:MardiaCopula}, U::AbstractMatrix, ::Val{:igamma})
    γ̂ = Copulas.corgini(U')[1, 2]
    θ  = sign(γ̂) * abs(γ̂)^(1/3)
    θ  = clamp(θ, -1.0, 1.0)
    Ĉ = MardiaCopula(θ)
    return Ĉ, (; θ̂ = (; θ = θ), γ̂ = γ̂, method = :igamma)
end
```

This approach bypasses the need for a log-likelihood function (since the copula lacks a Lebesgue density)
while maintaining compatibility with all higher-level fitting utilities.

---

### Example: fitting and model summary

```@example generic_copula_example
using StatsBase

M = fit(CopulaModel, MardiaCopula, U; method = :igamma, vcov = false)
println(M)
```
---

!!! note "Example purpose"
This example illustrates a fully functional copula defined *from scratch*.
Once these minimal methods are implemented (`cdf`, `pdf`, `rand`, and `params`),
the family automatically integrates with the `Distributions.jl` and `StatsBase` ecosystems.


## 3.2 Archimedean example — *Nelsen2Copula*

The `Nelsen2Copula` is a simple **Archimedean** copula defined by the generator

$$\varphi(t) = (t^{-θ} - 1) / θ, \quad θ > 0.$$

This example demonstrates how to define a new Archimedean copula family using the `Generator` sub-API.
Once the generator is defined, all the usual functions (`cdf`, `pdf`, `rand`, `fit`, etc.)
are automatically inherited from the generic `ArchimedeanCopula` implementation.

### Defining the generator

Every Archimedean copula in `Copulas.jl` is built from a subtype of `Generator` that defines
the core functional behavior of the family.

```@example archimedean_copula_example
using Copulas, Distributions, Random

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

# Generator and its inverse
ϕ(G::Nelsen2Generator, t) = (t^(-G.θ) - 1) / G.θ
ϕ⁻¹(G::Nelsen2Generator, s) = (1 + G.θ * s)^(-1 / G.θ) # This is not mandatory

# Validity and parameters
max_monotony(G::Nelsen2Generator) = Inf
Distributions.params(G::Nelsen2Generator) = (; θ = G.θ,)
```

### Building the Archimedean copula

With the generator defined, we can directly construct the copula via:

```@example archimedean_copula_example
const Nelsen2Copula{d, T} = ArchimedeanCopula{d, Nelsen2Generator{T}}
C = Nelsen2Copula(2, 3.5)
```

The resulting object already supports all standard functionality from the general API:

```@example archimedean_copula_example
u = [0.3, 0.8]
cdf(C, u)
pdf(C, u)
```

### Fitting interface

Because the `ArchimedeanCopula` type already provides estimation routines for Kendall’s τ and Spearman’s ρ,
no explicit `_fit` definition is needed unless you wish to override the defaults.

To verify:

```@example archimedean_copula_example
_unbound_params(::Type{<:Nelsen2Generator}, d, θ) = [log(θ.θ - 1)]
_rebound_params(::Type{<:Nelsen2Generator}, d, α) = (; θ = exp(α[1]) + 1)
_available_fitting_methods(::Type{Nelsen2Copula}, d) = (:igamma, :mle)
_example(::Type{Nelsen2Copula}, d) = Nelsen2Copula(d, 2.5)
_θ_bounds(::Type{<:Nelsen2Generator}, d) = (1, Inf)

function _fit(::Type{CT}, U, method::Union{Val{:igamma}, Val{:mle}}) where {CT<:Nelsen2Copula}
    d = size(U, 1)
    cop(α) = CT(d, _rebound_params(CT, d, α)...)
    α₀ = _unbound_params(CT, d, Distributions.params(_example(CT, d)))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use :igamma since there are too many parameters."
    # Compute Gini’s γ on the data (scalar)
    γ̂ = Copulas.corgini(U')[1, 2]
    # Loss = squared deviation from empirical γ
    loss(C) = (Copulas.corgini(rand(C, size(U,2))')[1,2] - γ̂)^2
    # Optimize parameter
    res  = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    # Return fitted copula and metadata
    return CT(d, θhat...), (; θ̂=θhat,
        optimizer  = Optim.summary(res),
        converged  = Optim.converged(res),
        iterations = Optim.iterations(res))
end
```

### Example: quick γ-based fit

```@example archimedean_copula_example
using Optim
Random.seed!(123)
U = rand(C, 250)
Fit = fit(Nelsen2Copula, U; method=:igamma) # igamma is the default method
Fit
```

For completeness, you can also use the model-based interface using the `:mle` default fit
for one-parameter Archimedean copulas:

```@example archimedean_copula_example
using Optim
FitModel = fit(CopulaModel, Nelsen2Copula, U; method=:mle, start = 1.5)
FitModel
```

!!! tip "About the `:mle` method for Archimedean copulas"
The `:mle` fitting method is automatically available for all
`ArchimedeanCopula{d, GT}` types where the generator `GT <: UnivariateGenerator`.
It performs maximum-likelihood estimation over the parameter bounds
defined by `_θ_bounds(GT, d)`, using an adaptive LBFGS optimizer within a
box-constrained (`Fminbox`) setup.

!!! note "Automatic inheritance"
Once the generator defines `ϕ`, `ϕ⁻¹`, and `max_monotony`,
the following are automatically available:
`cdf`, `pdf`, `rand`, `fit`, and dependence measures (`τ`, `ρ`, `λ_L`, `λ_U`).

---

## 3.3 Extreme-Value example — *GumbelEVCopula*

The `GumbelEVCopula` (also known as the *logistic model*) is one of the most common
**Extreme-Value (EV)** copulas. It is defined through its **Pickands dependence function**:

$$A(t) = \bigl(t^{θ} + (1-t)^{θ}\bigr)^{1/θ}, \quad θ \ge 1.$$

### Defining the tail function

All bivariate EV copulas in `Copulas.jl` are defined via a subtype of `Tail`,
which specifies the Pickands function `A(t)` and its parameterization.

```@example extremevalue_copula_example
using Copulas, Distributions, Random, LogExpFunctions

struct GumbelTail{T} <: Copulas.AbstractUnivariateTail2 # subtype of Tail2
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
function A(tail::GumbelTail, t::Real)
    θ = tail.θ
    logB = LogExpFunctions.logaddexp(θ * log(t), θ * log1p(-t))
    return exp(logB / θ)
end

# Parameters and bounds
Distributions.params(tail::GumbelTail) = (; θ = tail.θ,)
_unbound_params(::Type{<:GumbelTail}, d, θ) = [log(θ.θ - 1)]      # θ ≥ 1
_rebound_params(::Type{<:GumbelTail}, d, α) = (; θ = exp(α[1]) + 1)
_θ_bounds(::Type{<:GumbelTail}, d) = (1, Inf)
```

### Building the EV copula

Once the tail is defined, constructing the copula is immediate:

```@example extremevalue_copula_example
const GumbelEVCopula{T} = ExtremeValueCopula{2, GumbelTail{T}}
C = GumbelEVCopula(2, 2.5)
```

All standard API methods (`cdf`, `pdf`, `rand`, `fit`, etc.) are automatically inherited
from `ExtremeValueCopula`, with internal numerical integration based on the Pickands function.

```@example extremevalue_copula_example
u = [0.4, 0.7]
cdf(C, u)
pdf(C, u)
```

### Fitting interface

The fitting API for EV copulas relies on dependence-based estimators (`:itau`, `:irho`, `:igamma`),
since likelihood evaluation involves non-smooth densities.

For the `GumbelEVCopula`, we define the available methods and optional parameter reparameterizations:

```@example extremevalue_copula_example
_available_fitting_methods(::Type{GumbelEVCopula}, d) = (:iupper, :mle)
_example(::Type{GumbelEVCopula}, d) = GumbelEVCopula(2, 2.5)
```

#### Closed-form estimator from upper-tail dependence

The Gumbel EV copula has a closed-form expression for the upper-tail coefficient:

$$\lambda_U = 2 - 2^{1/θ}.$$

This can be inverted to obtain a simple plug-in estimator for (θ):

$$\hat{θ} = 1 / \log_2(2 - \hat{\lambda}_U).$$

Hence, the `:iupper` method can be implemented as:

```@example extremevalue_copula_example
function _fit(::Type{CT}, U, ::Val{:iupper}) where {CT<:GumbelEVCopula}
    d = size(U, 1)
    λ̂ = Copulas.λᵤ(U)                # empirical upper-tail dependence
    θ  = 1 / log2(2 - λ̂)
    θ  = clamp(θ, 1.0, 50.0)
    Ĉ = CT(d, θ)
    return Ĉ, (; θ̂ = (; θ = θ), λ̂ = λ̂, method = :iupper)
end
```

### Example: sampling and fitting

```@example extremevalue_copula_example
Random.seed!(123)
U = rand(GumbelEVCopula(2, 4.5), 300)
M = fit(CopulaModel, GumbelEVCopula, U)
M
```

---

!!! note "Automatic inheritance"
For all `ExtremeValueCopula` types, once the `A(t)` function is defined and satisfies the convexity
and boundary conditions, the generic API automatically provides:
`cdf`, `pdf`, `rand`, and dependence measures (`τ`, `ρ`, `λ_L`, `λ_U`).

!!! tip "Analytical inversion and custom estimators"
EV copulas with known analytical relationships between parameters and tail coefficients
can provide fast and numerically stable estimators (e.g., the `:iupper` method shown above),
which can complement or replace likelihood-based methods.

```

!!! tip "Recommended practice"
EV copulas usually lack smooth closed-form densities.
Analytical forms are optional but highly recommended to improve numerical stability.
Otherwise, `Copulas.jl` will fall back to numerical integration based on the Pickands function.

---

## 3.4 Other structures

Other possible extensions include:

* Elliptical copulas (Gaussian, Student-t)
* Empirical copulas

Each follows the same general interface:
define `cdf`, `pdf`, and `rand`, and optionally
add fitting or dependence methods.

---

# 4 Integration

Once your copula implements the required methods, it becomes automatically compatible with:

* `fit`, `CopulaModel`
* `StatsBase.vcov`, `StatsBase.confint`
* `Distributions.loglikelihood`
* `StatsBase.aic`, `StatsBase.bic`, `Copulas.aicc`, `Copulas.hqc`

Example:

```@example extremevalue_copula_example
using StatsBase
aic(M)
bic(M)
Copulas.aicc(M)
Copulas.hqc(M)
```

Your new family will now integrate fully within the `Copulas.jl` framework.
