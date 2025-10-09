```@meta
CurrentModule = Copulas
```

# [Developer Guide: Implementing Fitting for New Families](@id developer_fitting)

This page documents the internal API and minimal requirements for adding new copula or Sklar families compatible with the `fit` interface in `Copulas.jl`.

---

## 1. Overview

The `fit` interface is generic across all copula types.
To make a new family compatible with `fit(::Type{CopulaModel}, ...)`, implement the following methods for your copula type `CT`.

!!! info "Target audience"
This page is intended for package contributors or advanced users wishing to extend `Copulas.jl` with additional families.

---

## 2. Required methods

| Function                            | Purpose                                                                           |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| `_example(CT, d)`                   | Returns a representative instance used to extract defaults.                       |
| `_unbound_params(CT, d, params)`    | Maps parameter `NamedTuple` → unconstrained `Vector{Float64}` for optimization.   |
| `_rebound_params(CT, d, α)`         | Inverse map from optimizer vector to parameter `NamedTuple`.                      |
| `_available_fitting_methods(CT, d)` | Declares supported fitting methods (e.g., `:mle`, `:itau`, `:irho`, `:ibeta`, …). |
| `_fit(CT, U, ::Val{:method})`       | Performs the actual fit; returns `(copula::CT, meta::NamedTuple)`.                |

### Minimal skeleton

```julia
_example(::Type{MyCopula}, d) = MyCopula(d, default_parameters...)
_unbound_params(::Type{MyCopula}, d, params) = [log(params.θ)]
_rebound_params(::Type{MyCopula}, d, α) = (; θ = exp(α[1]))
_available_fitting_methods(::Type{MyCopula}, d) = (:mle, :itau)
function _fit(::Type{MyCopula}, U, ::Val{:mle})
    θ̂ = optimize_mle(U) # your fitting routine
    return MyCopula(size(U, 2), θ̂), (; θ̂, optimizer=:lbfgs, converged=true)
end
```

!!! tip "Naming convention"
Each specialized fitting method is dispatched on `Val{:method}` for clarity and performance.
Example: `_fit(::Type{MyCopula}, U, ::Val{:itau})`.

---

## 3. Full example: implementing a new Archimedean copula

Below we show a complete example with a small, self-contained family `Nelsen2Copula`, including generator, parameter transformations, and fitting methods.

```@example developer_fitting
using Copulas, StatsBase, Random, Distributions, Optim

"""
    Nelsen2Copula{T}

Archimedean copula with generator φ(t) = 1 - t^(1/θ), θ ∈ [1, ∞).
"""
struct Nelsen2Generator{T} <: AbstractUnivariateGenerator
    θ::T
    function Nelsen2Generator(θ)
        if θ < 1
            throw(ArgumentError("θ must be ≥ 1"))
        elseif θ == 1
            return WGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const Nelsen2Copula{d,T} = ArchimedeanCopula{d, Nelsen2Generator{T}}
# Generator core
ϕ(G::Nelsen2Generator, t) = (1 + t^(1/G.θ))^(-1)
ϕ⁻¹(G::Nelsen2Generator, u) = (u^(-1) - 1)^(G.θ)
max_monotony(G::Nelsen2Generator) = Inf
# Interfaces (generator-level)
# --- Interfaces ---
Distributions.params(G::Nelsen2Generator) = (θ = G.θ,)
_unbound_params(::Type{<:Nelsen2Generator}, d, θ) = [log(θ.θ - 1)]
_rebound_params(::Type{<:Nelsen2Generator}, d, α) = (; θ = exp(α[1]) + 1)
_available_fitting_methods(::Type{Nelsen2Copula}, d) = (:igamma)
_example(::Type{Nelsen2Copula}, d) = Nelsen2Copula(d, 2.5)
function _fit(::Type{CT}, U, method::Union{Val{:igamma}, Val{:mle}}) where {CT<:Nelsen2Copula}
    d = size(U, 1)
    cop(α) = CT(d, _rebound_params(CT, d, α)...)
    α₀ = _unbound_params(CT, d, Distributions.params(_example(CT, d)))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use :igamma since there are too many parameters."
    # Compute Gini’s γ on the data
    γ̂ = Copulas.corgini(U')
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

Fit it directly (quick fit):

```@example developer_fitting
Random.seed!(123)
U = rand(Nelsen2Copula(2, 3.5), 250)
F = fit(Nelsen2Copula, U)
F
```
Full model (with covariance and metadata)

```@example developer_fitting
Model = fit(CopulaModel, Nelsen2Copula, U; vcov_method=:bootstrap)
Model
```

---

## 4. Integration notes

* **Automatic compatibility.**
  Once the above methods are implemented, your family is automatically compatible with:

  * `fit`, `CopulaModel`, `StatsBase.vcov`, `StatsBase.confint`
  * `Distributions.loglikelihood`, `StatsBase.aic`, `StatsBase.bic`, etc.

```@example developer_fitting
# Information criteria for the fitted model
aic(Model)
bic(Model)
Copulas.aicc(Model)
Copulas.hqc(Model)
```

* **Covariance estimation (`vcov`).**
  By default, the `vcov` machinery handles parametric families automatically (via Hessian or Godambe estimators).
  For nonparametric models, set `vcov=false` or provide a simplified jackknife estimator.

* **Archimedean generators.**
  Any generator subtype implementing `ϕ(G, t)` and `max_monotony(G)` is automatically usable through `ArchimedeanCopula{d,G}`.
