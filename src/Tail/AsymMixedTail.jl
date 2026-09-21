"""
  AsymMixedCopula{2}(θ₁, θ₂)
  AsymMixedCopula(2, θ₁, θ₂)
  AsymMixedTail(θ₁, θ₂)

The (bivariate) asymmetric Mixed extreme-value copula is parameterized by two parameters ``\\theta_1``, ``\\theta_2`` subject to the following constraints:

* θ₁ ≥ 0
* θ₁ + θ₂ ≤ 1
* θ₁ + 2θ₂ ≤ 1
* θ₁ + 3θ₂ ≥ 0

Its Pickands dependence function is

```math
A(t) = \\theta_{2}t^3 + \\theta_{1}t^2 - (\\theta_1+\\theta_2)t + 1,\\quad t\\in[0,1].
```

Special cases:

* θ₁ = θ₂ = 0 ⇒ IndependentCopula
* θ₂ = 0      ⇒ symmetric Mixed copula

This polynomial Pickands model is bivariate. The admissible set is the
intersection of all four inequalities above, not a rectangular parameter box;
fitting or proposing parameters independently can therefore leave the valid
region.

See also: [`Tail`](@ref), [`ExtremeValueCopula`](@ref), [`A`](@ref),
[`MixedTail`](@ref), [`Distributions.fit`](@ref).

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
AsymMixedTail, AsymMixedCopula

Paramorph.@paramorph T struct AsymMixedTail{T<:Real} <: BivariatePickandsTail
    θ₁::T
    θ₂::T
    @geometry ((θ₁, θ₂) ~ Paramorph.asymmetric_mixed())
end
function AsymMixedTail(θ₁::Real, θ₂::Real)
    T = promote_type(typeof(float(θ₁)), typeof(float(θ₂)))
    return AsymMixedTail{T}(T(θ₁), T(θ₂))
end


@inline limit_kind(tail::AsymMixedTail, ::Val{2}) =
    iszero(tail.θ₁) && iszero(tail.θ₂) ? Π_LIMIT : NO_LIMIT

const AsymMixedCopula{d,T} = ExtremeValueCopula{d, AsymMixedTail{T}}
function (::Type{AsymMixedCopula{d}})(args...; kwargs...) where {d}
    return _wrap_extreme_value(Val(d), AsymMixedTail(args...; kwargs...))
end
(::Type{AsymMixedCopula})(d::Int, args...; kwargs...) = _wrap_extreme_value(Val(d), AsymMixedTail(args...; kwargs...))

Distributions.params(C::ExtremeValueCopula{D,<:AsymMixedTail}) where {D} =
    (C.tail.θ₁, C.tail.θ₂)

function _fit(
    CT::Type{<:ExtremeValueCopula{D,<:AsymMixedTail} where D},
    U,
    ::Val{:mle},
)
    d = size(U, 1)
    d == 2 || throw(DimensionMismatch("AsymMixedCopula is only defined in dimension two"))

    cop(α) = ExtremeValueCopula{2}(Paramorph.constraint(AsymMixedTail{eltype(α)}, α))

    # Start away from the symmetric θ₂=0 line.
    α₀ = [0.0, 0.5]
    loss(α) = -Distributions.loglikelihood(cop(α), U)
    res = Optim.optimize(loss, α₀, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())
    return cop(Optim.minimizer(res))
end

function A(tail::AsymMixedTail, t::Real)
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    tt = _safett(t)
    return θ₂*tt^3 + θ₁*tt^2 - (θ₁+θ₂)*tt + 1
end

function dA(tail::AsymMixedTail, t::Real)
    tt = _safett(t)
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    return 3θ₂ * tt^2 + 2θ₁ * tt - (θ₁ + θ₂)
end

function d²A(tail::AsymMixedTail, t::Real)
    tt = _safett(t)
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    return 6θ₂ * tt + 2θ₁
end
