"""
    FRA1Generator(η, θ)
    FRA1Copula{d}(η, θ)
    FRA1Copula(d, η, θ)

The full-range Archimedean copula of type I (FRA1) has parameters
``η, θ ∈ [-1, 1)`` controlling lower- and upper-tail dependence. Its completely
monotone generator is constructed as ``ψ_{η,θ}(t) = G_η(W_θ(t))``; see Hua
(2026) for the definitions and proof of validity in every finite dimension.
The pair ``(η, θ) = (-1, -1)`` gives independence.

See also: [`Generator`](@ref), [`ArchimedeanCopula`](@ref),
[`Distributions.fit`](@ref).

References:
* [hua2026fra1](@cite) Hua, L. (2026). A new tractable Archimedean copula for
  full-range tail dependence. arXiv:2609.18742.
"""
FRA1Generator, FRA1Copula

Paramorph.@paramorph T struct FRA1Generator{T<:Real} <: Generator
    η::T ~ bounded_interval(-one(T), one(T); right_closed=false)
    θ::T ~ bounded_interval(-one(T), one(T); right_closed=false)
end

function FRA1Generator(η::Real, θ::Real)
    T = promote_type(typeof(float(η)), typeof(float(θ)))
    return FRA1Generator{T}(T(η), T(θ))
end

const FRA1Copula{d, T} = ArchimedeanCopula{d, FRA1Generator{T}}
function (::Type{FRA1Copula{d}})(args...; kwargs...) where {d}
    return _wrap_archimedean(Val(d), FRA1Generator(args...; kwargs...))
end
(::Type{FRA1Copula})(d::Int, args...; kwargs...) = FRA1Copula{d}(args...; kwargs...)

max_monotony(::FRA1Generator) = Inf
@inline limit_kind(G::FRA1Generator, ::Val) =
    G.η == -1 && G.θ == -1 ? Π_LIMIT : NO_LIMIT

function _fra1_F(p, s)
    isone(p) && return s
    angle = 2 * asinh(sqrt(s / 2))
    iszero(p) && return angle^2 / 2
    return 2 * sinh(p * angle / 2)^2 / p^2
end

function _fra1_F_inverse(p, y)
    isone(p) && return y
    iszero(p) && return 2 * sinh(sqrt(y / 2))^2
    angle = 2 * asinh(sqrt(p^2 * y / 2)) / p
    return 2 * sinh(angle / 2)^2
end

function _fra1_G(η, s)
    if η <= 0
        return exp(-_fra1_F(-η, s))
    end

    b = (1 - η) / η^2
    value = _fra1_F(η, s)
    iszero(value) && return one(value)
    isinf(b) && return exp(-value)
    return exp(-b * log1p(value / b))
end

function _fra1_G_inverse(η, u)
    if η <= 0
        return _fra1_F_inverse(-η, -log(u))
    end

    b = (1 - η) / η^2
    value = -log(u)
    isinf(b) && return _fra1_F_inverse(η, value)
    return _fra1_F_inverse(η, b * expm1(value / b))
end

function _fra1_J(p, t)
    isone(p) && return t
    iszero(t) && return zero(t)
    x = inv(t)
    shifted = _fra1_F_inverse(p, _fra1_F(p, x) + one(x))
    offset = _fra1_F_inverse(p, one(x))
    return inv(shifted - offset)
end

function _fra1_J_inverse(p, y)
    isone(p) && return y
    iszero(y) && return zero(y)
    isinf(y) && return oftype(float(y), Inf)
    offset = _fra1_F_inverse(p, one(y))
    shifted = _fra1_F(p, offset + inv(y)) - one(y)
    return inv(_fra1_F_inverse(p, max(zero(shifted), shifted)))
end

function _fra1_K(a, t)
    iszero(t) && return zero(t)
    return expm1(log1p(exp(a * log(t))) / a)
end

function _fra1_K_inverse(a, x)
    iszero(x) && return zero(x)
    return expm1(a * log1p(x))^(inv(a))
end

function _fra1_W(θ, t)
    θ <= 0 && return _fra1_J(-θ, t)
    return _fra1_J(zero(θ), _fra1_K(1 - θ, t))
end

function _fra1_W_inverse(θ, y)
    θ <= 0 && return _fra1_J_inverse(-θ, y)
    a = 1 - θ
    return _fra1_K_inverse(a, _fra1_J_inverse(zero(θ), y))
end

function ϕ(G::FRA1Generator, t)
    return _fra1_G(G.η, _fra1_W(G.θ, t))
end

function ϕ⁻¹(G::FRA1Generator, u)
    u <= 0 && return oftype(float(u), Inf)
    u >= 1 && return zero(float(u))
    return _fra1_W_inverse(G.θ, _fra1_G_inverse(G.η, u))
end