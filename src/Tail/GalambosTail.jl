"""
    GalambosTail{T}, GalambosCopula{d,T}

    GalambosCopula{d}(θ)
    GalambosCopula(d, θ)

Galambos (negative-logistic) extreme-value copula in dimension `d ≥ 2`, with
`θ ∈ [0, ∞]`. Its stable tail dependence function is

```math
\\ell(x)
=
\\sum_{\\varnothing \\ne I \\subseteq \\{1,\\ldots,d\\}}
(-1)^{|I|+1}
\\left(\\sum_{i\\in I}x_i^{-\\theta}\\right)^{-1/\\theta}.
```

For `d = 2`, the equivalent Pickands dependence function is

```math
A(t)=1-\\left(t^{-\\theta}+(1-t)^{-\\theta}\\right)^{-1/\\theta},
```

and the implementation uses the native bivariate derivatives when beneficial.

Special cases:

* `θ = 0` returns `IndependentCopula(d)`.
* `θ = ∞` returns `MCopula(d)`.

References:

* [galambos1975order](@cite) Galambos, J. (1975). Order statistics of samples from multivariate distributions. Journal of the American Statistical Association, 70(351a), 674-680.
"""
GalambosTail, GalambosCopula

struct GalambosTail{T} <: OneParameterPickandsTail
    θ::T
    function GalambosTail(θ)
        θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
        new{typeof(float(θ))}(float(θ))
    end
end
_reduced_tail(tail::GalambosTail) = iszero(tail.θ) ? NoTail() : isinf(tail.θ) ? MTail() : nothing

const GalambosCopula{d,T} = ExtremeValueCopula{d, GalambosTail{T}}
_is_valid_in_dim(::GalambosTail, d::Int) = d >= 2
Distributions.params(tail::GalambosTail) = (θ = tail.θ,)
_unbound_params(::Type{<:GalambosTail}, d, θ) = [log(θ.θ)]           # θ > 0
_rebound_params(::Type{<:GalambosTail}, d, α) = (; θ = exp(α[1]))
_θ_bounds(::Type{<:GalambosTail}, d) = (0.0, Inf)

function ℓ(tail::GalambosTail, x)
    any(isinf, x) && return maximum(x)
    θ = tail.θ
    out = sum(x)
    d = length(x)
    for k in 2:d, I in Combinatorics.combinations(1:d, k)
        any(i -> iszero(x[i]), I) && continue
        m = minimum(x[i] for i in I)
        s = sum((x[i] / m)^(-θ) for i in I)
        out += (isodd(k) ? one(out) : -one(out)) * m * s^(-inv(θ))
    end
    return out
end

function _ellpartial_signlog(tail::GalambosTail, x, I::Tuple{Vararg{Int}})
    function evaluate(current_tail, current_x)
        θ = current_tail.θ
        k = length(I)
        expected = isodd(k) ? 1 : -1
        base = float(current_x[first(I)] + θ)
        logpos = logneg = oftype(base, -Inf)
        rest = [j for j in eachindex(current_x) if j ∉ I]

        for r in 0:length(rest), J in Combinatorics.combinations(rest, r)
            any(j -> iszero(current_x[j]), J) && continue
            S = (I..., J...)
            m = minimum(current_x[j] for j in S)
            s = sum((current_x[j] / m)^(-θ) for j in S)
            logterm = (one(θ) - k) * log(m) - (inv(θ) + k) * log(s)
            logterm += (-θ - one(θ)) * sum(log(current_x[i] / m) for i in I)
            k > 1 && (logterm += sum(log1p(q * θ) for q in 1:k-1))

            if isodd(length(S))
                logpos = LogExpFunctions.logaddexp(logpos, logterm)
            else
                logneg = LogExpFunctions.logaddexp(logneg, logterm)
            end
        end

        dominant, other = expected == 1 ? (logpos, logneg) : (logneg, logpos)
        isfinite(dominant) || return expected, dominant, false
        !isfinite(other) && return expected, dominant, true
        dominant > other || return expected, dominant, false

        reldiff = -expm1(-(dominant - other))
        tol = base isa AbstractFloat ? sqrt(eps(base)) : zero(base)
        reldiff > tol || return expected, dominant, false
        return expected, dominant + log(reldiff), true
    end

    sgn, logabs, resolved = evaluate(tail, x)
    resolved && return sgn, logabs
    all(xi -> xi isa AbstractFloat, x) || return sgn, logabs
    tail.θ isa AbstractFloat || return sgn, logabs

    # Inclusion-exclusion can lose hundreds of digits for strong dependence.
    # Retry only unresolved partials at increasing precision.
    T = typeof(float(x[first(I)] + tail.θ))
    bits = max(256,
               x[first(I)] isa BigFloat ? precision(x[first(I)]) : 0,
               tail.θ isa BigFloat ? precision(tail.θ) : 0)
    for _ in 1:7
        sgn, logabs, resolved = setprecision(BigFloat, bits) do
            evaluate(GalambosTail(BigFloat(tail.θ)), BigFloat.(x))
        end
        resolved && return sgn, convert(T, logabs)
        bits *= 2
    end
    throw(ArgumentError("Galambos mixed partial could not be resolved numerically"))
end


# Galambos uses its exact spectral sampler in every dimension, including d=2
# where it is substantially faster than the generic Ghoudi/Pickands sampler.
# The common scale of the Weibull/Gamma construction cancels after
# normalization to the simplex.
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:GalambosTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    S = promote_type(T, typeof(C.tail.θ))
    θ = S(C.tail.θ)
    invθ = inv(θ)
    shape = one(S) + invθ
    weibull = Distributions.Weibull(θ, one(S))
    gamma = Distributions.Gamma(shape, one(S))
    q = Vector{S}(undef, d)
    z = Vector{S}(undef, d)
    invd = inv(S(d))

    for col in axes(X, 2)
        fill!(z, zero(S))
        arrival = S(Random.randexp(rng)) * invd
        radius = inv(arrival)

        while radius > minimum(z)
            j = rand(rng, 1:d)
            @inbounds for i in 1:d
                q[i] = rand(rng, weibull)
            end
            q[j] = rand(rng, gamma)^invθ

            qsum = sum(q)
            @inbounds for i in 1:d
                qi = q[i] / qsum
                z[i] = max(z[i], radius * qi)
            end

            arrival += S(Random.randexp(rng)) * invd
            radius = inv(arrival)
        end

        @inbounds for i in 1:d
            X[i, col] = exp(-inv(z[i]))
        end
    end
    return X
end

needs_binary_search(tail::GalambosTail) = (tail.θ > 19.5)
function A(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ  = tail.θ
    if θ == 0
        return 1.0
    elseif isinf(θ)
        return max(tt, 1-tt)
    else
        return -LogExpFunctions.expm1(-LogExpFunctions.logaddexp(-θ*log(tt), -θ*log(1-tt)) / θ)
    end
end
function d²A(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    if θ == 0
        return 0.0
    elseif isinf(θ)
        return 0.0
    end
    a = tt
    b = 1 - tt
    L1 = -θ*log(a)
    L2 = -θ*log(b)
    M  = max(L1, L2)
    E1 = exp(L1 - M)
    E2 = exp(L2 - M)
    S  = E1 + E2
    # B = (a^-θ + b^-θ)^(-1/θ) with numerically stable rescaling
    B  = exp(-(M/θ)) * S^(-1/θ)

    inva = inv(a); invb = inv(b)
    D    = E2*invb - E1*inva
    term1 = (E2*invb^2 + E1*inva^2) / S
    term2 = (D/S)^2
    return (1 + θ) * B * (term1 - term2)
end
function dA(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    if θ == 0 || isinf(θ)
        return 0.0
    end
    a = tt
    b = 1 - tt
    L1 = -θ*log(a)
    L2 = -θ*log(b)
    M  = max(L1, L2)
    E1 = exp(L1 - M)
    E2 = exp(L2 - M)
    S  = E1 + E2
    B  = exp(-(M/θ)) * S^(-1/θ)
    inva = inv(a); invb = inv(b)
    D    = E2*invb - E1*inva
    # A'(t) = B * (D/S)
    return B * (D / S)
end

_tau_galambos(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(GalambosTail(θ),t)*t*(1-t)/max(A(GalambosTail(θ),t),_δ(t)), 0, 1; kw...)[1]
_rho_galambos(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(GalambosTail(θ),t))^2, 0, 1; kw...)[1] - 3

τ(C::ExtremeValueCopula{2,<:GalambosTail}) = _tau_galambos(C.tail.θ)
ρ(C::ExtremeValueCopula{2,<:GalambosTail}) = _rho_galambos(C.tail.θ)
β(C::ExtremeValueCopula{2,<:GalambosTail}) = 2.0^( 2.0^(-1.0/C.tail.θ) ) - 1.0
λᵤ(C::ExtremeValueCopula{2,<:GalambosTail}) = 2.0^(-1.0/C.tail.θ)

τ⁻¹(::Type{<:ExtremeValueCopula{D,<:GalambosTail} where D}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? Inf : _invmono(θ -> _tau_galambos(θ) - τ; kw...)
τ⁻¹(::Type{<:GalambosTail}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? Inf : _invmono(θ -> _tau_galambos(θ) - τ; kw...)
ρ⁻¹(::Type{<:ExtremeValueCopula{D,<:GalambosTail} where D}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? Inf : _invmono(θ -> _rho_galambos(θ) - ρ; kw...)
β⁻¹(::Type{<:ExtremeValueCopula{D,<:GalambosTail} where D}, beta) =
    beta <= 0 ? 0.0 : beta >= 1 ? Inf : -inv(log2(log2(beta + 1)))
λᵤ⁻¹(::Type{<:ExtremeValueCopula{D,<:GalambosTail} where D}, λ) =
    λ <= 0 ? 0.0 : λ >= 1 ? Inf : -inv(log2(λ))
