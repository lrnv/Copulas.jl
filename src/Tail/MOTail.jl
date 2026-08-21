"""
    MOTail{T}, MOCopula{d,T}

    MOCopula(2, λ₁, λ₂, λ₁₂)
    MOCopula(λ::AbstractVector)
    MOCopula(d, λ::AbstractVector)

Marshall-Olkin extreme-value family.

The specialized bivariate representation uses private-shock intensities
`λ₁, λ₂ ≥ 0` and common-shock intensity `λ₁₂ ≥ 0`.

The multivariate representation assigns one nonnegative shock intensity `λ_S`
to every nonempty subset `S ⊆ {1,…,d}`. Therefore `λ` has length `2^d-1`,
ordered by subset cardinality and then lexicographically. If only `λ` is
supplied, the dimension is inferred from its length.

With

```math
r_i=\\sum_{S\\ni i}\\lambda_S,
```

the stable tail dependence function is

```math
\\ell(x)=\\sum_{\\varnothing\\ne S}\\max_{i\\in S}
\\left(\\frac{\\lambda_S}{r_i}x_i\\right).
```

Every margin must have positive total shock rate. Multiplying all shock
intensities by the same positive constant leaves the copula unchanged.

References:

* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications. World Scientific.
"""
MOTail, MOCopula

struct MOTail{T} <: Tail2
    λ₁::T
    λ₂::T
    λ₁₂::T
    function MOTail(λ₁, λ₂, λ₁₂)
        (λ₁ ≥ 0 && λ₂ ≥ 0 && λ₁₂ ≥ 0) || throw(ArgumentError("All λ must be ≥ 0"))
        T = promote_type(typeof(λ₁), typeof(λ₂), typeof(λ₁₂))
        return new{T}(T(λ₁), T(λ₂), T(λ₁₂))
    end
end

const MOCopula{d,T} = ExtremeValueCopula{d, MOTail{T}}
Distributions.params(tail::MOTail) = (λ₁ = tail.λ₁, λ₂ = tail.λ₂, λ₃ = tail.λ₁₂)
_unbound_params(::Type{<:MOTail}, d, θ) = [log(θ.λ₁), log(θ.λ₂), log(θ.λ₃)]
_rebound_params(::Type{<:MOTail}, d, α) = (; λ₁ = exp(α[1]), λ₂ = exp(α[2]), λ₃ = exp(α[3]))

function A(tail::MOTail{T}, t::Real) where T
    tt = _safett(t)
    zz = zero(promote_type(T, typeof(tt)))
    λ₁, λ₂, λ₁₂ = tail.λ₁, tail.λ₂, tail.λ₁₂
    om = 1 - tt
    d1 = λ₁ + λ₁₂
    d2 = λ₂ + λ₁₂
    # Use inv where possible; if a denominator is zero (degenerate), treat the corresponding ratio as zero
    r1 = d1 > 0 ? om * (λ₁ / d1) : zz
    r2 = d2 > 0 ? tt * (λ₂ / d2) : zz
    m1 = d1 > 0 ? (om / d1) : zz
    m2 = d2 > 0 ? (tt / d2) : zz
    term3 = λ₁₂ * max(m1, m2)
    return r1 + r2 + term3
end
function _mo_exponents(tail::MOTail, ::Type{R}) where R
    λ₁, λ₂, λ₁₂ = R(tail.λ₁), R(tail.λ₂), R(tail.λ₁₂)
    d1, d2 = λ₁ + λ₁₂, λ₂ + λ₁₂
    a = iszero(d2) ? zero(R) : λ₂ / d2  # exponent of u
    b = iszero(d1) ? zero(R) : λ₁ / d1  # exponent of v
    return a, b
end
function _pickands_left_slope(tail::MOTail, x::Real)
    R = promote_type(typeof(x), typeof(tail.λ₂), typeof(tail.λ₁₂))
    a, _ = _mo_exponents(tail, R)
    return a - one(R)
end
function _pickands_right_slope(tail::MOTail, x::Real)
    R = promote_type(typeof(x), typeof(tail.λ₁), typeof(tail.λ₁₂))
    _, b = _mo_exponents(tail, R)
    return one(R) - b
end
function τ(C::ExtremeValueCopula{2,<:MOTail})
    a = C.tail.λ₁/(C.tail.λ₁+C.tail.λ₁₂)
    b = C.tail.λ₂/(C.tail.λ₂+C.tail.λ₁₂)
    return a*b/(a+b-a*b)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:MOTail}, A::AbstractMatrix{S}) where {S<:Real}
    size(A, 1) == 2 || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    λ₁, λ₂, λ₁₂ = C.tail.λ₁, C.tail.λ₂, C.tail.λ₁₂
    T = promote_type(typeof(float(λ₁)), typeof(float(λ₂)), typeof(float(λ₁₂)))
    λ₁T, λ₂T, λ₁₂T = T(λ₁), T(λ₂), T(λ₁₂)
    rate_u, rate_v = λ₂T + λ₁₂T, λ₁T + λ₁₂T
    (rate_u > 0 && rate_v > 0) ||
        throw(ArgumentError("Each Marshall-Olkin margin must have a positive total rate"))
    waiting_time(rate) = iszero(rate) ? T(Inf) : T(Random.randexp(rng)) / rate

    # The first Pickands coordinate used by A is -log(u), so its private
    # shock has rate λ₂; the second coordinate analogously uses rate λ₁.
    @inbounds for col in axes(A, 2)
        private_u = waiting_time(λ₂T)
        private_v = waiting_time(λ₁T)
        common = waiting_time(λ₁₂T)
        A[1, col] = exp(-rate_u * min(private_u, common))
        A[2, col] = exp(-rate_v * min(private_v, common))
    end
    return A
end

function Distributions.logcdf(D::BivEVDistortion{MOTail{T}, S}, z::Real) where {T, S}
    R = promote_type(T, S, typeof(float(z)))
    a, b = _mo_exponents(D.tail, R)

    z ≤ 0 && return R(-Inf)
    z ≥ 1 && return zero(R)
    D.uⱼ ≤ 0 && return _biv_ev_endpoint_logcdf(D, z, true, R)
    D.uⱼ ≥ 1 && return _biv_ev_endpoint_logcdf(D, z, false, R)

    u, v = D.j == 2 ? (R(z), R(D.uⱼ)) : (R(D.uⱼ), R(z))
    lu, lv = log(u), log(v)
    s1, s2 = a*lu + lv, lu + b*lv

    if D.j == 2
        # Equality is the post-jump side as the free variable u increases.
        logC, factor = _ev_le(s1, s2) ? (s1, one(R)) : (s2, b)
        return iszero(factor) ? R(-Inf) : logC - lv + log(factor)
    else
        # Equality is the post-jump side as the free variable v increases.
        logC, factor = _ev_lt(s1, s2) ? (s1, a) : (s2, one(R))
        return iszero(factor) ? R(-Inf) : logC - lu + log(factor)
    end
end
function Distributions.quantile(D::BivEVDistortion{MOTail{T}, S}, α::Real) where {T, S}
    R = promote_type(T, S, typeof(float(α)))
    p = R(α)
    zero(R) ≤ p ≤ one(R) || throw(ArgumentError("α must be in [0,1]"))
    p == zero(R) && return zero(R)

    a, b = _mo_exponents(D.tail, R)
    t = R(D.uⱼ)
    t ≤ zero(R) && return _biv_ev_endpoint_quantile(D, p, true, R)
    t ≥ one(R) && return _biv_ev_endpoint_quantile(D, p, false, R)

    # Degenerate parameter cases are safest through the generalized inverse.
    if !(zero(R) < a < one(R) && zero(R) < b < one(R))
        return _unit_quantile(D, p)
    end

    if D.j == 2
        logt = log(t)
        star = exp(((one(R) - b) / (one(R) - a)) * logt)
        α2 = exp(a * log(star))
        α1 = b * α2
        p < α1 && return (p / b) * exp((one(R) - b) * logt)
        p <= α2 && return star
        return exp(log(p) / a)
    else
        logt = log(t)
        star = exp(((one(R) - a) / (one(R) - b)) * logt)
        α2 = exp(b * log(star))
        α1 = a * α2
        p < α1 && return (p / a) * exp((one(R) - a) * logt)
        p <= α2 && return star
        return exp(log(p) / b)
    end
end


"""
    MOMultivariateTail(d, λ)

General `d`-variate Marshall-Olkin extreme-value tail. `λ` contains one
nonnegative shock intensity for every nonempty subset of `1:d`, ordered first
by subset cardinality and then lexicographically.

For margin `i`, let `rᵢ = sum(λ[S] for S containing i)`. The STDF is

    ℓ(x) = sum_S max_{i in S} (λ[S] / rᵢ) xᵢ.

The historical bivariate `MOTail(λ₁, λ₂, λ₁₂)` uses crossed names for the two
private shocks. `MOMultivariateTail(tail::MOTail)` preserves that convention
exactly.
"""
struct MOMultivariateTail{T} <: Tail
    d::Int
    λ::Vector{T}
    spectral::DiscreteSpectralTail{T}
end

function MOMultivariateTail(d::Int, λ::AbstractVector)
    d >= 2 || throw(ArgumentError("Marshall-Olkin dimension must be at least two",))

    subsets = _spectral_subsets(d)
    length(λ) == length(subsets) || throw(DimensionMismatch("expected $(length(subsets)) shock intensities for dimension $d",))

    vals = collect(λ)
    T = promote_type(Float64, map(typeof, vals)...)
    rates = T.(λ)

    all(isfinite, rates) || throw(ArgumentError("all Marshall-Olkin shock intensities must be finite",))
    all(v -> v >= zero(T), rates) || throw(ArgumentError("all Marshall-Olkin shock intensities must be nonnegative",))

    r = zeros(T, d)
    @inbounds for (k, S) in enumerate(subsets)
        rate = rates[k]
        for i in S
            r[i] += rate
        end
    end

    all(v -> v > zero(T), r) || throw(ArgumentError("every Marshall-Olkin margin must have positive total shock rate",))

    B = zeros(T, d, length(subsets))
    @inbounds for (k, S) in enumerate(subsets)
        rate = rates[k]
        for i in S
            B[i, k] = rate / r[i]
        end
    end

    spectral = DiscreteSpectralTail(B)
    return MOMultivariateTail{T}(d, rates, spectral)
end

MOMultivariateTail(tail::MOTail) = MOMultivariateTail(2, [tail.λ₂, tail.λ₁, tail.λ₁₂],)

MOMultivariateCopula(d::Int, λ::AbstractVector) = ExtremeValueCopula(d, MOMultivariateTail(d, λ))

function _mo_dimension(λ::AbstractVector)
    m = length(λ) + 1
    ispow2(m) || throw(DimensionMismatch("Marshall-Olkin λ must have length 2^d-1",))
    d = trailing_zeros(m)
    d >= 2 || throw(ArgumentError("Marshall-Olkin dimension must be at least 2"))
    return d
end

(::Type{<:ExtremeValueCopula{D,<:MOTail} where D})(λ::AbstractVector) = MOMultivariateCopula(_mo_dimension(λ), λ)

(::Type{<:ExtremeValueCopula{D,<:MOTail} where D})(d::Int, λ::AbstractVector,) = MOMultivariateCopula(d, λ)

MOMultivariateCopula(tail::MOTail) = ExtremeValueCopula(2, MOMultivariateTail(tail))

Distributions.params(tail::MOMultivariateTail) = (λ = tail.λ,)
_is_valid_in_dim(tail::MOMultivariateTail, d::Int) = tail.d == d
ℓ(tail::MOMultivariateTail, x) = ℓ(tail.spectral, x)

_rand_ev_multivariate!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:MOMultivariateTail}, X::AbstractMatrix{T},) where {d,T<:Real} = _discrete_spectral_rand!(rng, C.tail.spectral, X)

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:MOMultivariateTail}, X::AbstractMatrix{T},) where {T<:Real}
    size(X, 1) == 2 || throw(DimensionMismatch("output must have two rows for a bivariate Marshall-Olkin copula",))
    return _discrete_spectral_rand!(rng, C.tail.spectral, X)
end
