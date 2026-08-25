"""
    BC2Tail{T}, BC2Copula{d,T}

    BC2Copula{2}(a, b)
    BC2Copula(2, a, b)
    BC2Copula{d}(a::AbstractVector)
    BC2Copula(d, a::AbstractVector)
    BC2Copula(a::AbstractVector)

BC2 extreme-value family with a finite two-atom spectral representation.

For the classical bivariate model [mai2011bivariate](@cite),

```math
A(t)
=
\\max\\{at,b(1-t)\\}
+
\\max\\{(1-a)t,(1-b)(1-t)\\}.
```

Copulas.jl also accepts a vector `a=(a₁,…,a_d)` and uses the direct
`d`-dimensional two-atom spectral extension

```math
\\ell(x)
=
\\max_i(a_i x_i)
+
\\max_i((1-a_i)x_i).
```

The vector length determines `d`; a vector of length two is reduced to the
specialized bivariate representation.

!!! note "Copulas.jl multivariate parameterization"
    The bivariate BC2 model is documented in [mai2011bivariate](@cite). The
    vector constructor is the direct higher-dimensional two-atom spectral
    construction used by Copulas.jl; general finite spectral constructions are
    discussed in [mai2012simulating](@cite).
"""
BC2Tail, BC2Copula

struct BC2Tail{T} <: BivariatePickandsTail
    a::T
    b::T
    function BC2Tail(a, b)
        T = promote_type(typeof(a), typeof(b))
        (0 ≤ a ≤ 1) || throw(ArgumentError("a must be in [0,1]"))
        (0 ≤ b ≤ 1) || throw(ArgumentError("b must be in [0,1]"))
        return new{T}(T(a), T(b))
    end
end

const BC2Copula{d,T} = ExtremeValueCopula{d, BC2Tail{T}}
Distributions.params(tail::BC2Tail) = (a = tail.a, b = tail.b)
_unbound_params(::Type{<:BC2Tail}, d, θ) = [log(θ.a) - log1p(-θ.a), log(θ.b) - log1p(-θ.b)]
_rebound_params(::Type{<:BC2Tail}, d, α) = begin
    σ(x) = 1 / (1 + exp(-x))
    (; a = σ(α[1]), b = σ(α[2]))
end

function A(tail::BC2Tail, t::Real)
    tt = _safett(t)
    a, b = tail.a, tail.b
    return max(a*tt, b*(1-tt)) + max((1-a)*tt, (1-b)*(1-tt))
end
function _pickands_left_slope(tail::BC2Tail, x::Real)
    R = promote_type(typeof(x), typeof(tail.a), typeof(tail.b))
    a, b = R(tail.a), R(tail.b)
    return iszero(b) ? a - one(R) : isone(b) ? -a : -one(R)
end
function _pickands_right_slope(tail::BC2Tail, x::Real)
    R = promote_type(typeof(x), typeof(tail.a), typeof(tail.b))
    a, b = R(tail.a), R(tail.b)
    return iszero(a) ? one(R) - b : isone(a) ? b : one(R)
end
τ(C::ExtremeValueCopula{2, BC2Tail{T}}) where {T} = 1 - abs(C.tail.a - C.tail.b)
function ρ(C::ExtremeValueCopula{2, BC2Tail{T}}) where {T}
    a, b = C.tail.a, C.tail.b
    num = 2 * (a + b + a*b + max(a,b) - 2a^2 - 2b^2)
    den = (3 - a - b - min(a,b)) * (a + b + max(a,b))
    return num / den
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, BC2Tail{T}}, A::AbstractMatrix{S}) where {T,S<:Real}
    size(A, 1) == 2 || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    a, b = C.tail.a, C.tail.b
    V = rand(rng, S, 2, size(A, 2))
    @inbounds for (j, col) in enumerate(axes(A, 2))
        v1, v2 = V[1, j], V[2, j]
        A[1, col] = max(v1^(1/a), v2^(1/(1-a)))
        A[2, col] = max(v1^(1/b), v2^(1/(1-b)))
    end
    return A
end
function Distributions.logcdf(D::BivEVDistortion{<:BC2Tail{TF1}, TF2}, z::Real) where {TF1,TF2}
    T = promote_type(TF1, TF2, typeof(z))

    a, b = D.tail.a, D.tail.b

    if !(0.0 < z < 1.0)
        return z <= 0 ? T(-Inf) : T(0.0)
    end
    ucond = D.uⱼ
    ucond <= 0 && return _biv_ev_endpoint_logcdf(D, z, true, T)
    ucond >= 1 && return _biv_ev_endpoint_logcdf(D, z, false, T)

    if D.j == 2
        # Condition on V = v, free = u = z
        u = z; v = ucond
        lu, lv = log(u), -D.negloguⱼ
        lhs1, rhs1 = a*lu, b*lv
        lhs2, rhs2 = (1-a)*lu, (1-b)*lv
        c1 = _ev_lt(lhs1, rhs1)  # post-jump side at equality
        c2 = _ev_lt(lhs2, rhs2)
        if c1 && c2
            # C = u, dC/dv = 0
            return T(-Inf)
        elseif c1 && !c2
            # C = u^a v^{1-b}, dC/dv = (1-b) u^a v^{-b}
            logC = a*lu + (1-b)*lv
            factor = (1-b)
            return factor <= 0 ? T(-Inf) : T(logC - log(v) + log(factor))
        elseif !c1 && c2
            # C = v^b u^{1-a}, dC/dv = b v^{b-1} u^{1-a}
            logC = b*lv + (1-a)*lu
            factor = b
            return factor <= 0 ? T(-Inf) : T(logC - log(v) + log(factor))
        else
            # both mins pick v parts: C = v, dC/dv = 1
            logC = lv
            factor = 1.0
            return T(logC - log(v) + log(factor))
        end
    else
        # Condition on U = u, free = v = z
        v = z; u = ucond
        lu, lv = -D.negloguⱼ, log(v)
        lhs1, rhs1 = a*lu, b*lv
        lhs2, rhs2 = (1-a)*lu, (1-b)*lv
        c1 = _ev_le(lhs1, rhs1)  # post-jump side at equality
        c2 = _ev_le(lhs2, rhs2)
        if c1 && c2
            # C = u, dC/du = 1
            logC = lu
            factor = 1.0
            return T(logC - log(u) + log(factor))
        elseif c1 && !c2
            # C = u^a v^{1-b}, dC/du = a u^{a-1} v^{1-b}
            logC = a*lu + (1-b)*lv
            factor = a
            return factor <= 0 ? T(-Inf) : T(logC - log(u) + log(factor))
        elseif !c1 && c2
            # C = v^b u^{1-a}, dC/du = (1-a) v^b u^{-a}
            logC = b*lv + (1-a)*lu
            factor = (1-a)
            return factor <= 0 ? T(-Inf) : T(logC - log(u) + log(factor))
        else
            # both mins pick v parts: C = v, dC/du = 0
            return T(-Inf)
        end
    end
end
function Distributions.quantile(D::BivEVDistortion{BC2Tail{T}, S}, α::Real) where {T, S}
    t = D.uⱼ
    if !(0.0 <= α <= 1.0)
        throw(ArgumentError("α must be in [0,1]"))
    end
    R = promote_type(T, S, typeof(float(α)))
    t <= 0 && return _biv_ev_endpoint_quantile(D, α, true, R)
    t >= 1 && return _biv_ev_endpoint_quantile(D, α, false, R)

    # The two interior atoms can change order; invert the CDF robustly.
    return _unit_quantile(D, α)
end


"""
    BC2MultivariateTail(a)

Multivariate two-atom discrete-spectral extension of the historical BC2 model.
For `a = (a₁,...,a_d)` with every `aᵢ in [0,1]`,

    ℓ(x) = max_i(aᵢ xᵢ) + max_i((1-aᵢ) xᵢ).

For `d=2`, `BC2MultivariateTail([a,b])` reproduces `BC2Tail(a,b)` exactly.
"""
struct BC2MultivariateTail{T} <: DiscreteSpectralBackedTail
    a::Vector{T}
    spectral::DiscreteSpectralTail{T}
end

function BC2MultivariateTail(a::AbstractVector)
    length(a) >= 2 || throw(ArgumentError("BC2MultivariateTail requires at least two coordinates",))

    vals = collect(a)
    T = promote_type(Float64, map(typeof, vals)...)
    aa = T.(a)

    all(isfinite, aa) || throw(ArgumentError("all BC2 weights must be finite",))
    all(v -> zero(T) <= v <= one(T), aa) || throw(ArgumentError("all BC2 weights must lie in [0,1]",))

    B = hcat(aa, one(T) .- aa)
    spectral = DiscreteSpectralTail(B)
    return BC2MultivariateTail{T}(aa, spectral)
end

BC2MultivariateTail(tail::BC2Tail) = BC2MultivariateTail([tail.a, tail.b])
BC2MultivariateCopula(a::AbstractVector) =
    ExtremeValueCopula{length(a)}(BC2MultivariateTail(a))

function _bc2_public_copula(d::Int, a::AbstractVector)
    d == length(a) || throw(DimensionMismatch(
        "d=$d does not match BC2 weight-vector dimension $(length(a))",
    ))
    d == 2 && return ExtremeValueCopula{2}(BC2Tail(a[1], a[2]))
    return ExtremeValueCopula{d}(BC2MultivariateTail(a))
end

function (CT::Type{<:ExtremeValueCopula{D,<:BC2Tail} where D})(a::AbstractVector)
    d = _ev_resolve_dimension(CT, length(a), "BC2 weight-vector")
    return _bc2_public_copula(d, a)
end

(::Type{<:ExtremeValueCopula{D,<:BC2Tail} where D})(d::Int, a::AbstractVector,) =
    _bc2_public_copula(d, a)

BC2MultivariateCopula(tail::BC2Tail) =
    ExtremeValueCopula{2}(BC2MultivariateTail(tail))

Distributions.params(tail::BC2MultivariateTail) = (a = tail.a,)
_is_valid_in_dim(tail::BC2MultivariateTail, d::Int) = length(tail.a) == d
