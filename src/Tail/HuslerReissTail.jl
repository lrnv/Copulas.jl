"""
    HuslerReissTail(θ)
    HuslerReissTail(Γ)
    HuslerReissCopula{d}(θ)
    HuslerReissCopula(d, θ)
    HuslerReissCopula{d}(Γ)
    HuslerReissCopula(d, Γ)
    HuslerReissCopula(Γ)

Hüsler-Reiss extreme-value copula.

The internal representation is always a variogram matrix `Γ`. Scalar constructors
are convenience adapters to the exchangeable variogram with off-diagonal entry

```math
\gamma=\left(\frac{2}{\theta}\right)^2.
```

For `d = 2`, this is exactly the usual one-parameter representation. A `2×2`
variogram is converted back with `θ = 2 / sqrt(Γ[1,2])`.

Special cases are stored in the same matrix representation: the zero variogram
is complete dependence and the variogram with `+Inf` off-diagonal entries is
independence.

See also: [`Tail`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [husler1989maxima](@cite) Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between independence and complete dependence. Statistics & Probability Letters, 7(4), 283-286.
"""
HuslerReissTail, HuslerReissCopula

struct _HuslerReissVariogramGeometry{G} <: Paramorph.TransformVariables.VectorTransform
    d::Int
    interior::G
end
_HuslerReissVariogramGeometry(d::Integer) =
    _HuslerReissVariogramGeometry(Int(d), Paramorph.variogram_matrix(d))
Paramorph.TransformVariables.dimension(t::_HuslerReissVariogramGeometry) =
    Paramorph.TransformVariables.dimension(t.interior)

function _hr_zero_variogram(::Type{T}, d::Int) where {T}
    return zeros(T, d, d)
end
function _hr_independence_variogram(::Type{T}, d::Int) where {T}
    Γ = fill(T(Inf), d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(T)
    end
    return Γ
end
_hr_is_zero_variogram(Γ::AbstractMatrix) = all(iszero, Γ)
function _hr_is_independence_variogram(Γ::AbstractMatrix)
    d1, d2 = size(Γ)
    d1 == d2 || return false
    @inbounds for j in 1:d1, i in 1:d1
        if i == j
            iszero(Γ[i, j]) || return false
        else
            isinf(Γ[i, j]) && Γ[i, j] > 0 || return false
        end
    end
    return true
end

function Paramorph.TransformVariables.transform_with(
    flag::Paramorph.TransformVariables.NoLogJac,
    t::_HuslerReissVariogramGeometry,
    x::AbstractVector,
    index,
)
    n = Paramorph.TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates)
        if all(>(zero(eltype(coordinates))), coordinates)
            return _hr_independence_variogram(eltype(coordinates), t.d), flag, index + n
        elseif all(<(zero(eltype(coordinates))), coordinates)
            return _hr_zero_variogram(eltype(coordinates), t.d), flag, index + n
        end
    end
    return Paramorph.TransformVariables.transform_with(flag, t.interior, x, index)
end
function Paramorph.TransformVariables.transform_with(
    ::Paramorph.TransformVariables.LogJac,
    t::_HuslerReissVariogramGeometry,
    x::AbstractVector,
    index,
)
    n = Paramorph.TransformVariables.dimension(t)
    coordinates = @view x[index:(index + n - 1)]
    if all(isinf, coordinates)
        if all(>(zero(eltype(coordinates))), coordinates)
            return _hr_independence_variogram(eltype(coordinates), t.d), -Inf, index + n
        elseif all(<(zero(eltype(coordinates))), coordinates)
            return _hr_zero_variogram(eltype(coordinates), t.d), -Inf, index + n
        end
    end
    return Paramorph.TransformVariables.transform_with(
        Paramorph.TransformVariables.LogJac(), t.interior, x, index,
    )
end
Paramorph.TransformVariables.inverse_eltype(
    ::_HuslerReissVariogramGeometry,
    ::Type{M},
) where {T,M<:AbstractMatrix{T}} = float(T)
function Paramorph.TransformVariables.inverse_at!(
    x::AbstractVector,
    index,
    t::_HuslerReissVariogramGeometry,
    Γ::AbstractMatrix,
)
    size(Γ) == (t.d, t.d) || throw(DimensionMismatch(
        "expected a $(t.d) × $(t.d) matrix",
    ))
    n = Paramorph.TransformVariables.dimension(t)
    if _hr_is_zero_variogram(Γ)
        fill!(@view(x[index:(index + n - 1)]), -Inf)
        return index + n
    elseif _hr_is_independence_variogram(Γ)
        fill!(@view(x[index:(index + n - 1)]), Inf)
        return index + n
    end
    return Paramorph.TransformVariables.inverse_at!(x, index, t.interior, Γ)
end

Paramorph.@paramorph T struct HuslerReissTail{d,T<:Real} <: BivariatePickandsTail
    Γ::Matrix{T} ~ _HuslerReissVariogramGeometry(d)
end

function _hr_exchangeable_variogram(d::Int, θ::Real)
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    T = typeof(float(θ))
    θf = T(θ)
    γ = abs2(T(2) / θf)
    Γ = fill(γ, d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(T)
    end
    return Γ
end

function _hr_tail_from_matrix(::Val{d}, Γ::AbstractMatrix) where {d}
    size(Γ) == (d, d) || throw(DimensionMismatch(
        "variogram dimension $(size(Γ)) does not match d=$d",
    ))
    T = float(eltype(Γ))
    G = Matrix{T}(Γ)
    if !_hr_is_independence_variogram(G)
        all(isfinite, G) || throw(ArgumentError("Γ must contain only finite entries"))
    end
    try
        return HuslerReissTail{d,T}(G)
    catch err
        (err isa DomainError || err isa LinearAlgebra.PosDefException) || rethrow()
        throw(ArgumentError("Γ must be a strict Hüsler-Reiss variogram or a supported limit"))
    end
end

function (::Type{HuslerReissTail{d}})(θ::Real) where {d}
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
    return _hr_tail_from_matrix(Val(d), _hr_exchangeable_variogram(d, θ))
end
HuslerReissTail(θ::Real) = HuslerReissTail{2}(θ)

(::Type{HuslerReissTail{d}})(Γ::AbstractMatrix) where {d} =
    _hr_tail_from_matrix(Val(d), Γ)
function HuslerReissTail(Γ::AbstractMatrix)
    size(Γ, 1) == size(Γ, 2) || throw(DimensionMismatch("variogram must be square"))
    return HuslerReissTail{size(Γ, 1)}(Γ)
end

@inline function limit_kind(tail::HuslerReissTail, ::Val)
    _hr_is_independence_variogram(tail.Γ) && return Π_LIMIT
    _hr_is_zero_variogram(tail.Γ) && return M_LIMIT
    return NO_LIMIT
end
@inline _hr_is_independent(tail::HuslerReissTail) =
    limit_kind(tail, Val(size(tail.Γ, 1))) === Π_LIMIT

const HuslerReissCopula{d,T} = ExtremeValueCopula{d,HuslerReissTail{d,T}}

function (::Type{HuslerReissCopula{d}})(θ::Real) where {d}
    return _wrap_extreme_value(Val(d), HuslerReissTail{d}(θ))
end
(::Type{HuslerReissCopula})(d::Int, θ::Real) =
    _wrap_extreme_value(Val(d), HuslerReissTail{d}(θ))

function (::Type{HuslerReissCopula{d}})(Γ::AbstractMatrix) where {d}
    return _wrap_extreme_value(Val(d), HuslerReissTail{d}(Γ))
end
HuslerReissCopula(Γ::AbstractMatrix) =
    _wrap_extreme_value(Val(size(Γ, 1)), HuslerReissTail(Γ))
function (::Type{HuslerReissCopula})(d::Int, Γ::AbstractMatrix)
    return _wrap_extreme_value(Val(d), HuslerReissTail(Γ))
end

_is_valid_in_dim(::HuslerReissTail{D}, d::Int) where {D} = D == d

_hr_theta(tail::HuslerReissTail{2}) = 2 / sqrt(tail.Γ[1, 2])
function _hr_variogram(tail::HuslerReissTail{D}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    return tail.Γ
end

Distributions.params(C::ExtremeValueCopula{2,<:HuslerReissTail}) =
    (_hr_theta(C.tail),)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail}) where {D} =
    (copy(C.tail.Γ),)

_tail_constructor_parameter_names(::Type{<:HuslerReissTail{2}}, _) = (:θ,)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail}, _) = (:Γ,)

_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, d,
) = d == 2 ? (:mle, :itau, :irho, :ibeta, :iupper) : ()

function A(tail::HuslerReissTail{2}, t::Real)
    tt = _safett(t)
    θ = _hr_theta(tail)
    θ == 0 && return 1.0
    isinf(θ) && return max(tt, 1 - tt)
    Φ = Distributions.cdf
    N = Distributions.Normal()
    term1 = tt * Φ(N, inv(θ) + 0.5 * θ * log(tt / (1 - tt)))
    term2 = (1 - tt) * Φ(N, inv(θ) + 0.5 * θ * log((1 - tt) / tt))
    return term1 + term2
end
function _hr_stdf(Γ::AbstractMatrix, x)
    d = length(x)
    any(isinf, x) && return Inf

    active = findall(!iszero, x)
    isempty(active) && return 0.0
    length(active) == 1 && return Float64(x[only(active)])
    length(active) < d && return _hr_stdf(Γ[active, active], x[active])

    scale = maximum(x)
    y = Float64.(x) ./ Float64(scale)

    out = 0.0
    for i in 1:d
        yi = y[i]
        J = [j for j in 1:d if j != i]
        q = length(J)
        upper = Vector{Float64}(undef, q)
        R = Matrix{Float64}(undef, q, q)

        @inbounds for a in 1:q
            j = J[a]
            γij = Float64(Γ[i, j])
            σij = sqrt(γij)
            upper[a] = 0.5 * σij + log(yi / y[j]) / σij
            R[a, a] = 1.0

            for b in 1:a-1
                k = J[b]
                γik = Float64(Γ[i, k])
                ρ = (γij + γik - Float64(Γ[j, k])) /
                    (2 * sqrt(γij * γik))
                R[a, b] = R[b, a] = ρ
            end
        end

        probability = q == 1 ?
            Distributions.cdf(Distributions.Normal(), upper[1]) :
            MvNormalCDF.mvnormcdf(
                R,
                fill(-Inf, q),
                upper;
                rng=Random.Xoshiro(0),
            )[1]
        out += yi * probability
    end
    return Float64(scale) * out
end

function ℓ(tail::HuslerReissTail{D}, x) where {D}
    kind = limit_kind(tail, Val(D))
    kind === Π_LIMIT && return sum(x)
    kind === M_LIMIT && return maximum(x)
    if D == 2
        x1, x2 = x
        (isinf(x1) || isinf(x2)) && return max(x1, x2)
        s = x1 + x2
        iszero(s) && return zero(s)
        return s * A(tail, x1 / s)
    end
    return _hr_stdf(tail.Γ, x)
end

function _hr_anchor_covariance(Γ::AbstractMatrix, k::Int)
    d = size(Γ, 1)
    J = [j for j in 1:d if j != k]
    q = length(J)
    Σ = Matrix{Float64}(undef, q, q)
    @inbounds for a in 1:q
        i = J[a]
        for b in 1:q
            j = J[b]
            Σ[a, b] = 0.5 * (Float64(Γ[i, k]) + Float64(Γ[j, k]) - Float64(Γ[i, j]))
        end
    end
    return J, Σ
end

function _ellpartial_signlog(tail::HuslerReissTail, x, I::Tuple{Vararg{Int}})
    if _hr_is_independent(tail)
        isempty(I) && return 1, log(float(sum(x)))
        length(I) == 1 && return 1, zero(float(first(x)))
        return 0, oftype(float(first(x)), -Inf)
    end

    Γ = _hr_variogram(tail, length(x))
    isempty(I) && return 1, log(_hr_stdf(Γ, x))
    all(xi -> xi >= 0, x) || return 0, -Inf
    all(i -> x[i] > 0, I) || return 0, -Inf

    active = findall(>(0), x)
    if length(active) < length(x)
        length(active) == 1 && return length(I) == 1 ? (1, 0.0) : (0, -Inf)
        positions = Dict(i => k for (k, i) in pairs(active))
        reduced_I = Tuple(positions[i] for i in I)
        reduced_tail = HuslerReissTail(Γ[active, active])
        return _ellpartial_signlog(reduced_tail, x[active], reduced_I)
    end

    d = length(x)
    k = first(I)
    Aidx = Base.tail(I)
    C = Tuple(i for i in 1:d if i ∉ I)

    J, Σ = _hr_anchor_covariance(Γ, k)
    pos = Dict(j => a for (a, j) in enumerate(J))
    t = [log(Float64(x[k] / x[j])) + 0.5 * Float64(Γ[k, j]) for j in J]

    apos = [pos[i] for i in Aidx]
    cpos = [pos[i] for i in C]

    logϕ = 0.0
    tA = Float64[]
    ΣAA = zeros(0, 0)
    if !isempty(apos)
        tA = t[apos]
        ΣAA = Σ[apos, apos]
        q = length(tA)
        logϕ = q == 1 ?
            Distributions.logpdf(
                Distributions.Normal(0.0, sqrt(ΣAA[1, 1])),
                tA[1],
            ) :
            Distributions.logpdf(
                Distributions.MvNormal(zeros(q), LinearAlgebra.Symmetric(ΣAA)),
                tA,
            )
    end

    logΦ = 0.0
    if !isempty(cpos)
        tC = t[cpos]
        if isempty(apos)
            μC = zeros(length(cpos))
            Σcond = Σ[cpos, cpos]
        else
            ΣCA = Σ[cpos, apos]
            ΣAC = Σ[apos, cpos]
            F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(ΣAA))
            μC = ΣCA * (F \ tA)
            Σcond = Σ[cpos, cpos] - ΣCA * (F \ ΣAC)
            Σcond = Matrix(LinearAlgebra.Symmetric(Σcond))
        end
        q = length(tC)
        if q == 1
            logΦ = Distributions.logcdf(
                Distributions.Normal(μC[1], sqrt(Σcond[1, 1])),
                tC[1],
            )
        else
            p = MvNormalCDF.mvnormcdf(
                μC,
                Matrix(Σcond),
                fill(-Inf, q),
                tC;
                rng=Random.Xoshiro(0),
            )[1]
            logΦ = iszero(p) ? -Inf : log(p)
        end
    end

    logjac = isempty(Aidx) ? 0.0 : sum(log(Float64(x[i])) for i in Aidx)
    logabs = logϕ + logΦ - logjac
    return isodd(length(I)) ? 1 : -1, logabs
end

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:HuslerReissTail},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        Γ = C.tail.Γ

        roots = Vector{Vector{Int}}(undef, d)
        means = Vector{Vector{Float64}}(undef, d)
        factors = Vector{Matrix{Float64}}(undef, d)

        for m in 1:d
            J, Σ = _hr_anchor_covariance(Γ, m)
            roots[m] = J
            means[m] = [-0.5 * Float64(Γ[j, m]) for j in J]
            factors[m] = Matrix(
                LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ)).L,
            )
        end

        logw = Vector{Float64}(undef, d)
        logz = Vector{Float64}(undef, d)
        ε = Vector{Float64}(undef, d - 1)
        work = Vector{Float64}(undef, d - 1)

        for col in axes(X, 2)
            fill!(logz, -Inf)

            arrival = Random.randexp(rng) / d
            logradius = -log(arrival)

            while logradius > minimum(logz)
                m = rand(rng, 1:d)
                J = roots[m]
                μ = means[m]
                L = factors[m]

                Random.randn!(rng, ε)
                LinearAlgebra.mul!(work, L, ε)

                logw[m] = 0.0
                @inbounds for a in eachindex(J)
                    logw[J[a]] = μ[a] + work[a]
                end

                lognorm = LogExpFunctions.logsumexp(logw)
                @inbounds for i in 1:d
                    candidate = logradius + logw[i] - lognorm
                    logz[i] = max(logz[i], candidate)
                end

                arrival += Random.randexp(rng) / d
                logradius = -log(arrival)
            end

            @inbounds for i in 1:d
                X[i, col] = exp(-exp(-logz[i]))
            end
        end

        return X
    end
end

function dA(tail::HuslerReissTail{2}, t::Real)
    θ = _hr_theta(tail)
    iszero(θ) && return zero(t * θ)
    N = Distributions.Normal()
    Φ = Distributions.cdf
    ϕ = Distributions.pdf

    arg1 = inv(θ) + 0.5 * θ * log(t / (1 - t))
    arg2 = inv(θ) + 0.5 * θ * log((1 - t) / t)

    dA_term1 = Φ(N, arg1) + t * ϕ(N, arg1) * (0.5 * θ * (1 / t + 1 / (1 - t)))
    dA_term2 = -Φ(N, arg2) + (1 - t) * ϕ(N, arg2) * (0.5 * θ * (-1 / t - 1 / (1 - t)))

    return dA_term1 + dA_term2
end
function d²A(tail::HuslerReissTail{2}, t::Real)
    θ = _hr_theta(tail)
    iszero(θ) && return zero(t * θ)
    N = Distributions.Normal()
    ϕ = Distributions.pdf
    invθ = inv(θ)
    L = log(t / (1 - t))
    a1 = invθ + 0.5 * θ * L
    a2 = invθ - 0.5 * θ * L
    s = 1 / t + 1 / (1 - t)
    s2 = -1 / t^2 + 1 / (1 - t)^2
    a1p = 0.5 * θ * s
    a1pp = 0.5 * θ * s2
    ϕ1 = ϕ(N, a1)
    ϕ2 = ϕ(N, a2)
    return 2 * (ϕ1 + ϕ2) * a1p +
           t * ϕ1 * (a1pp - a1 * a1p^2) +
           (1 - t) * ϕ2 * (-a1pp - a2 * a1p^2)
end

_tau_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(HuslerReissTail(θ), t) * t * (1 - t) / max(A(HuslerReissTail(θ), t), _δ(t)), 0, 1; kw...)[1]
_rho_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12 * QuadGK.quadgk(t -> inv(1 + A(HuslerReissTail(θ), t))^2, 0, 1; kw...)[1] - 3

τ(C::ExtremeValueCopula{2,<:HuslerReissTail}) = _tau_HuslerReiss(_hr_theta(C.tail))
ρ(C::ExtremeValueCopula{2,<:HuslerReissTail}) = _rho_HuslerReiss(_hr_theta(C.tail))
λᵤ(C::ExtremeValueCopula{2,<:HuslerReissTail}) =
    2 * (1 - Distributions.cdf(Distributions.Normal(), 1 / _hr_theta(C.tail)))
β(C::ExtremeValueCopula{2,<:HuslerReissTail}) =
    4^(1 - Distributions.cdf(Distributions.Normal(), 1 / _hr_theta(C.tail))) - 1

τ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, τ; kw...) =
    τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_HuslerReiss(θ) - τ; kw...)
τ⁻¹(::Type{<:HuslerReissTail}, τ; kw...) =
    τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_HuslerReiss(θ) - τ; kw...)
ρ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, ρ; kw...) =
    ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_HuslerReiss(θ) - ρ; kw...)
λᵤ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, λ) =
    1 / Distributions.quantile(Distributions.Normal(), 1 - λ / 2)
function β⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, beta)
    p = 1 - log(beta + 1) / log(4)
    p = clamp(p, eps(), 1 - eps())
    return 1 / Distributions.quantile(Distributions.Normal(), p)
end
