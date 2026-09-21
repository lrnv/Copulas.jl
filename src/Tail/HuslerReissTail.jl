"""
    HuslerReissTail(θ)
    HuslerReissTail(Γ)
    HuslerReissCopula{d}(θ)
    HuslerReissCopula(d, θ)
    HuslerReissCopula{d}(Γ)
    HuslerReissCopula(d, Γ)
    HuslerReissCopula(Γ)

Hüsler-Reiss extreme-value copula.

`HuslerReissCopula(d, θ)` is the exchangeable representation with
`θ ∈ [0,∞]`. For `d > 2`, it corresponds to a variogram with constant
off-diagonal entry

```math
\\gamma=\\left(\\frac{2}{\\theta}\\right)^2.
```

`HuslerReissCopula(Γ)` is the general variogram representation. The square
matrix `Γ` determines the dimension. It must be finite and symmetric, with
zero diagonal, and must satisfy the Hüsler-Reiss variogram validity conditions.
For a non-degenerate `d ≥ 3` representation, off-diagonal entries are strictly
positive and the variogram is strictly conditionally negative definite.

A `2×2` variogram is equivalent to the scalar parameter
``\\theta=2/\\sqrt{\\Gamma_{12}}``. In dimension two,

```math
A(t)
=
t\\Phi\\!\\left(\\theta^{-1}+\\frac{\\theta}{2}\\log\\frac{t}{1-t}\\right)
+
(1-t)\\Phi\\!\\left(\\theta^{-1}+\\frac{\\theta}{2}\\log\\frac{1-t}{t}\\right).
```

Special cases:

* `θ = 0` represents `IndependentCopula(d)`.
* `θ = ∞`, or an all-zero variogram, represents `MCopula(d)`.

The scalar `θ` and variogram magnitudes run in opposite directions: larger
`θ` means stronger dependence, whereas smaller variogram entries do. Near a
degenerate variogram boundary, likelihood and matrix calculations can become
ill-conditioned.

See also: [`Tail`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [husler1989maxima](@cite) Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between independence and complete dependence. Statistics & Probability Letters, 7(4), 283-286.
"""
HuslerReissTail, HuslerReissCopula

_hr_parameter_geometry(::Val{:exchangeable}, ::Int, ::Type{T}) where {T<:Real} =
    Paramorph.nonnegative()
_hr_parameter_geometry(::Val{:general}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.variogram_matrix(d)

Paramorph.@paramorph T struct HuslerReissTail{d,R,T<:Real} <: OneParameterPickandsTail
    parameter::Union{T,Matrix{T}} ~ _hr_parameter_geometry(Val(R), d, T)
end

function _hr_exchangeable_variogram(d::Int, θ::Real)
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    γ = abs2(2 / θ)
    Γ = fill(float(γ), d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(eltype(Γ))
    end
    return Γ
end

function _hr_general_tail(::Val{d}, Γ::AbstractMatrix) where {d}
    size(Γ) == (d, d) || throw(DimensionMismatch(
        "variogram dimension $(size(Γ)) does not match d=$d",
    ))
    T = float(eltype(Γ))
    G = Matrix{T}(Γ)
    all(isfinite, G) || throw(ArgumentError("Γ must contain only finite entries"))

    # The zero variogram is the complete-dependence boundary.  Keep exact
    # boundary points in the exchangeable chart; the general matrix chart is
    # deliberately the strict variogram interior.
    all(iszero, G) && return HuslerReissTail{d,:exchangeable,T}(T(Inf))

    try
        return HuslerReissTail{d,:general,T}(G)
    catch err
        (err isa DomainError || err isa LinearAlgebra.PosDefException) || rethrow()
        throw(ArgumentError("Γ must be a strict Hüsler-Reiss variogram"))
    end
end

function (::Type{HuslerReissTail{d}})(θ::Real) where {d}
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
    θf = float(θ)

    # In d=2 the scalar and general representations are the same one-dimensional
    # model. Canonicalize interior scalar input to the general matrix geometry;
    # retain the scalar chart only for exact limit points, where the strict
    # variogram chart has no finite matrix representative.
    if d == 2 && isfinite(θf) && !iszero(θf)
        return _hr_general_tail(Val(d), _hr_exchangeable_variogram(d, θf))
    end
    return HuslerReissTail{d,:exchangeable,typeof(θf)}(θf)
end
HuslerReissTail(θ::Real) = HuslerReissTail{2}(θ)

(::Type{HuslerReissTail{d}})(Γ::AbstractMatrix) where {d} =
    _hr_general_tail(Val(d), Γ)
HuslerReissTail(Γ::AbstractMatrix) =
    HuslerReissTail{size(Γ, 1)}(Γ)

@inline _hr_representation(::HuslerReissTail{d,R}) where {d,R} = R
@inline _hr_is_independent(tail::HuslerReissTail) =
    _hr_representation(tail) === :exchangeable && iszero(tail.parameter)
@inline function limit_kind(tail::HuslerReissTail, ::Val)
    _hr_representation(tail) === :general && return NO_LIMIT
    iszero(tail.parameter) && return Π_LIMIT
    isinf(tail.parameter) && return M_LIMIT
    return NO_LIMIT
end

const HuslerReissCopula{d,R,T} = ExtremeValueCopula{d,HuslerReissTail{d,R,T}}

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
    # Infer the tail's own dimension first so the runtime-dimension form keeps
    # ExtremeValueCopula's ArgumentError contract on a mismatch.
    return _wrap_extreme_value(Val(d), HuslerReissTail(Γ))
end

_is_valid_in_dim(::HuslerReissTail{D}, d::Int) where {D} = D == d

_hr_theta(tail::HuslerReissTail{D,:exchangeable}) where {D} = tail.parameter
_hr_theta(tail::HuslerReissTail{D,:general}) where {D} =
    2 / sqrt(tail.parameter[1, 2])
function _hr_variogram(tail::HuslerReissTail{D,:exchangeable}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    return _hr_exchangeable_variogram(d, tail.parameter)
end
_hr_variogram(tail::HuslerReissTail{D,:general}, d::Int) where {D} = begin
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    tail.parameter
end

# Keep the public natural parameter convention stable: bivariate Hüsler-Reiss
# is exposed by its scalar θ even though the interior is stored in the general
# 2×2 variogram representation.
Distributions.params(C::ExtremeValueCopula{2,<:HuslerReissTail}) =
    (_hr_theta(C.tail),)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{D,:exchangeable}}) where {D} =
    (C.tail.parameter,)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{D,:general}}) where {D} =
    (copy(C.tail.parameter),)

_tail_constructor_parameter_names(::Type{<:HuslerReissTail{2}}, _) = (:θ,)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail{D,:exchangeable}}, _) where {D} = (:θ,)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail{D,:general}}, _) where {D} = (:Γ,)

# General d>2 variograms remain construction/evaluation objects rather than a
# default fitted family. In d=2 the general chart is one-dimensional and is the
# canonical representation of the ordinary Hüsler-Reiss family.
_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:HuslerReissTail{D,:general}} where D}, d,
) = d == 2 ? (:mle, :itau, :irho, :ibeta, :iupper) : ()

function A(tail::HuslerReissTail, t::Real)
    tt = _safett(t)
    θ = _hr_theta(tail)
    θ == 0 && return 1.0
    isinf(θ) && return max(tt, 1-tt)
    Φ = Distributions.cdf
    N = Distributions.Normal()
    term1 = tt * Φ(N, inv(θ) + 0.5*θ*log(tt/(1-tt)))
    term2 = (1-tt) * Φ(N, inv(θ) + 0.5*θ*log((1-tt)/tt))
    return term1 + term2
end
function _hr_stdf(Γ::AbstractMatrix, x)
    d = length(x)
    any(isinf, x) && return Inf

    # Remove zero coordinates before numerical Gaussian integration so that
    # marginal consistency is exact rather than delegated to QMC at +Inf.
    active = findall(!iszero, x)
    isempty(active) && return 0.0
    length(active) == 1 && return Float64(x[only(active)])
    length(active) < d && return _hr_stdf(Γ[active, active], x[active])

    # Normalize once. This makes the numerical evaluation inherit the exact
    # one-homogeneity of the STDF as closely as floating-point arithmetic allows.
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

function ℓ(tail::HuslerReissTail{D,:exchangeable}, x) where {D}
    θ = tail.parameter
    d = length(x)

    # Keep the historical bivariate route AD-friendly. The general
    # multivariate implementation below uses MvNormalCDF/Float64 and is not
    # intended to be differentiated by ForwardDiff.
    if d == 2
        x1, x2 = x

        (isinf(x1) || isinf(x2)) && return max(x1, x2)

        s = x1 + x2
        iszero(s) && return zero(s)

        return s * A(tail, x1 / s)
    end

    γ = abs2(2 / θ)
    isinf(γ) && return sum(x)
    iszero(γ) && return maximum(x)

    Γ = fill(float(γ), d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(eltype(Γ))
    end
    return _hr_stdf(Γ, x)
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
        reduced_tail = _hr_representation(tail) === :exchangeable ?
                       HuslerReissTail{length(active)}(_hr_theta(tail)) :
                       HuslerReissTail(Γ[active, active])
        return _ellpartial_signlog(reduced_tail, x[active], reduced_I)
    end

    d = length(x)
    k = first(I)
    A = Base.tail(I)
    C = Tuple(i for i in 1:d if i ∉ I)

    J, Σ = _hr_anchor_covariance(Γ, k)
    pos = Dict(j => a for (a, j) in enumerate(J))
    t = [log(Float64(x[k] / x[j])) + 0.5 * Float64(Γ[k, j]) for j in J]

    apos = [pos[i] for i in A]
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

    logjac = isempty(A) ? 0.0 : sum(log(Float64(x[i])) for i in A)
    logabs = logϕ + logΦ - logjac
    return isodd(length(I)) ? 1 : -1, logabs
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:HuslerReissTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        Γ = _hr_variogram(C.tail, d)

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

ℓ(tail::HuslerReissTail{D,:general}, x) where {D} =
    _hr_stdf(tail.parameter, x)

function dA(tail::HuslerReissTail, t::Real)
    θ = _hr_theta(tail)
    iszero(θ) && return zero(t * θ)
    N = Distributions.Normal()
    Φ = Distributions.cdf
    ϕ = Distributions.pdf

    arg1 = inv(θ) + 0.5*θ*log(t/(1-t))
    arg2 = inv(θ) + 0.5*θ*log((1-t)/t)

    dA_term1 = Φ(N, arg1) + t * ϕ(N, arg1) * (0.5*θ * (1/t + 1/(1-t)))
    dA_term2 = -Φ(N, arg2) + (1-t) * ϕ(N, arg2) * (0.5*θ * (-1/t - 1/(1-t)))

    return dA_term1 + dA_term2
end
function d²A(tail::HuslerReissTail, t::Real)
    θ = _hr_theta(tail)
    iszero(θ) && return zero(t * θ)
    N  = Distributions.Normal()
    ϕ  = Distributions.pdf
    invθ = inv(θ)
    L   = log(t/(1 - t))
    a1  = invθ + 0.5*θ*L
    a2  = invθ - 0.5*θ*L
    s   = 1/t + 1/(1 - t)
    s2  = -1/t^2 + 1/(1 - t)^2
    a1p = 0.5*θ*s
    a1pp= 0.5*θ*s2
    ϕ1  = ϕ(N, a1)
    ϕ2  = ϕ(N, a2)
    return 2*(ϕ1 + ϕ2)*a1p + t*ϕ1*(a1pp - a1*a1p^2) + (1 - t)*ϕ2*(-a1pp - a2*a1p^2)
end

_tau_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(HuslerReissTail(θ),t)*t*(1-t)/max(A(HuslerReissTail(θ),t),_δ(t)), 0, 1; kw...)[1]
_rho_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(HuslerReissTail(θ),t))^2, 0, 1; kw...)[1] - 3

τ(C::ExtremeValueCopula{2,<:HuslerReissTail}) = _tau_HuslerReiss(_hr_theta(C.tail))
ρ(C::ExtremeValueCopula{2,<:HuslerReissTail}) = _rho_HuslerReiss(_hr_theta(C.tail))
λᵤ(C::ExtremeValueCopula{2,<:HuslerReissTail}) = 2 * (1 - Distributions.cdf(Distributions.Normal(), 1 / _hr_theta(C.tail)))
β(C::ExtremeValueCopula{2,<:HuslerReissTail}) = 4^(1 - Distributions.cdf(Distributions.Normal(), 1 / _hr_theta(C.tail))) - 1

τ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_HuslerReiss(θ) - τ; kw...)
τ⁻¹(::Type{<:HuslerReissTail}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_HuslerReiss(θ) - τ; kw...)
ρ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_HuslerReiss(θ) - ρ; kw...)
λᵤ⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, λ) = 1 / Distributions.quantile(Distributions.Normal(), 1 - λ/2)
function β⁻¹(::Type{<:ExtremeValueCopula{D,<:HuslerReissTail} where D}, beta)
    p = 1 - log(beta + 1) / log(4)
    p = clamp(p, eps(), 1 - eps())
    return 1 / Distributions.quantile(Distributions.Normal(), p)
end
