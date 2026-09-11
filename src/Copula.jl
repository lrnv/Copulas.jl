###############################################################################
#####  Main Copula interface.
#####  User-facing function:
#####       1) Distributions.jl's API: cdf, pdf, logpdf, loglikelyhood, etc..
#####       2) ρ, τ, β, γ, ι, λₗ and λᵤ: repectively the spearman rho, kendall tau,  blomqvist's beta,
#####          gini's gamma,entropy eta, and lower and upper tail dependencies.
#####       3) measure(C, us, vs) that get the measure associated with the copula.
#####       3) pseudos(data) constructs pseudo-observations from a given dataset.
#####
#####  When implementing a new copula, you have to overwrite `Copulas._cdf()`
#####  and `Distributions._rand!()` for matrix inputs.
#####  and you may overwrite ρ, τ, β, γ, ι, λₗ, λᵤ, measure for performances.
###############################################################################
"""
    Copula{d} <: Distributions.ContinuousMultivariateDistribution

Abstract type for a `d`-dimensional copula: a multivariate distribution on the
unit hypercube with uniform univariate margins. Concrete families support the
standard `Distributions.jl` operations documented for that family. The type
parameter records dimension; internal storage parameters of concrete subtypes
are not part of the public contract.

See also: [`SklarDist`](@ref), [`subsetdims`](@ref), [`condition`](@ref),
[`Distributions.fit`](@ref), [`measure`](@ref).
"""
abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end

# Distributions.jl uses `eltype` as the default element type allocated by
# `rand`. Parameter-free copulas therefore sample as Float64 unless a concrete
# family propagates another numeric representation below its own definition.
Base.eltype(::Copula) = Float64
Distributions.partype(C::Copula) = eltype(C)

"""
    CopulaMeasureStyle

Internal measure-capability trait distinguishing copulas with an ordinary
Lebesgue density from copulas with singular or mixed components. Algorithms use
this trait to avoid manufacturing density-based behavior from the historical
`ContinuousMultivariateDistribution` supertype. It is not public API.

See also: [`copula_measure_style`](@ref), [`LimitKind`](@ref), [`Copula`](@ref).
"""
abstract type CopulaMeasureStyle end
struct AbsolutelyContinuousMeasure <: CopulaMeasureStyle end
struct NonAbsolutelyContinuousMeasure <: CopulaMeasureStyle end

"""
    LimitKind

Internal classification of exact parameter limits: no recognized limit,
independence (`Π`), comonotonicity (`M`), or the bivariate lower
Fréchet--Hoeffding bound (`W`). Constructors and algorithms use it to select
mathematically exact boundary behavior. Enum values are not stable API.

See also: [`limit_kind`](@ref), [`CopulaMeasureStyle`](@ref).
"""
@enum LimitKind::UInt8 begin
    NO_LIMIT
    Π_LIMIT
    M_LIMIT
    W_LIMIT
end

"""
    copula_measure_style(C)

Return the internal `CopulaMeasureStyle` of `C`. The default assumes absolute
continuity; singular or mixed families and exact parameter limits must
specialize it. This trait controls density-dependent generic operations and is
not a downstream extension contract.

See also: [`CopulaMeasureStyle`](@ref), [`limit_kind`](@ref),
[`Distributions.logpdf`](@ref).
"""
copula_measure_style(::Type{<:Copula}) = AbsolutelyContinuousMeasure()
copula_measure_style(C::Copula) = copula_measure_style(typeof(C))

Base.broadcastable(C::Copula) = Ref(C)
Base.length(::Copula{d}) where d = d

function _rand_M!(rng::Distributions.AbstractRNG, A::AbstractMatrix{T}) where {T<:Real}
    Random.rand!(rng, view(A, 1, :))
    @inbounds for row in 2:size(A, 1)
        A[row, :] .= view(A, 1, :)
    end
    return A
end

function _rand_W!(rng::Distributions.AbstractRNG, A::AbstractMatrix{T}) where {T<:Real}
    size(A, 1) == 2 || throw(ArgumentError("W limit only exists in dimension 2"))
    Random.rand!(rng, view(A, 1, :))
    @inbounds for col in axes(A, 2)
        A[2, col] = one(T) - A[1, col]
    end
    return A
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::Copula{d}, x::AbstractVector{T}) where {d,T<:Real}
    length(x) == d || throw(ArgumentError("Dimension mismatch between copula and output vector"))
    Distributions._rand!(rng, C, reshape(x, d, 1))
    return x
end

"""
    Distributions._rand!(rng, C::Copula{d}, X::AbstractMatrix)

Internal sampling primitive for copulas. A concrete implementation fills and
returns the preallocated `d × n` matrix `X`, with one observation per column,
using only `rng` for randomness and preserving the buffer element type. The
public vector sampler delegates to this method. Concrete families must provide
a matrix specialization; callers should use `rand` or `rand!`.

See also: [`_cdf`](@ref), [`Copula`](@ref), [`inverse_rosenblatt`](@ref).
"""
function Distributions._rand!(::Distributions.AbstractRNG, C::Copula{d}, ::AbstractMatrix{T}) where {d,T<:Real}
    throw(ArgumentError("$(typeof(C)) must implement a matrix Distributions._rand! method"))
end
function Distributions.cdf(C::Copula{d},u::VT) where {d,VT<:AbstractVector}
    length(u) != d && throw(ArgumentError("Dimension mismatch between copula and input vector"))
    if any(x -> x <= zero(x), u)
        return zero(u[1])
    elseif all(x -> x >= one(x), u)
        return one(u[1])
    end
    bounded = any(x -> x > one(x), u) ? min.(u, one(eltype(u))) : u
    return _cdf(C, bounded)
end
function Distributions.cdf(C::Copula{d},A::AbstractMatrix) where d
    size(A,1) != d && throw(ArgumentError("Dimension mismatch between copula and input vector"))
    return [Distributions.cdf(C,u) for u in eachcol(A)]
end
Distributions.logcdf(C::Copula, A::AbstractMatrix) = log.(Distributions.cdf(C, A))
Distributions.logcdf(C::Copula, v::AbstractVector) = log(Distributions.cdf(C,v))
@inline function Distributions.logpdf(
    C::Copula{d}, u::AbstractVector{<:Real},
) where d
    @boundscheck length(u) == d || throw(DimensionMismatch(
        "input dimension does not match copula dimension",
    ))
    value = Distributions._logpdf(C, u)
    isnan(value) || return value
    return _resolve_boundary_logpdf(value, u)
end
function _resolve_boundary_logpdf(value, u)
    @inbounds for x in u
        (iszero(x) || isone(x)) && return oftype(value, -Inf)
    end
    return value
end
function Distributions.logpdf(C::Copula{d}, A::AbstractMatrix) where d
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and input matrix"))
    return [Distributions.logpdf(C, u) for u in eachcol(A)]
end
"""
    _cdf(C::Copula, u)

Evaluate the copula CDF at an already dimension-checked point inside the unit
hypercube. Concrete families normally specialize this internal primitive. The
generic fallback numerically integrates `pdf(C, ·)` over `[0,u]` and therefore
requires an ordinary density; it is unsuitable for singular copulas and may be
expensive in high dimension. Public callers must use `cdf`.

See also: [`Distributions._rand!`](@ref), [`copula_measure_style`](@ref),
[`Distributions.cdf`](@ref).
"""
function _cdf(C::CT,u) where {CT<:Copula}
    f(x) = Distributions.pdf(C,x)
    z = zeros(eltype(u),length(C))
    return HCubature.hcubature(f,z,u,rtol=sqrt(eps()))[1]
end

# Multivariate dependence metrics
"""
    ρ(C::Copula)
    ρ(U::AbstractMatrix)

Return multivariate Spearman's rho for a copula or for pseudo-observations
stored as a `d × n` matrix. Family methods may provide exact formulas; the
generic copula method uses numerical integration.

This is the normalized multivariate concordance coefficient based on
`∫_[0,1]^d C(u) du`; it is zero under independence and one under complete
positive dependence. The sample form ranks each row internally, so it can also
be applied to continuous raw observations. Numerical integration becomes
costly as dimension grows and may be less accurate near singular limits.

See also: [`τ`](@ref), [`StatsBase.corspearman`](@ref), [`subsetdims`](@ref).
"""
function ρ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    z = zeros(d)
    i = ones(d)
    r = HCubature.hcubature(F, z, i, rtol=sqrt(eps()))[1]
    value = (2^d * (d+1) * r - d - 1)/(2^d - d - 1)
    return clamp(value, -one(value), one(value))
end

"""
    τ(C::Copula)
    τ(U::AbstractMatrix)

Return multivariate Kendall's tau for a copula or a `d × n` matrix of
pseudo-observations. Family methods may replace the generic expectation-based
calculation with an exact formula.

The population coefficient normalizes `E[C(U)]` for `U ∼ C`; the sample form
counts concordant unordered pairs. It is zero under independence and one under
complete positive dependence. The generic population method uses Monte Carlo
expectation, so repeated calls need not be bitwise identical and exact family
methods should be preferred when available. Ties in sample data do not receive
a dedicated correction.

See also: [`ρ`](@ref), [`StatsBase.corkendall`](@ref), [`subsetdims`](@ref).
"""
function τ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    r = Distributions.expectation(F, C; nsamples=10^4)
    return (2^d / (2^(d-1) - 1)) * r - 1 / (2^(d-1) - 1)
end

"""
    β(C::Copula)
    β(U::AbstractMatrix)

Return multivariate Blomqvist's beta, a median-orthant measure of concordance,
for a copula or pseudo-observations stored by columns.

In two dimensions this is `4C(1/2,1/2)-1`; the multivariate extension combines
the lower and upper median orthants. Independence maps to zero and complete
positive dependence to one. The data form expects values already represented
on the uniform scale and classifies observations relative to `1/2`; use
`pseudos` first for raw continuous margins.

See also: [`corblomqvist`](@ref), [`pseudos`](@ref), [`τ`](@ref).
"""
function β(C::Copula{d}) where {d}
    d == 2 && return 4*Distributions.cdf(C, [0.5, 0.5]) - 1
    u     = fill(0.5, d)
    C0    = Distributions.cdf(C, u)
    Cbar0 = Distributions.cdf(SurvivalCopula(C, Tuple(1:d)), u)
    return (2.0^(d-1) * C0 + Cbar0 - 1) / (2^(d-1) - 1)
end

"""
    γ(C::Copula)
    γ(U::AbstractMatrix)

Return multivariate Gini's gamma for a copula or for pseudo-observations stored
as a `d × n` matrix. The generic copula method estimates the defining
expectation numerically.

The normalization maps independence to zero and complete positive dependence
to one. The sample form expects uniform-scale observations and replaces the
population expectation by an empirical average. The generic copula method uses
Monte Carlo expectation; its result therefore has sampling error unless a
family supplies an exact specialization.

See also: [`corgini`](@ref), [`pseudos`](@ref), [`ρ`](@ref).
"""
function γ(C::Copula{d}) where {d}
    _integrand(u) = (1 + minimum(u) - maximum(u) + max(abs(sum(u) - d/2) - (d - 2)/2, 0.0)) / 2
    I = Distributions.expectation(_integrand, C; nsamples=10^4)
    a = 1/(d+1) + _div_factorial(one(float(I)), d+1)   # independence
    b = (2 + 4.0^(1-d)) / 3          # comonotonicity
    return (I - a) / (b - a)
end

"""
    ι(C::Copula)
    ι(U::AbstractMatrix; k=5, p=Inf, leafsize=32)

Return copula entropy. For a copula, this is the expected negative log-density
and therefore requires an ordinary Lebesgue density. For data, a nearest-neighbor
entropy estimator is applied to the `d × n` pseudo-observation matrix.

With the sign convention used here, independence has entropy zero and an
absolutely continuous dependent copula has a non-positive value. The data
estimator uses the `k`th neighbor under the Minkowski `p`-norm; `leafsize`
controls only search performance. It requires at least `k+1` observations and
can be sensitive to ties, boundary effects and the choice of `k`. It is not a
definition of entropy for singular copulas.

See also: [`corentropy`](@ref), [`Copula`](@ref), [`pseudos`](@ref).
"""
function ι(C::Copula{d}) where {d}
    return Distributions.expectation(u -> -Distributions.logpdf(C, u), C; nsamples=10^4)
end

"""
    λₗ(C::Copula; ε=1e-10)
    λₗ(U::AbstractMatrix; p=nothing)

Return lower-tail dependence. The generic copula method extrapolates diagonal
CDF ratios near zero; the data method estimates joint lower-tail frequency at
threshold `p`, defaulting to `1/√n`. Family-specific exact formulas take
precedence when available.

For a `d × n` input, rows are variables, columns are observations, and values
must already be on the uniform scale. Smaller `p` targets a more extreme region
but uses fewer observations. Likewise, `ε` is a numerical extrapolation scale,
not a statistical tolerance; results can be unstable when a closed form is
unavailable.

See also: [`λᵤ`](@ref), [`corlowertail`](@ref), [`pseudos`](@ref).
"""
function λₗ(C::Copula{d}; ε::Float64 = 1e-10) where {d}
    g(e) = Distributions.cdf(C, fill(e, d)) / e
    return clamp(2*g(ε/2) - g(ε), 0.0, 1.0)
end

"""
    λᵤ(C::Copula; ε=1e-10)
    λᵤ(U::AbstractMatrix; p=nothing)

Return upper-tail dependence. The generic copula method applies the lower-tail
calculation to the survival copula; the data method estimates joint upper-tail
frequency at threshold `p`, defaulting to `1/√n`. Family-specific exact formulas
take precedence when available.

For a `d × n` input, rows are variables, columns are observations, and values
must already be on the uniform scale. Smaller `p` targets a more extreme region
but uses fewer observations. Likewise, `ε` is a numerical extrapolation scale,
not a statistical tolerance; results can be unstable when a closed form is
unavailable.

See also: [`λₗ`](@ref), [`coruppertail`](@ref), [`pseudos`](@ref).
"""
function λᵤ(C::Copula{d}; ε::Float64 = 1e-10) where {d}
    Sc   = SurvivalCopula(C, Tuple(1:d))
    f(e) = Distributions.cdf(Sc, fill(e, d)) / e
    return clamp(2*f(ε/2) - f(ε), 0.0, 1.0)
end

# Multivariate dependence metrics applied to a matrix.
function β(U::AbstractMatrix)
    # Assumes psuedo-data given. β multivariate (Hofert–Mächler–McNeil, ec. (7))
    d, n = size(U)
    count = sum(j -> all(U[:, j] .<= 0.5) || all(U[:, j] .> 0.5), 1:n)
    h_d = 2.0^(d-1) / (2.0^(d-1) - 1.0)
    return h_d * (count/n - 2.0^(1-d))
end
function τ(U::AbstractMatrix)
    # Sample version of multivariate Kendall's tau for pseudo-data
    d, n = size(U)
    comp = 0
    @inbounds for j in 2:n, i in 1:j-1
        uᵢ = @view U[:, i]; uⱼ = @view U[:, j]
        comp += (all(uᵢ .<= uⱼ) || all(uᵢ .>= uⱼ))
    end
    pc = comp / (n*(n-1)/2)
    return (2.0^d * pc - 2.0) / (2.0^d - 2.0)
end
function ρ(U::AbstractMatrix)
    # Sample version of multivariate Spearman's rho for pseudo-observations
    d, n = size(U)
    R = hcat((StatsBase.tiedrank(U[k, :]) for k in 1:d)...)   # n×d
    μ = Statistics.mean(prod(R, dims=2)) / (n + 1)^d          # ≈ E[∏ U_i]
    h = (d + 1) / (2.0^d - (d + 1))
    return h * (2.0^d * μ - 1.0)
end
function γ(U::AbstractMatrix)
    d, n = size(U)
    I = zero(eltype(U))
    for j in 1:n
        u = U[:,j]
        I += (1 + minimum(u) - maximum(u) + max(abs(sum(u) - d/2) - (d - 2)/2, 0.0)) / 2
    end
    I /= n
    a = 1/(d+1) + _div_factorial(one(float(I)), d+1)
    b = (2 + 4.0^(1-d)) / 3
    return (I - a) / (b - a)
end
function _λ(U::AbstractMatrix; t::Symbol=:upper, p::Union{Nothing,Real}=nothing)
    # Assumes pseudo-data given. Multivariate tail’s lambda (Schmidt, R. & Stadtmüller, U. 2006)
    p === nothing && (p = 1/sqrt(size(U, 2)))
    (0 < p < 1) || throw(ArgumentError("p must be in (0,1)"))
    in_tail = t=== :upper ? Base.Fix2(>=, 1-p) : Base.Fix2(<=, p)
    prob = Statistics.mean(all(in_tail, U, dims=1))
    return clamp(prob/p, 0.0, 1.0)
end
λₗ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:lower, p=p)
λᵤ(U::AbstractMatrix; p::Union{Nothing,Real}=nothing) = _λ(U; t=:upper, p=p)
function ι(U::AbstractMatrix; k::Int=5, p::Real=Inf, leafsize::Int=32)
    # Assumes pseudo-data given. Multivariate copula entropy (L.F. Kozachenko and N.N. Leonenko., 1987)
    d, n = size(U)
    n ≥ k+1 || throw(ArgumentError("n ≥ k+1 is required"))
    (p ≥ 1 || isinf(p)) || throw(ArgumentError("invalid Minkowski norm: p ∈ [1,∞]"))
    any(isnan, U) && return NaN
    X = Array{Float64}(U)
    lp(u, v, p) = (sum(abs.(u .- v) .^ p))^(1/p)
    cheb(u, v)  = maximum(abs.(u .- v))
    function lb_lp(q, lo, hi, p)
        s = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            s += δ^p
        end
        return s^(1/p)
    end
    function lb_inf(q, lo, hi)
        m = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            m = δ > m ? δ : m
        end
        return m
    end
    nodes = Vector{Vector{Any}}()
    function build(idxs::Vector{Int})
        lo = fill( Inf, d); hi = fill(-Inf, d)
        @inbounds for j in idxs, r in 1:d
            v = X[r, j]
            lo[r] = v < lo[r] ? v : lo[r]
            hi[r] = v > hi[r] ? v : hi[r]
        end
        if length(idxs) ≤ leafsize
            push!(nodes, Any[copy(idxs), 0, 0.0, 0, 0, lo, hi])  # hoja
            return length(nodes)
        end
        spans = hi .- lo
        sd = findmax(spans)[2]
        sv = (lo[sd] + hi[sd]) / 2
        left = Int[]; right = Int[]
        @inbounds for j in idxs
            (X[sd, j] ≤ sv ? push!(left, j) : push!(right, j))
        end
        if isempty(left) || isempty(right)
            ord = sort(idxs; by = j -> X[sd, j]); m = length(ord) ÷ 2
            left  = ord[1:m]; right = ord[m+1:end]; sv = X[sd, ord[m]]
        end
        L = build(left); R = build(right)
        push!(nodes, Any[Int[], sd, sv, L, R, lo, hi])
        return length(nodes)
    end
    root = build(collect(1:n))
    function knn!(q::AbstractVector{<:Real}, selfidx::Int, K::Int,
                  D::Vector{Float64}, I::Vector{Int}, node::Int)
        nd = nodes[node]
        idxs, sd, sv, L, R, lo, hi = nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], nd[7]
        worst = isempty(D) ? Inf : maximum(D)
        lb = isinf(p) ? lb_inf(q, lo, hi) : lb_lp(q, lo, hi, p)
        if length(D) == K && lb ≥ worst
            return
        end
        if sd == 0
            @inbounds for j in idxs
                j == selfidx && continue
                xj = @view X[:, j]
                dj = isinf(p) ? cheb(q, xj) : lp(q, xj, p)
                dj = ifelse(iszero(dj), eps(Float64), dj)
                if length(D) < K
                    push!(D, dj); push!(I, j)
                elseif dj < worst
                    t = findmax(D)[2]; D[t] = dj; I[t] = j
                end
                worst = length(D) == K ? maximum(D) : Inf
            end
            return
        end
        near = (q[sd] ≤ sv) ? L : R
        far  = (q[sd] ≤ sv) ? R : L
        knn!(q, selfidx, K, D, I, near)
        worst = length(D) == K ? maximum(D) : Inf
        if length(D) < K || abs(q[sd] - sv) < worst
            knn!(q, selfidx, K, D, I, far)
        end
    end
    ρ = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        D = Float64[]; I = Int[]
        q = @view X[:, j]
        knn!(q, j, k, D, I, root)
        ρ[j] = maximum(D)
    end
    ρ .= max.(ρ, eps(Float64))

    #KL: H = -ψ(k)+ψ(n)+log c_{d,p} + (d/n)∑log ρ  ; for L∞, we absorb log c_{d,∞}=d log 2
    H = -SpecialFunctions.digamma(k) + SpecialFunctions.digamma(n)
    if isinf(p)
        H += (d / n) * sum(log.(2 .* ρ))
    else
        logcd = d*log(2*SpecialFunctions.gamma(1 + 1/p)) - SpecialFunctions.loggamma(1 + d/p)
        H += logcd + (d / n) * sum(log.(ρ))
    end
    return H
end

"""
    measure(C::Copula, lower, upper)

Return the probability assigned by `C` to the axis-aligned half-open rectangle
with opposite corners `lower` and `upper`, using CDF inclusion--exclusion.
Bounds are clipped to the unit hypercube; a rectangle with any non-positive
width has measure zero. Both corners must contain one value per copula
dimension.

See also: [`Distributions.cdf`](@ref), [`subsetdims`](@ref), [`Copula`](@ref).
"""
function measure(C::Copula{d}, us,vs) where {d}

    # Computes the value of the cdf at each corner of the hypercube [u,v]
    # To obtain the C-volume of the box.
    # This assumes u[i] < v[i] for all i
    # Based on Computing the {{Volume}} of {\emph{n}} -{{Dimensional Copulas}}, Cherubini & Romagnoli 2009

    # We use a gray code according to the proposal at https://discourse.julialang.org/t/looping-through-binary-numbers/90597/6

    T = promote_type(eltype(us), eltype(vs), Float64)
    u = ntuple(j -> clamp(us[j], 0, 1), d)
    v = ntuple(j -> clamp(vs[j], 0, 1), d)
    any(v .≤ u) && return T(0)
    all(iszero.(u)) && all(isone.(v)) && return T(1)

    eval_pt = collect(u)
    # Inclusion–exclusion: the sign for the corner at u is (-1)^d
    # (for d even it's +1, for d odd it's -1). The Gray-code loop below
    # then applies alternating signs matching (-1)^(d - |ε|) as bits flip.
    sign = isodd(d) ? -one(T) : one(T)
    r = sign * Distributions.cdf(C, eval_pt)
    graycode = 0    # use a gray code to flip one element at a time
    which = fill(false, d) # false/true to use u/v for each component (so false here)
    for s = 1:(1<<d)-1
        graycode′ = s ⊻ (s >> 1)
        graycomp = trailing_zeros(graycode ⊻ graycode′) + 1
        graycode = graycode′
        eval_pt[graycomp] = (which[graycomp] = !which[graycomp]) ? v[graycomp] : u[graycomp]
        sign *= -1
        r += sign * Distributions.cdf(C, eval_pt)
    end
    return max(r,0)
end
function measure(C::Copula{2}, us, vs)
    T = promote_type(eltype(us), eltype(vs), Float64)
    u1 = clamp(T(us[1]), 0, 1)
    u2 = clamp(T(us[2]), 0, 1)
    v1 = clamp(T(vs[1]), 0, 1)
    v2 = clamp(T(vs[2]), 0, 1)
    (v1 <= u1 || v2 <= u2) && return zero(T)
    u1 == 0 && u2 == 0 && v1 == 1 && v2 == 1 && return one(T)
    c11 = Distributions.cdf(C, [v1, v2])
    c10 = Distributions.cdf(C, [v1, u2])
    c01 = Distributions.cdf(C, [u1, v2])
    c00 = Distributions.cdf(C, [u1, u2])
    r = c11 - c10 - c01 + c00
    return max(r, T(0))
end
