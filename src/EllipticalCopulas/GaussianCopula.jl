"""
    GaussianCopula{d, MT}

Fields:
- `Σ::MT` — correlation matrix (the constructor coerces the input to a correlation matrix).

Constructors

    GaussianCopula(Σ)
    GaussianCopula(d, ρ)
    GaussianCopula(d::Integer, ρ::Real)

Where `Σ` is a (symmetric) covariance or correlation matrix. The two-argument
form with `(d, ρ)` builds the equicorrelation matrix with ones on the diagonal
and constant off-diagonal correlation `ρ`:

```julia
Σ = fill(ρ, d, d); Σ[diagind(Σ)] .= 1
C = GaussianCopula(d, ρ)            # == GaussianCopula(Σ)
```

Validity domain (equicorrelated PD matrix): `-1/(d-1) < ρ < 1`. The boundary
`ρ = -1/(d-1)` is singular and rejected. If `ρ == 0`, this returns
`IndependentCopula(d)` (same fast-path as when passing a diagonal matrix).

The Gaussian copula is the copula of a multivariate normal distribution. It is defined by

```math
C(\\mathbf{x}; \\boldsymbol{\\Sigma}) = F_{\\Sigma}(F_{\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\Sigma}`` is the cdf of a centered multivariate normal with covariance/correlation ``\\Sigma`` and ``F_{\\Sigma,i}`` its i-th marginal cdf.

Example usage:
```julia
C = GaussianCopula(Σ)
u = rand(C, 1000)
pdf(C, u); cdf(C, u)
Ĉ = fit(GaussianCopula, u)
```

Special case:
- If `isdiag(Σ)`, the constructor returns `IndependentCopula(d)`.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GaussianCopula{d,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function GaussianCopula(Σ)
        if LinearAlgebra.isdiag(Σ)
            return IndependentCopula(size(Σ,1))
        end
        make_cor!(Σ)
        N(GaussianCopula)(Σ)
        return new{size(Σ,1),typeof(Σ)}(Σ)
    end
end

# Equicorrelation convenience constructor
function GaussianCopula(d::Int, ρ::Real)
    d < 2 && throw(ArgumentError("Use a bivariate or higher dimension (d ≥ 2) or pass a 1×1 matrix."))
    # Positive definiteness condition for equicorrelation matrix
    lower = -1/(d-1)
    ρ ≤ lower && throw(ArgumentError("Equicorrelation value ρ=$(ρ) not in (-1/(d-1), 1). For d=$d the lower open bound is $(lower)."))
    ρ ≥ 1 && throw(ArgumentError("Equicorrelation value ρ must be < 1."))
    if ρ == 0
        return IndependentCopula(d)
    end
    Σ = fill(float(ρ), d, d)
    @inbounds for i in 1:d
        Σ[i,i] = one(ρ)
    end
    return GaussianCopula(Σ)
end
GaussianCopula(d::Int, Σ::AbstractMatrix) = GaussianCopula(Σ) 
GaussianCopula{D, MT}(d::Int, Σ::AbstractMatrix) where {D, MT} = GaussianCopula(Σ) 
GaussianCopula{D, MT}(d::Int, ρ::Real) where {D, MT} = GaussianCopula(d, ρ) 

U(::Type{T}) where T<: GaussianCopula = Distributions.Normal()
N(::Type{T}) where T<: GaussianCopula = Distributions.MvNormal
#function _cdf(C::CT,u) where {CT<:GaussianCopula}
#    x = StatsBase.quantile.(Distributions.Normal(), u)
#    d = length(C)
#    return MvNormalCDF.mvnormcdf(C.Σ, fill(-Inf, d), x)[1]
#end

function rosenblatt(C::GaussianCopula, u::AbstractMatrix{<:Real})
    return Distributions.cdf.(Distributions.Normal(), inv(LinearAlgebra.cholesky(C.Σ).L) * Distributions.quantile.(Distributions.Normal(), u))
end

function inverse_rosenblatt(C::GaussianCopula, s::AbstractMatrix{<:Real})
    return Distributions.cdf.(Distributions.Normal(), LinearAlgebra.cholesky(C.Σ).L * Distributions.quantile.(Distributions.Normal(), s))
end

# Kendall tau of bivariate gaussian:
# Theorem 3.1 in Fang, Fang, & Kotz, The Meta-elliptical Distributions with Given Marginals Journal of Multivariate Analysis, Elsevier, 2002, 82, 1–16
τ(C::GaussianCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π
ρ(C::GaussianCopula{2,MT}) where MT = 6*asin(C.Σ[1,2]/2)/π

# Conditioning and subsetting fast paths colocated with the type
function DistortionFromCop(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,MT,p}
    ist = Tuple(setdiff(1:D, js))
    @assert i in ist
    J = collect(js)
    zⱼ = Distributions.quantile.(Distributions.Normal(), collect(uⱼₛ))
    if length(J) == 1 # if we condition on only one variable
        μz = C.Σ[i, J[1]] * zⱼ[1]
        σz = sqrt(1 - C.Σ[i, J[1]]^2)
    else
        Reg = C.Σ[i:i, J] * inv(C.Σ[J, J])
        μz = (Reg * zⱼ)[1]
        σz = sqrt(1 - (Reg * C.Σ[J, i:i])[1])
    end
    return GaussianDistortion(float(μz), float(σz))
end
function ConditionalCopula(C::GaussianCopula{D,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}) where {D,MT,p}
    @assert 0 < p < D-1
    J = collect(Int, js)
    I = collect(setdiff(1:D, J))
    Σcond = C.Σ[I, I] - C.Σ[I, J] * inv(C.Σ[J, J]) * C.Σ[J, I]
    return GaussianCopula(Σcond)
end

# Subsetting colocated
SubsetCopula(C::GaussianCopula, dims::NTuple{p, Int}) where p = GaussianCopula(C.Σ[collect(dims),collect(dims)])


# Fitting collocated
StatsBase.dof(C::Copulas.GaussianCopula)    = (p = length(C); p*(p-1) ÷ 2)
Distributions.params(C::GaussianCopula) = (; Σ = C.Σ)
_example(::Type{<:GaussianCopula}, d::Int) = GaussianCopula(d, 0.2)
function _unbound_params(::Type{<:GaussianCopula}, d::Int, θ::NamedTuple)
    return _unbound_corr_params(d, θ.Σ)
end
function _rebound_params(::Type{<:GaussianCopula}, d::Int, α::AbstractVector{T}) where {T}
    return (; Σ = _rebound_corr_params(d, α))
end
function _fit(CT::Type{<:GaussianCopula}, u, ::Val{:mle})
    dd = Distributions.fit(N(CT), StatsBase.quantile.(U(CT),u))
    Σ = Matrix(dd.Σ)
    return GaussianCopula(Σ), (; θ̂ = (; Σ = Σ))
end
_available_fitting_methods(::Type{<:GaussianCopula}, d) = (:mle, :itau, :irho, :ibeta)


function _cdf_base(C::CT, u; abseps=1e-4, releps=1e-4, maxpts=50_000, m0=1028, r=2, rng=Random.default_rng()) where {CT<:GaussianCopula}
    T = eltype(u) 
    d = length(u)
    x = StatsFuns.norminvcdf.(u)
    Σ = C.Σ

    # Standardize to correlation and prepare b* limits
    σ = sqrt.(LinearAlgebra.diag(Σ))
    R = Σ ./ (σ * σ')
    bstar0 = x ./ σ
    widths = StatsFuns.normcdf.(bstar0)         # Φ(b*_j) because a=-Inf

    # "Short interval" rearrangement
    P = sortperm(widths)                         # ascendent
    R1 = R[P, P]
    bstar1 = bstar0[P]

    # Cholesky with pivoting + permutation composition
    F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(R1), LinearAlgebra.RowMaximum(); check=false)
    L = Matrix(F.L)
    pinv = invperm(F.p)
    bstar = bstar1[pinv]

    # Adaptive stopping + replications
    total_used = 0
    m = m0
    best_mean = NaN
    best_se = Inf

    y1 = Vector{T}(undef, d)
    y2 = Vector{T}(undef, d)

    while true
        reps = min(r, max(2, fld(maxpts - total_used, max(m,1))))
        reps == 0 && break
        means = Vector{T}(undef, reps)

        for rep in 1:reps
            sob = Sobol.SobolSeq(d)
            shift = rand(rng, d)                 # Cranley–Patterson shift in sobol... for quase montecarlo
            acc = 0.0

            @inbounds for _ in 1:m
                uvec = (Sobol.next!(sob) .+ shift) .% 1.0

                logp1 = 0.0; logp2 = 0.0
                alive1 = true; alive2 = true

                for j in 1:d
                    if alive1
                        μ1 = 0.0
                        @simd for k in 1:(j-1)
                            μ1 += L[j,k] * y1[k]
                        end
                        tj1 = (bstar[j] - μ1) / L[j,j]
                        β1  = StatsFuns.normcdf(tj1)
                        if β1 <= eps()
                            alive1 = false; logp1 = -Inf
                        else
                            t1 = clamp(uvec[j]*β1, floatmin(Float64), 1 - eps(Float64))
                            y1[j] = StatsFuns.norminvcdf(t1)
                            logp1 += log(β1)
                        end
                    end

                    if alive2
                        μ2 = 0.0
                        @simd for k in 1:(j-1)
                            μ2 += L[j,k] * y2[k]
                        end
                        tj2 = (bstar[j] - μ2) / L[j,j]
                        β2  = StatsFuns.normcdf(tj2)
                        if β2 <= eps()
                            alive2 = false; logp2 = -Inf
                        else
                            t2 = clamp((1.0 - uvec[j])*β2, floatmin(Float64), 1 - eps())
                            y2[j] = StatsFuns.norminvcdf(t2)
                            logp2 += log(β2)
                        end
                    end

                    if !alive1 && !alive2
                        break
                    end
                end

                if isfinite(logp1) || isfinite(logp2)
                    M = max(logp1, logp2)
                    acc += 0.5 * exp(M) * ((isfinite(logp1) ? exp(logp1 - M) : 0.0) +
                                            (isfinite(logp2) ? exp(logp2 - M) : 0.0))
                end
            end

            means[rep] = acc / m
        end

        μ = sum(means) / length(means)
        v = 0.0
        @inbounds for z in means
            v += (z - μ)^2
        end
        se = sqrt((v / max(length(means)-1,1)) / length(means))

        total_used += reps*m
        best_mean = μ; best_se = se

        if se ≤ max(abseps, releps*abs(μ)) || total_used ≥ maxpts
            break
        end
        m = min(2m, maxpts - total_used)
        m ≤ 0 && break
    end

    return best_mean
end
function _cdf(C::GaussianCopula, u; fast::Bool=true, kwargs...)
    if fast
        return _cdf_base(C, u;
            abseps=1e-4, releps=1e-4, maxpts=50_000, m0=1028, r=2, kwargs...)
    else
        return _cdf_base(C, u;
            abseps=1e-6, releps=1e-6, maxpts=1_000_000, m0=2048, r=8, kwargs...)
    end
end