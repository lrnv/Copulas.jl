"""
    TCopula{d, df, MT}

Fields:
- `df::Int` — degrees of freedom
- `Σ::MT` — correlation matrix

Constructor

    TCopula(df, Σ)

The Student t copula is the copula of a multivariate Student t distribution. It is defined by

```math
C(\\mathbf{x}; \\nu, \\boldsymbol{\\Sigma}) = F_{\\nu,\\Sigma}(F_{\\nu,\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\nu,\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\nu,\\Sigma}`` is the cdf of a centered multivariate t with correlation ``\\Sigma`` and ``\\nu`` degrees of freedom.

Example usage:
```julia
C = TCopula(2, Σ)
u = rand(C, 1000)
pdf(C, u); cdf(C, u)
Ĉ = fit(TCopula, u)
```

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct TCopula{d,df,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function TCopula(df,Σ)
        make_cor!(Σ)
        N(TCopula{size(Σ,1),df,typeof(Σ)})(Σ)
        return new{size(Σ,1),df,typeof(Σ)}(Σ)
    end
end
TCopula(d::Int, ν::Real, Σ::AbstractMatrix) = TCopula(ν, Σ)
TCopula{D,df,MT}(d::Int, ν::Real, Σ::AbstractMatrix)  where {D,df,MT} = TCopula(ν, Σ)



U(::Type{TCopula{d,df,MT}}) where {d,df,MT} = Distributions.TDist(df)
N(::Type{TCopula{d,df,MT}}) where {d,df,MT} = function(Σ)
    Distributions.MvTDist(df,Σ)
end

# Kendall tau of bivariate student: 
# Lindskog, F., McNeil, A., & Schmock, U. (2003). Kendall’s tau for elliptical distributions. In Credit risk: Measurement, evaluation and management (pp. 149-156). Heidelberg: Physica-Verlag HD.
τ(C::TCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π 
function τ(C::TCopula{d,MT}) where {d, MT}
    T = (2/π) .* asin.(C.Σ)
    @inbounds for i in 1:d
        T[i,i] = 1.0
    end
    return LinearAlgebra.Symmetric(T, :U)
end
##############################
function ρ(C::TCopula{2,ν,MT}) where {ν,MT}
    ρ_ = C.Σ[1,2]
    rtol = 1e-10
    #  Normalization constant off_{Ṽ}
    Cν = 2 * SpecialFunctions.gamma(ν)^2 * SpecialFunctions.gamma(3ν/2) / (SpecialFunctions.gamma(ν/2)^3 * SpecialFunctions.gamma(2ν))
    f(v) = begin
        # if we use HypergeometricFunctions.jl we can make:
        F = HypergeometricFunctions.pFq((ν, ν), (2ν,), 1 - v^2)
        # and if not... The implemented functions work well and in particular are quite fast.
        # F = Copulas._Gauss2F1_hybrid(ν, 1 - v^2)
        return asin(ρ_ * v) * Cν * v^(ν - 1) * (1 - v^2)^(ν/2 - 1) * F
    end
    try
        val, _ = QuadGK.quadgk(f, 0.0, 1.0; rtol=rtol)
        return (6/π) * val
    catch err
        if ν > 20
            # asymptotic fallback (equivalent to normal copula)
            ρ_norm = (6/π) * asin(ρ_/2)
            return ρ_norm
        else
            rethrow(err)
        end
    end
end
# Conditioning colocated
function DistortionFromCop(C::TCopula{D,ν,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p,D,ν,MT}
    Σ = C.Σ; jst = js; ist = Tuple(setdiff(1:D, jst)); @assert i in ist
    Jv = collect(jst); zJ = Distributions.quantile.(Distributions.TDist(ν), collect(uⱼₛ))
    ΣJJ = Σ[Jv, Jv]; RiJ = Σ[i, Jv]; RJi = Σ[Jv, i]
    if length(Jv) == 1
        r = RiJ[1]; μz = r * zJ[1]; σ0² = 1 - r^2; δ = zJ[1]^2
    else
        L = LinearAlgebra.cholesky(Symmetric(ΣJJ))
        μz = dot(RiJ, (L' \ (L \ zJ)))
        σ0² = 1 - dot(RiJ, (L' \ (L \ RJi)))
        y = L \ zJ; δ = dot(y, y)
    end
    νp = ν + length(Jv); σz = sqrt(max(σ0², zero(σ0²))) * sqrt((ν + δ) / νp)
    return StudentDistortion(float(μz), float(σz), Int(ν), Int(νp))
end
function ConditionalCopula(C::TCopula{D,df,MT}, js, uⱼₛ) where {D,df,MT}
    p = length(js); J = collect(Int, js); I = collect(setdiff(1:D, J)); Σ = C.Σ
    if p == 1
        Σcond = Σ[I, I] - Σ[I, J] * (Σ[J, I] / Σ[J, J])
    else
        L = LinearAlgebra.cholesky(Symmetric(Σ[J, J]))
        Σcond = Σ[I, I] - Σ[I, J] * (L' \ (L \ Σ[J, I]))
    end
    σ = sqrt.(LinearAlgebra.diag(Σcond))
    R_cond = Matrix(Σcond ./ (σ * σ'))
    return TCopula(df + p, R_cond)
end
# Subsetting colocated
SubsetCopula(C::TCopula{d,df,MT}, dims::NTuple{p, Int}) where {d,df,MT,p} = TCopula(df, C.Σ[collect(dims),collect(dims)])

# Fitting collocated
StatsBase.dof(C::Copulas.TCopula)           = (p = length(C); p*(p-1) ÷ 2 + 1)
function Distributions.params(C::TCopula{d,df,MT}) where {d,df,MT}
    return (; ν = df, Σ = C.Σ)
end
_example(::Type{<:TCopula}, d::Int) = TCopula(5.0, Matrix(LinearAlgebra.I, d, d) .+ 0.2 .* (ones(d, d) .- Matrix(LinearAlgebra.I, d, d)))
function _unbound_params(::Type{<:TCopula}, d::Int, θ::NamedTuple)
    α = _unbound_corr_params(d, θ.Σ)
    return vcat(log(θ.ν), α)
end
function _rebound_params(::Type{<:TCopula}, d::Int, α::AbstractVector{T}) where {T}
    ν = exp(α[1])
    Σ = _rebound_corr_params(d, @view α[2:end])
    return (; ν = ν, Σ = Σ)
end

_available_fitting_methods(::Type{<:TCopula}, d) = (:mle,)

# t-ortant (copulates t with ν g.l.)
function qmc_orthant_t!(R::AbstractMatrix{T}, b::AbstractVector{T}, ν::Integer; m::Integer = 10_000, r::Integer = 12,
    rng::Random.AbstractRNG = Random.default_rng()) where T
    # ¡muta R y b!
    (ch, bs) = _chlrdr_orthant!(R, b)

    # extra Richtmyer root for the radial dimension (χ²)
    qχ  = richtmyer_roots(T, length(b) + 1)[end]
    chi = Distributions.Chisq(ν)

    # scale generator w[k] = √(ν / S_k), S_k ~ χ²_ν (quasi-random)
    fill_w! = function (w::AbstractVector{T}, _j::Int, nv::Int, δ::T, rng_local)
        xrχ = rand(rng_local, T)
        @inbounds @simd for k in 1:nv
            t = k*qχ + xrχ; t -= floor(t)
            u = clamp(t, δ, one(T)-δ)                    # u ∈ (δ, 1-δ)
            s = T(Distributions.quantile(chi, Real(u)))            # quantile χ²_ν
            w[k] = sqrt(T(ν) / s)                       # radial scale
        end
        nothing
    end

    return qmc_orthant_core!(ch, bs; m=m, r=r, rng=rng, fill_w! = fill_w!)
end

function Distributions.cdf(C::TCopula{d,df,MT}, u::AbstractVector; m::Integer = 2000*(d+1), r::Int = 12, rng = Random.default_rng()) where {d,df,MT}
    b = Distributions.quantile.(Distributions.TDist(df), u)
    Tb = eltype(b)
    Σ_promoted = Tb.(copy(C.Σ))
    p, _ = qmc_orthant_t!(Σ_promoted, b, df; m=m, r=r, rng=rng)
    return p
end