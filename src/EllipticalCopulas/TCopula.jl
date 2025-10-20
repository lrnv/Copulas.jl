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
ρ(C::TCopula{2,df,MT}) where {df,MT} = rhoS_t(df, C.Σ[1,2])
function ρ(C::TCopula{d,df,MT}) where {d,df,MT}
    T = Matrix{Float64}(I, d, d)
    @inbounds for j in 1:d-1, i in j+1:d
        T[i,j] = T[j,i] = rhoS_t(df, C.Σ[i,j])
    end
    LinearAlgebra.Symmetric(T, :U)
end
##############################
function rhoS_t(ν::Real, ρ::Real; rtol::Real=1e-10)
    ν ≤ 0 && throw(ArgumentError("ν>0 requerido"))
    ρ_ = clamp(float(ρ), -prevfloat(1.0), prevfloat(1.0))
    #  Normalization constant off_{Ṽ}
    Cν = 2 * SpecialFunctions.gamma(ν)^2 * SpecialFunctions.gamma(3ν/2) / (SpecialFunctions.gamma(ν/2)^3 * SpecialFunctions.gamma(2ν))
    f(v) = begin
        # if we use HypergeometricFunctions.jl we can make:
        # F = HypergeometricFunctions.pFq((ν, ν), (2ν,), 1 - v^2)
        # and if not... The implemented functions work well and in particular are quite fast.
        F = Copulas._Gauss2F1_hybrid(ν, 1 - v^2)
        asin(ρ_ * v) * Cν * v^(ν - 1) * (1 - v^2)^(ν/2 - 1) * F
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


###############
##############################

"""
    _Gauss_hypergeometric(ν, z; rtol=1e-12, maxiters=50_000, zsplit=0.7)

Evaluates ₂F₁(ν, ν; 2ν; z) **sin** HypergeometricFunctions.jl.

- For `z ≤ zsplit` use the **Gauss series**: 
    sum_{n≥0} ((ν)_n)^2 / ((2ν)_n) * z^n / n!
- For `z > zsplit` use the **analytic continuation** on `x = 1 - z` (case `c=a+b`): 
    ₂F₁(ν,ν;2ν;z) = Γ(2ν)/Γ(ν)^2 * ∑_{n≥0} ((ν)_n)^2/(n!)^2 * x^n * [2ψ(n+1) - 2ψ(ν+n) - log(x)]

Returns `Float64`. Raises `DomainError` if `z ≥ 1` (logarithmic singularity).
"""
function _Gauss_hypergeometric(ν::Real, z::Real; rtol::Real=1e-12, maxiters::Int=50_000, zsplit::Real=0.7)
    ν ≤ 0 && throw(ArgumentError("ν>0 requerido"))
    z == 0 && return 1.0
    z ≥ 1 && throw(DomainError(z, "₂F₁(ν,ν;2ν;z) diverge en z≥1 (caso c=a+b)."))
    z1 = clamp(float(z), -1 + eps(), 1 - eps())
    if z1 ≤ zsplit
        # --- Serie Gauss: t_{n+1} = t_n * ((ν+n)^2 / ((2ν+n)*(n+1))) * z ---
        n = 0
        t = 1.0
        s = t
        while n < maxiters
            n += 1
            t *= ((ν + n - 1)^2 / ((2ν + n - 1) * n)) * z1
            s_old = s
            s += t
            if abs(s - s_old) ≤ rtol * max(1.0, abs(s))
                return s
            end
        end
        error("_Gauss_hypergeometric: does not converge (series) on maxiters=$maxiters; z=$z1, ν=$ν")
    else
        # --- Analytical continuation at x = 1 - z (case c=a+b⇒ log term) ---
        x   = 1.0 - z1
        lnx = log(x)
        pref = SpecialFunctions.gamma(2ν) / (SpecialFunctions.gamma(ν)^2)
        # Serie: s = Σ t_n * B_n,  t_0=1,  t_{n+1} = t_n * ((ν+n)^2/(n+1)^2) * x
        #        B_n = 2ψ(n+1) - 2ψ(ν+n) - log(x)
        n = 0
        t = 1.0
        s = 0.0
        ccor = 0.0  # Kahan/Neumaier compensation
        while n < maxiters
            Bn = 2*SpecialFunctions.digamma(n + 1) - 2*SpecialFunctions.digamma(ν + n) - lnx
            add = t * Bn
            y = add - ccor
            s_new = s + y
            ccor = (s_new - s) - y
            s = s_new
            n += 1
            t *= ((ν + n - 1)^2 / (n^2)) * x
            if abs(add) ≤ rtol * max(1.0, abs(s))
                return pref * s
            end
        end
        error("_Gauss_hypergeometric: does not converge (analytic con.) in maxiters=$maxiters; z=$z1, ν=$ν")
    end
end

"""
    _Gauss_Euler(ν, z; rtol=1e-10)

Evaluate ₂F₁(ν, ν; 2ν; z) using the **Euler integral**: 

B(ν,ν) * ₂F₁(ν,ν;2ν; z) = ∫₀¹ t^{ν-1} (1-t)^{ν-1} (1 - z t)^{-ν} dt,
where B(ν,ν) = Γ(ν)Γ(ν)/Γ(2ν). Implemented with `quadgk`.

- Valid and stable for `z < 1`. For `z ≥ 1` the integrand becomes non-integrable.
"""
function _Gauss_Euler(ν::Real, z::Real; rtol::Real=1e-12, k::Int=4)
    ν ≤ 0 && throw(ArgumentError("ν>0 required"))
    (z < 0 || z >= 1) && throw(DomainError(z, "Euler requires 0 ≤ z < 1."))
    zf = float(z)
    x  = 1 - zf
    λ  = zf / x
    pref = SpecialFunctions.gamma(2ν) / (SpecialFunctions.gamma(ν)^2) * x^(-ν)   # 1/B(ν,ν) * (1-z)^(-ν)
    # "Central" part: [0, t0]
    t0  = 1 - min(0.25, max(5x, 1e-6))   # adaptive threshold
    f1(t) = t^(ν-1) * (1 - t)^(ν-1) * (1 + λ*(1 - t))^(-ν)  # ya escalado
    I1, _ = QuadGK.quadgk(f1, 0.0, t0; rtol=rtol)
    # tail: [t0, 1] with u=1-t = u_max*y^k
    umax = 1 - t0
    g(y) = begin
        u   = umax * y^k
        t   = 1 - u
        jac = umax * k * y^(k-1)
        jac * t^(ν-1) * u^(ν-1) * (1 + λ*u)^(-ν)
    end
    I2, _ = QuadGK.quadgk(g, 0.0, 1.0; rtol=rtol)
    return pref * (I1 + I2)
end

"""
    _Gauss2F1_hybrid(ν, z; rtol_series=1e-12, rtol_euler=1e-12,
                      zsplit=0.7, thresh=1.0)

Evaluate ₂F₁(ν,ν;2ν;z) by choosing the most stable method:
- zz split → Gaussian series.
- z > zsplit → if ν^2*(1-z) ≥ thresh → Euler; otherwise → analytic continuation.
"""
function _Gauss2F1_hybrid(ν::Real, z::Real;
                          rtol_series::Real=1e-12, rtol_euler::Real=1e-12,
                          zsplit::Real=0.7, thresh::Real=1.0, k::Int=4)
    z < 0 && throw(DomainError(z, "requires 0 ≤ z < 1."))
    z == 0 && return 1.0
    z >= 1 && throw(DomainError(z, "₂F₁ diverges at z≥1 (c=a+b)."))
    if z <= zsplit
        return Copulas._Gauss_hypergeometric(ν, z; rtol=rtol_series, zsplit=zsplit)
    else
        x = 1 - z
        return (ν^2 * x >= thresh) ? Copulas._Gauss_Euler(ν, z; rtol=rtol_euler, k=k) : Copulas._Gauss_hypergeometric(ν, z; rtol=rtol_series, zsplit=zsplit)
    end
end
# t-ortant (copulates t with ν g.l.)
function qmc_orthant_t!(R::AbstractMatrix{T}, b::AbstractVector{T}, ν::Integer; m::Integer = 10_000, r::Integer = 12,
    rng::Random.AbstractRNG = Random.default_rng()) where {T<:AbstractFloat}

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
            s = T(Distributions.quantile(chi, Float64(u)))            # quantile χ²_ν
            w[k] = sqrt(T(ν) / s)                       # radial scale
        end
        nothing
    end

    return qmc_orthant_core!(ch, bs; m=m, r=r, rng=rng, fill_w! = fill_w!)
end

