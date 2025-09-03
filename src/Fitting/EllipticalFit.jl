# ========== utilidades SPD / correlación ==========
@inline _as_pxn_(U::AbstractMatrix) = _as_pxn(size(U, 1), U)
@inline function _symmetrize!(S::AbstractMatrix{<:Real})
    @inbounds S .= 0.5 .* (S .+ S')
    return S
end

@inline function _clamp_offdiag!(S::AbstractMatrix{<:Real}, lo::Float64, hi::Float64)
    @inbounds for i in 1:size(S,1), j in 1:size(S,2)
        if i != j
            v = S[i,j]
            S[i,j] = v < lo ? lo : (v > hi ? hi : v)
        end
    end
    return S
end

# Proyección a matriz de correlación SPD (simétrica, diag=1, λᵢ≥eps)
function _proj_corr!(Σ::Matrix{Float64}; eps::Float64=1e-10)
    _symmetrize!(Σ)
    # Espectral con "floor" de autovalores
    vals, vecs = LinearAlgebra.eigen(LinearAlgebra.Symmetric(Σ))
    vals .= max.(vals, eps)
    Σ .= vecs * LinearAlgebra.Diagonal(vals) * vecs'
    _symmetrize!(Σ)
    # Normaliza a correlación (diag = 1)
    d = sqrt.(LinearAlgebra.diag(Σ))
    @inbounds for i in 1:length(d)
        d[i] = d[i] > 0 ? d[i] : sqrt(eps)
    end
    Dinv = LinearAlgebra.Diagonal(@. 1.0 / d)
    Σ .= Dinv * Σ * Dinv
    _symmetrize!(Σ)
    # Clampea sólo fuera de la diagonal
    _clamp_offdiag!(Σ, -0.999999, 0.999999)
    @inbounds for i in 1:size(Σ,1); Σ[i,i] = 1.0; end
    return Σ
end

# Fuerza matriz de correlación: simétrica, diag=1, SPD + jitter suave
function _make_corr_safe(Σ::AbstractMatrix{<:Real}; eps::Float64=1e-10, jitter::Float64=1e-12)
    S = Matrix{Float64}(Σ)
    _proj_corr!(S; eps=eps)
    if jitter > 0
        @inbounds for i in 1:size(S,1); S[i,i] += jitter; end
        # Re-normaliza y re-proyecta tras el jitter para mantener diag=1 y SPD
        _proj_corr!(S; eps=eps)
    end
    return S
end

# ========== Blomqvist β (par a par) ==========

# β = 4 * P( (U-1/2)(V-1/2) ≥ 0 ) - 1
function _beta_matrix(Up::AbstractMatrix{<:Real})
    d, n = size(Up)
    B = Matrix{Float64}(LinearAlgebra.I, d, d)
    @inbounds for i in 1:d-1, j in i+1:d
        u = @view Up[i, :]
        v = @view Up[j, :]
        same = 0
        @inbounds @simd for k in 1:n
            same += ((u[k]-0.5)*(v[k]-0.5) ≥ 0) ? 1 : 0
        end
        β = 4.0 * (same / n) - 1.0
        B[i,j] = β
        B[j,i] = β
    end
    return B
end

# ========== Correlaciones a partir de τ / ρ_S / β ==========

function _gaussian_corr_from_tau(Up::AbstractMatrix)
    τ = StatsBase.corkendall(Up')                   # variables en columnas
    ρ = @. sin((π/2) * τ)
    return _make_corr_safe(ρ)
end

function _gaussian_corr_from_spearman(Up::AbstractMatrix)
    ρS = StatsBase.corspearman(Up')
    ρ  = @. 2 * sin((π/6) * ρS)
    return _make_corr_safe(ρ)
end

function _gaussian_corr_from_beta(Up::AbstractMatrix)
    β = _beta_matrix(Up)
    ρ = @. sin((π/2) * β)
    return _make_corr_safe(Matrix{Float64}(ρ))
end

# ========== Correlación por normal/t-scores (para MLE/pseudo-MLE) ==========

function _corr_from_scores(Z::AbstractMatrix{<:Real}; eps::Float64=1e-12)
    d, n = size(Z)
    μ = Vector{Float64}(undef, d)
    @inbounds for i in 1:d
        μ[i] = sum(@view Z[i, :]) / n
    end
    # Covarianza muestral simétrica
    S = Matrix{Float64}(undef, d, d); fill!(S, 0.0)
    @inbounds for j in 1:n
        zj = @view Z[:, j]
        for a in 1:d
            za = zj[a] - μ[a]
            @inbounds for b in a:d
                zb = zj[b] - μ[b]
                S[a,b] += za*zb
            end
        end
    end
    S ./= (n - 1)
    @inbounds for a in 1:d-1, b in a+1:d; S[b,a] = S[a,b]; end
    # A correlación y proyección segura
    D  = sqrt.(max.(LinearAlgebra.diag(S), eps))
    Σ  = S ./ (D * D')
    return _make_corr_safe(Σ)
end

# ========== GaussianCopula: ajustes rank-based y pseudo-MLE ==========

function fit_gaussian_itau(::Type{CT}, U::AbstractMatrix) where {CT<:GaussianCopula}
    Up = _as_pxn_(U)
    Σ  = _gaussian_corr_from_tau(Up)
    return CT(Σ)
end

function fit_gaussian_irho(::Type{CT}, U::AbstractMatrix) where {CT<:GaussianCopula}
    Up = _as_pxn_(U)
    Σ  = _gaussian_corr_from_spearman(Up)
    return CT(Σ)
end

function fit_gaussian_ibeta(::Type{CT}, U::AbstractMatrix) where {CT<:GaussianCopula}
    Up = _as_pxn_(U)
    Σ  = _gaussian_corr_from_beta(Up)
    return CT(Σ)
end

"""
    fit_gaussian_mle(::Type{CT}, U; clip=1e-10) where {CT<:GaussianCopula}

Pseudo-MLE: correlación de los normal scores + proyección SPD.
"""
function fit_gaussian_mle(::Type{CT}, U::AbstractMatrix; clip::Float64=1e-10) where {CT<:GaussianCopula}
    Up = _as_pxn_(U)
    Z  = Distributions.quantile.(Distributions.Normal(), clamp.(Up, clip, 1.0-clip))
    Σ  = _corr_from_scores(Z)
    return CT(Σ)
end

import Distributions
import StatsBase
import LinearAlgebra
import Optim
using Copulas: TCopula

# ===================== TCopula: Σ fijo por τ, MLE en ν =====================

# NLL de la t-cópula con Σ fijo; clipea U para evitar -Inf en bordes
@inline function _t_nll(U::AbstractMatrix, Σ::Matrix{Float64}, ν::Float64;
                        clip::Float64=1e-10)
    Up  = _as_pxn_(U)
    Upc = clamp.(Up, clip, 1.0 - clip)
    C = try
        TCopula(ν, Σ)
    catch
        return Inf
    end
    ll = Distributions.loglikelihood(C, Upc)
    return isfinite(ll) ? -ll : Inf
end

# Estima Σ a partir de τ de Kendall y lo vuelve correlación SPD
@inline function _corr_from_kendall(Up::AbstractMatrix)
    τ = StatsBase.corkendall(Up')             # d×d
    ρ = @. sin((π/2) * τ)                     # τ ↦ ρ
    return _make_corr_safe(Matrix{Float64}(ρ))  # simetriza, SPD, diag=1
end

"""
    fit_t_mle(::Type{CT}, U; ν_lo=2.05, ν_hi=20.0, tol=1e-6, clip=1e-10)
    where {CT<:TCopula}

Ajuste exclusivo t-cópula con Σ **fijo** por τ de Kendall; se optimiza **sólo** ν (Brent).
- Σ := sin(π/2·τ(U)) proyectada a correlación SPD.
- ν̂ := argmin_ν NLL(U | Σ fijo).
"""
function fit_t_mle(::Type{CT}, U::AbstractMatrix;
                   ν_lo::Float64=2.05, ν_hi::Float64=20.0,
                   tol::Real=1e-6, clip::Float64=1e-10) where {CT<:TCopula}
    Up = _as_pxn_(U)
    Σ  = _corr_from_kendall(Up)   # Σ fijo por τ (SPD & diag=1)
    fν = x -> _t_nll(U, Σ, x; clip=clip)

    res = Optim.optimize(fν, ν_lo, ν_hi, Optim.Brent();
                         rel_tol=tol, abs_tol=1e-10, iterations=400)
    ν̂ = clamp(Optim.minimizer(res), nextfloat(ν_lo), prevfloat(ν_hi))
    return TCopula(ν̂, Σ)
end
