# ====================== src/Fitting/ArchimedeanFit.jl =======================
# 
# Métodos: :itau, :irho, :ibeta, :mde
#
# Prerequisites in the module:
#  - generatorof(::Type{<:ArchimedeanCopula}) -> Generator type TG
#  - τ, τ⁻¹, ρ, ρ⁻¹, β, β⁻¹ (if there are no closed inverses, we use numerical inversion)
#  - Distributions.cdf(C, u) and loglikelihood(C, U)
#  - _as_pxn(C,U) or _as_pxn(p,U) defined in stats.jl
# ---- Construction of copulas from parameters (1- and 2-pair) -----------
# ====================== src/Fitting/ArchimedeanFit.jl =======================

# Devuelve el tipo de generador asociado a la cópula Archimediana CT
# (asumo que ya lo tienes definido en el módulo)
# generatorof(::Type{<:ArchimedeanCopula}) -> TG  (p.ej. GumbelGenerator)

# --- Fábrica robusta que promociona tipos y construye vía el generador ---

# 1 parámetro (Real)
# 1 parámetro
function _make_archimedean(::Type{CT}, d::Integer, θ::Real) where {CT<:ArchimedeanCopula}
    TG = generatorof(CT)                  # p.ej. ClaytonGenerator
    return ArchimedeanCopula(d, TG(float(θ)))
end

# vector -> varargs
function _make_archimedean(::Type{CT}, d::Integer, θ::AbstractVector{<:Real}) where {CT<:ArchimedeanCopula}
    return _make_archimedean(CT, d, θ...)
end

# k parámetros
function _make_archimedean(::Type{CT}, d::Integer, θs::Vararg{Real}) where {CT<:ArchimedeanCopula}
    TG = generatorof(CT)
    T  = promote_type(map(typeof, θs)...)
    return ArchimedeanCopula(d, TG(map(T, θs)...))   # ← sin TG{T}
end


# Número de parámetros del generador
param_length(::Type{GT}) where {GT<:Generator} = fieldcount(GT)

nparams(::Type{CT}) where {CT<:ArchimedeanCopula} =
    param_length(generatorof(CT))
_assert_oneparam(x) = (nparams(x) == 1) || throw(ArgumentError(
   "This method requires a 1-parameter family; " *
"use method=:mle for multi-parameter families."
))

# =============================== :ITAU =======================================
function _fit_itau_1par(::Type{CT}, U::AbstractMatrix; epsτ::Real=1e-10) where {CT<:ArchimedeanCopula}
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_itau(Archimedean) requires d≥2."))
    Up = _as_pxn(d, U)

    τmat = StatsBase.corkendall(Up')  # p×p
    GT   = generatorof(CT)

    θs = Float64[]
    @inbounds for j1 in 1:d-1, j2 in j1+1:d
        τc = clamp(τmat[j1,j2], -1 + epsτ, 1 - epsτ)
        push!(θs, τ⁻¹(GT, τc))
    end
    θ = StatsBase.mean(θs)

    θlo, θhi = θ_bounds(GT, d)
    if isfinite(θlo) && θ ≤ θlo; θ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ ≥ θhi; θ = prevfloat(float(θhi)); end
    return _make_archimedean(CT, d, θ)
end

function fit_itau(::Type{CT}, U::AbstractMatrix; epsτ::Real=1e-10) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    return _fit_itau_1par(CT, U; epsτ=epsτ)
end

# =============================== :IRHO =======================================
function _fit_irho_1par(::Type{CT}, U::AbstractMatrix; epsρ::Real=1e-10) where {CT<:ArchimedeanCopula}
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_irho requires d≥2."))
    Up  = (size(U,1)==d) ? U : permutedims(U)
    ρmat = StatsBase.corspearman(Up')
    GT   = generatorof(CT)

    accθ = 0.0; cnt = 0
    @inbounds for i in 1:d-1, j in i+1:d
        ρ̂ = clamp(ρmat[i,j], -1 + epsρ, 1 - epsρ)
        accθ += ρ⁻¹(GT, ρ̂); cnt += 1
    end
    θ̄ = accθ / cnt

    θlo, θhi = θ_bounds(GT, d)
    if isfinite(θlo) && θ̄ ≤ θlo; θ̄ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ̄ ≥ θhi; θ̄ = prevfloat(float(θhi)); end
    return _make_archimedean(CT, d, θ̄)
end

function fit_irho(::Type{CT}, U::AbstractMatrix; epsρ::Real=1e-10) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    return _fit_irho_1par(CT, U; epsρ=epsρ)
end

# =============================== :IBETA ======================================
function _fit_ibeta_1par(::Type{CT}, U::AbstractMatrix; epsβ::Real=1e-10) where {CT<:ArchimedeanCopula}
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_ibeta(Archimedean) requires d≥2."))
    Up = _as_pxn(d, U)

    β̂ = clamp(blomqvist_beta(Up), -1 + epsβ, 1 - epsβ)
    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)

    fβ = θ -> begin
        Cθ = ArchimedeanCopula(d, GT(θ))
        β(Cθ)
    end

    a0, b0 = _open(θlo, θhi)
    βmin, βmax = fβ(a0), fβ(b0)
    if βmin > βmax; βmin, βmax = βmax, βmin; end

    if β̂ ≤ βmin; return ArchimedeanCopula(d, GT(a0)); end
    if β̂ ≥ βmax; return ArchimedeanCopula(d, GT(b0)); end

    a, b, fa, fb = _bracket_beta!(fβ, β̂, θlo, θhi)
    if sign(fa) == sign(fb)
        K = 24
        @inbounds for t in 1:K-1
            λ = t / K
            x = a + λ*(b - a)
            fx = fβ(x) - β̂
            if sign(fa) != sign(fx); b, fb = x, fx; break
            elseif sign(fx) != sign(fb); a, fa = x, fx; break
            end
        end
    end
    θ = _root1d(fβ, β̂, a, b; tol=1e-10)

    if isfinite(θlo) && θ ≤ θlo; θ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ ≥ θhi; θ = prevfloat(float(θhi)); end
    return _make_archimedean(CT, d, θ)
end

function fit_ibeta(::Type{CT}, U::AbstractMatrix; epsβ::Real=1e-10) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    return _fit_ibeta_1par(CT, U; epsβ=epsβ)
end

# ============================== :MDE (generic) ==============================
function fit_mde(::Type{CT}, U::AbstractMatrix;
                 target::Symbol               = :chi,     # :chi or :gamma
                 distance::Symbol             = :cvm,     # :cvm or :ks
                 start                        = :itau,    # :itau, :irho, :ibeta or numérico
                 reduced::Union{Bool,Nothing} = nothing,
                 maxiter::Int                 = 1000,
                 tol::Real                    = 1e-6) where {CT<:ArchimedeanCopula}

    # --- data ---
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_mde requires d≥2."))
    Up = _as_pxn(d, U)
    reduced = isnothing(reduced) ? _default_reduced(d) : reduced

    # --- rfamily traits ---
    GT = generatorof(CT)
    k  = param_length(GT)
    k == 1 || throw(ArgumentError("fit_mde (this version) is only for 1-parameter families."))
    lo, hi = param_bounds(GT, d)

    # --- seed ---
    θ0_any = if start === :itau
        Distributions.params(fit_itau(CT, Up))
    elseif start === :irho
        Distributions.params(fit_irho(CT, Up))
    elseif start === :ibeta
        Distributions.params(fit_ibeta(CT, Up))
    elseif start isa AbstractVector || start isa Tuple || start isa Real || (start isa AbstractArray && ndims(start)==0)
        start
    else
        throw(ArgumentError("start ∈ {:itau,:irho,:ibeta} or numeric"))
    end
    θ0 = _to_param_vec(θ0_any, 1)

    # --- objetive buffers ---
    df = reduced ? d - 1 : d
    df ≥ 1 || throw(ArgumentError("invalid df (d=$d, reduced=$reduced)"))
    Y     = Vector{Float64}(undef, size(Up,2))
    upbuf = Vector{Float64}(undef, df)

    F = if target === :chi
        dist = Distributions.Chisq(df); y -> Distributions.cdf(dist, y)
    elseif target === :gamma
        dist = Distributions.Gamma(df, 1.0); y -> Distributions.cdf(dist, y)
    else
        throw(ArgumentError("target ∈ {:chi,:gamma}"))
    end

    _score = (target === :chi) ? _score_chi : _score_gamma
    _dist  = (distance === :cvm) ? _dist_cvm! :
             (distance === :ks  ? _dist_ks!  :
              throw(ArgumentError("distance ∈ {:cvm,:ks}")))

    function Qη(η::Float64)
        local Cη
        try
            Cη = _make_archimedean(CT, d, [η])      # CT(d, η)
        catch
            return Inf
        end
        @inbounds for j in axes(Up,2)
            ucol = @view Up[:, j]
            if reduced; _Uprime_reduced!(upbuf, Cη, ucol) else; _Uprime_full!(upbuf, Cη, ucol) end
            Y[j] = _score(upbuf)
        end
        val = _dist(Y, F)
        return isfinite(val) ? val : Inf
    end

    # --- optimization 1D ---
    # If distance=:cvm → smooth function → 1D Brent on [lo,hi]
    lo1, hi1 = float(lo[1]), float(hi[1])
    if isfinite(lo1) && isfinite(hi1)
        res = Optim.optimize(Qη, lo1, hi1)  # Brent
        η̂  = Optim.minimizer(res)
    else
        # fallback: finite box around θ0 if the bounds are ±Inf
        lo_eff, hi_eff = _finite_box([lo1], [hi1], θ0; width=50.0)
        res = Optim.optimize(Qη, lo_eff[1], hi_eff[1])
        η̂  = Optim.minimizer(res)
    end

    # final projection....
    if isfinite(lo1) && η̂ ≤ lo1; η̂ = nextfloat(lo1); end
    if isfinite(hi1) && η̂ ≥ hi1; η̂ = prevfloat(hi1); end

    return _make_archimedean(CT, d, [η̂])
end

# =============================== :MLE ===============================
@inline function _nll_1par(::Type{CT}, d::Integer, Up::AbstractMatrix, θ::Float64) where {CT<:ArchimedeanCopula}
    C = try
        _make_archimedean(CT, d, θ)     # CT(d, θ) o CT(θ...) si tu constructor ya está especializado
    catch
        return Inf
    end
    ll = Distributions.loglikelihood(C, Up)
    return isfinite(ll) ? -ll : Inf
end

@inline function _finite_box1(lo::Float64, hi::Float64, θ0::Float64; width::Float64=50.0)
    a = isfinite(lo) ? lo : (θ0 - width)
    b = isfinite(hi) ? hi : (θ0 + width)
    if !(a < b)
        a, b = min(θ0 - 1.0, θ0), max(θ0, θ0 + 1.0)
    end
    return (a, b)
end

# seed
function _seed1(::Type{CT}, d::Integer, Up::AbstractMatrix, start) where {CT<:ArchimedeanCopula}
    GT = generatorof(CT)
    if start === :itau
        return Distributions.params(_fit_itau_1par(CT, Up))[end]
    elseif start === :irho
        return Distributions.params(_fit_irho_1par(CT, Up))[end]
    elseif start === :ibeta
        return Distributions.params(_fit_ibeta_1par(CT, Up))[end]
    elseif start === :winztau
        τ̂ = StatsBase.corkendall(Up')[1,2]
        τ̂ = _winsorize_tau_vclib(τ̂)
        return τ⁻¹(GT, τ̂)
    elseif start isa Real
        return float(start)
    else
        throw(ArgumentError("start ∈ {:itau,:irho,:ibeta,:winztau} or numeric"))
    end
end

function fit_arch_mle(::Type{CT}, U::AbstractMatrix;
                 start::Union{Symbol,Real} = :itau,
                 maxiter::Int = 800, xtol::Real = 1e-8,
                 return_loglik::Bool = false) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    # --- data ---
    d = size(U,1) ≥ 2 ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_mle(Archimedean,1par) requires d≥2."))
    Up = _as_pxn(d, U)

    # --- parameter bounds ---
    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)
    lo = float(θlo); hi = float(θhi)

    # --- seed ---
    θ0 = _seed1(CT, d, Up, start)
    if isfinite(lo) && θ0 ≤ lo; θ0 = nextfloat(lo); end
    if isfinite(hi) && θ0 ≥ hi; θ0 = prevfloat(hi); end

    # --- objetive 1D ---
    f(θ) = _nll_1par(CT, d, Up, θ)

    # --- optimization 1D ---
    θ̂, fmin = if isfinite(lo) && isfinite(hi)
        res = Optim.optimize(f, lo, hi; abs_tol=xtol)   # Brent
        (Optim.minimizer(res), Optim.minimum(res))
    else
        a, b = _finite_box1(lo, hi, θ0; width=50.0)
        res  = Optim.optimize(f, a, b; abs_tol=xtol)
        (Optim.minimizer(res), Optim.minimum(res))
    end

    # final projection
    if isfinite(lo) && θ̂ ≤ lo; θ̂ = nextfloat(lo); end
    if isfinite(hi) && θ̂ ≥ hi; θ̂ = prevfloat(hi); end

    Ĉ = _make_archimedean(CT, d, θ̂)
    if return_loglik
        return Ĉ, -fmin
    else
        return Ĉ
    end
end