# ======================= src/Fitting/ArchimedeanFit.jl =======================
#
# Methods: :itau, :irho, :ibeta, :mde, :mle
#
# Prerequisites in the module:
# - generatorof(::Type{<:ArchimedeanCopula}) -> Generator type TG
# - τ, τ⁻¹, ρ, ρ⁻¹, β, β⁻¹ (if there are no closed inverses, we use numerical inversion)
# - Distributions.cdf(C, u) and loglikelihood(C, U)
# - _as_pxn(C,U) or _as_pxn(p,U) defined in utils.jl

# Returns the generator type associated with the Archimedean copula CT
# (I assume you already have it defined in the module)
# generatorof(::Type{<:ArchimedeanCopula}) -> TG (e.g. GumbelGenerator)

# --- Robust factory that promotes types and builds via the generator ---

# 1 parameter (Real)
# ====================== Archimedean (1 parameter) =======================

# --- Factory ---
function _make_archimedean(::Type{CT}, d::Integer, θ::Real) where {CT<:ArchimedeanCopula}
    TG = generatorof(CT)                      # p.ej. ClaytonGenerator
    return ArchimedeanCopula(d, TG(float(θ)))
end

# helper
param_length(::Type{GT}) where {GT<:Generator} = fieldcount(GT)
nparams(::Type{CT}) where {CT<:ArchimedeanCopula} = param_length(generatorof(CT))
_assert_oneparam(x) = (nparams(x) == 1) || throw(ArgumentError("This method requires a 1-parameter family; use method=:mle for multi-parameter families."))

# ====
# --- helper: estimate θ with the given rank-based estimator ---
# --- estimate θ̂ with the given rank-based method ---
@inline function _θhat_rank1(::Type{CT}, Up::AbstractMatrix; estimator::Symbol, kw...) where {CT<:ArchimedeanCopula}
    C = estimator === :itau  ? _fit_itau_1par(CT, Up; kw...) :
        estimator === :irho  ? _fit_irho_1par(CT, Up; kw...) :
        estimator === :ibeta ? _fit_ibeta_1par(CT, Up; kw...) :
        error("estimator ∈ {:itau,:irho,:ibeta}")
    return Float64(only(Distributions.params(C)))
end

function _vcov_jackknife_obs(::Type{CT}, U::AbstractMatrix; estimator::Symbol, kw...) where {CT<:ArchimedeanCopula}
    # data
    d  = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("jackknife requires d≥2."))
    Up = _as_pxn(d, U)
    n  = size(Up, 2)
    n ≥ 3 || throw(ArgumentError("jackknife requires n≥3."))

    θminus = Vector{Float64}(undef, n)
    idx    = Vector{Int}(undef, n-1)

    @inbounds for j in 1:n
        # índices 1…n excepto j
        k = 1
        for t in 1:n
            if t == j; continue; end
            idx[k] = t; k += 1
        end
        Uminus = view(Up, :, idx)  # ✅ vista válida (sin @view)
        θminus[j] = _θhat_rank1(CT, Uminus; estimator=estimator, kw...)
    end

    θbar = StatsBase.mean(θminus)
    vjk  = (n-1)/n * sum(abs2, θminus .- θbar)  # Float64

    # 1×1 matriz
    V = reshape([vjk], 1, 1)   # ✅ o bien: V = fill(vjk, 1, 1)
    return V, (; vcov_method=:jackknife_obs, n=n)
end


# =============================== :ITAU =======================================
function _fit_itau_1par(::Type{CT}, U::AbstractMatrix; epsτ::Real=1e-10) where {CT<:ArchimedeanCopula}
    d = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_itau(Archimedean) requires d≥2."))
    Up = _as_pxn(d, U)

    τmat = StatsBase.corkendall(Up')           # p×p
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

# Wrapper for dispatcher → (C, meta)
function _fit_itau(::Type{CT}, U::AbstractMatrix; epsτ::Real=1e-10, vcov_method::Symbol=:jackknife_obs) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    C = _fit_itau_1par(CT, U; epsτ=epsτ)
    V, vmeta = vcov_method === :jackknife_obs ? _vcov_jackknife_obs(CT, U; estimator=:itau, epsτ=epsτ) :
             error("vcov_method no soportado")
    return C, (; estimator=:itau, epsτ=epsτ, vcov=V, vcov_details=vmeta)
end

# =============================== :IRHO =======================================
function _fit_irho_1par(::Type{CT}, U::AbstractMatrix; epsρ::Real=1e-10) where {CT<:ArchimedeanCopula}
    d  = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_irho requires d≥2."))
    Up  = _as_pxn(d, U)
    ρmat = StatsBase.corspearman(Up')
    GT   = generatorof(CT)

    accθ = 0.0; cnt = 0
    @inbounds for i in 1:d-1, j in i+1:d
        ρ̂ = clamp(ρmat[i,j], -1 + epsρ, 1 - epsρ)
        accθ += ρ⁻¹(GT, ρ̂); cnt += 1
    end
    θ̄ = accθ / cnt

    θlo, θhi = θ_bounds(GT, d)
    lo = float(θlo); hi = float(θhi)
    if isfinite(lo) && θ̄ ≤ lo; θ̄ = nextfloat(lo); end
    if isfinite(hi) && θ̄ ≥ hi; θ̄ = prevfloat(hi); end
    return _make_archimedean(CT, d, θ̄)
end
function _fit_irho(::Type{CT}, U::AbstractMatrix; epsρ::Real=1e-10, vcov_method::Symbol=:jackknife_obs) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    C = _fit_irho_1par(CT, U; epsρ=epsρ)
    V, vmeta = vcov_method === :jackknife_obs ? _vcov_jackknife_obs(CT, U; estimator=:irho, epsρ=epsρ) :
             error("vcov_method no soportado")
    return C, (; estimator=:irho, epsρ=epsρ, vcov=V, vcov_details=vmeta)
end

# ============================= :IBETA =======================================
function _fit_ibeta_1par(::Type{CT}, U::AbstractMatrix;
                         epsβ::Real=1e-10, max_expand::Int=20) where {CT<:ArchimedeanCopula}
    # data
    d = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_ibeta(Archimedean) requires d≥2."))
    Up = _as_pxn(d, U)

    # β̂ multivariante (Hofert–Mächler–McNeil, ec. (7))
    β̂ = clamp(blomqvist_beta(Up), -1 + epsβ, 1 - epsβ)

    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)
    a = isfinite(θlo) ? nextfloat(float(θlo)) : -5.0
    b = isfinite(θhi) ? prevfloat(float(θhi)) :  5.0
    if !(a < b); a, b = b, a; end  # ensure order

    f(θ) = begin
        Cθ = ArchimedeanCopula(d, GT(θ))
        β(Cθ) - β̂
    end

    fa = f(a); fb = f(b)

    #If there is at least one infinite bound and there is no change of sign, we expand
    if ( !isfinite(θlo) || !isfinite(θhi) ) && sign(fa) == sign(fb)
        k = 0
        while sign(fa) == sign(fb) && k < max_expand
            if !isfinite(θhi)
                b *= 2
                fb = f(b)
                if sign(fa) != sign(fb); break; end
            end
            if !isfinite(θlo) && sign(fa) == sign(fb)
                a *= 2
                fa = f(a)
            end
            k += 1
        end
    end

    # If there is still no bracket, β̂ is outside the achievable range → nearest extreme
    if sign(fa) == sign(fb)
        θstar = (abs(fa) <= abs(fb)) ? a : b
        return ArchimedeanCopula(d, GT(θstar))
    end

    # Root by Brent (uses Roots.jl; if you've already imported it, you can leave Roots.find_zero)
    θ = Roots.find_zero(f, (a, b), Roots.Brent(); xatol=1e-10, rtol=0.0)

    if isfinite(θlo) && θ ≤ θlo; θ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ ≥ θhi; θ = prevfloat(float(θhi)); end

    return ArchimedeanCopula(d, GT(θ))
end


function _fit_ibeta(::Type{CT}, U::AbstractMatrix;
                    epsβ::Real=1e-10, vcov_method::Symbol=:jackknife_obs) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    C = _fit_ibeta_1par(CT, U; epsβ=epsβ)
    V, vmeta = vcov_method === :jackknife_obs ?
        _vcov_jackknife_obs(CT, U; estimator=:ibeta, epsβ=epsβ) :
        error("vcov_method no soportado")
    return C, (; estimator=:ibeta, epsβ=epsβ, vcov=V, vcov_details=vmeta)
end
# ============================= :MLE =======================================
# Seed
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
@inline function _nll_1par(::Type{CT}, d::Integer, Up::AbstractMatrix, θ::Real) where {CT<:ArchimedeanCopula}
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
function fit_arch_mle(::Type{CT}, U::AbstractMatrix;
                      start::Union{Symbol,Real} = :itau,
                      maxiter::Int = 800, xtol::Real = 1e-8,
                      return_loglik::Bool = false) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)

    d  = (size(U,1) ≥ 2) ? size(U,1) : size(U,2)
    d ≥ 2 || throw(ArgumentError("fit_mle(Archimedean,1par) requires d≥2."))
    Up = _as_pxn(d, U)

    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)
    lo, hi = float(θlo), float(θhi)

    θ0 = _seed1(CT, d, Up, start)
    if isfinite(lo) && θ0 ≤ lo; θ0 = nextfloat(lo); end
    if isfinite(hi) && θ0 ≥ hi; θ0 = prevfloat(hi); end

    f(θ) = _nll_1par(CT, d, Up, θ)

    θ̂, fmin = if isfinite(lo) && isfinite(hi)
        res = Optim.optimize(f, lo, hi; abs_tol=xtol)
        (Optim.minimizer(res), Optim.minimum(res))
    else
        a, b = _finite_box1(lo, hi, θ0; width=50.0)
        res  = Optim.optimize(f, a, b; abs_tol=xtol)
        (Optim.minimizer(res), Optim.minimum(res))
    end

    if isfinite(lo) && θ̂ ≤ lo; θ̂ = nextfloat(lo); end
    if isfinite(hi) && θ̂ ≥ hi; θ̂ = prevfloat(hi); end

    Ĉ = _make_archimedean(CT, d, θ̂)
    if return_loglik
        return Ĉ, -fmin
    else
        return Ĉ
    end
end

# Second numerical derivative of ℓ(θ) in θ̂ ⇒ I_obs(θ̂) = -ℓ''(θ̂)
# ℓ''(θ̂) robust (1D) with scaling and fallback; returns I_obs = -ℓ''(θ̂)
@inline function _obsinfo1(::Type{CT}, d::Integer, Up::AbstractMatrix, θ̂::Float64;
                           scale::Float64 = 1.0) where {CT<:ArchimedeanCopula}
    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)
    lo, hi = float(θlo), float(θhi)

    f(θ) = Distributions.loglikelihood(_make_archimedean(CT, d, θ), Up)

    hbase = scale * max(1.0, abs(θ̂)) * cbrt(eps(Float64))

    function fit_h(h::Float64)
        h0 = h
        if isfinite(lo) && θ̂ - h0 ≤ lo; h0 = max((θ̂ - nextfloat(lo))/2, eps()); end
        if isfinite(hi) && θ̂ + h0 ≥ hi; h0 = max((prevfloat(hi) - θ̂)/2, eps()); end
        return h0
    end

    #1) attempt with simple central difference
    for factor in (1.0, 3.0, 10.0)# fallback: increase the step if necessary
        h = fit_h(hbase * factor)
        h ≤ 0 && continue
        fp, f0, fm = f(θ̂ + h), f(θ̂), f(θ̂ - h)
        ℓpp = (fp - 2f0 + fm) / (h*h)
        I = -ℓpp
        if isfinite(I) && I > 0
            return I
        end
    end

    # 2) "smoothed" attempt: parabolic fitting with ±h and ±2h
    h = fit_h(hbase * 2)
    if h > 0
        θs = (θ̂ - 2h, θ̂ - h, θ̂, θ̂ + h, θ̂ + 2h)
        fs = map(f, θs)
        offs = (-2h, -h, 0.0, h, 2h)
        X = [ (x^2, x, 1.0) for x in offs ]
        A = zeros(3,3); y = zeros(3)
        for i in 1:5
            a1,b1,c1 = X[i]
            A[1,1] += a1*a1; A[1,2] += a1*b1; A[1,3] += a1*1
            A[2,1] += b1*a1; A[2,2] += b1*b1; A[2,3] += b1*1
            A[3,1] += 1*a1;  A[3,2] += 1*b1;  A[3,3] += 1*1
            y[1]   += a1*fs[i]; y[2] += b1*fs[i]; y[3] += 1*fs[i]
        end
        abc = A \ y
        a = abc[1]
        I = -(2a)
        if isfinite(I) && I > 0
            return I
        end
    end

    return NaN  # unsuccessful → the caller will put vcov = nothing
end


# MLE wrapper
function _fit_mle_a(::Type{CT}, U::AbstractMatrix;
                  start::Union{Symbol,Real} = :itau,
                  maxiter::Int = 800, xtol::Real = 1e-8) where {CT<:ArchimedeanCopula}
    _assert_oneparam(CT)
    t = @elapsed Ĉ, ll = fit_arch_mle(CT, U; start=start, maxiter=maxiter, xtol=xtol, return_loglik=true)
    d  = length(Ĉ)
    Up = _as_pxn(d, U)
    θ̂ = Float64(only(Distributions.params(Ĉ)))
    I  = _obsinfo1(CT, d, Up, θ̂)
    V = (isfinite(I) && I > 0) ? [1 / I;;] : nothing
    meta = (; estimator=:mle, θ̂=θ̂, ll=ll, optimizer=:Brent,
             maxiter=maxiter, xtol=xtol, vcov=V, vcov_method = isnothing(V) ? :none : :hessian,
             converged=true, iterations=0, elapsed_sec=t)
    return Ĉ, meta
end

############################################################################
##### In the future we could move this to a better place 
θ_bounds(::Type{<:ClaytonGenerator}, d::Integer) = (-1/(d-1),  Inf)
θ_bounds(::Type{<:AMHGenerator}, d::Integer)     = (-1,  1)
θ_bounds(::Type{<:GumbelGenerator},  ::Integer)  = (1.0,       Inf)
θ_bounds(::Type{<:JoeGenerator},     ::Integer)  = (1.0,       Inf)
θ_bounds(::Type{<:FrankGenerator}, d::Integer)   = d ≥ 3 ? (nextfloat(0.0),  Inf) : (-Inf, Inf)
θ_bounds(::Type{<:GumbelBarnettGenerator}, ::Integer) = (0.0, 1.0)

function _fit_mde(::Type{CT}, U::AbstractMatrix; kwargs...) where {CT<:Copula}
    throw(ArgumentError(throw(ArgumentError("method=:mde is not yet available for $(CT). Use :mle, :itau, :irho, :ibeta or :emp."))))
end
