# helper
params(::Type{T}) where {T<:Copula} = throw("No params() function defined for type T = $T...")
params(CT::Type{<:ArchimedeanCopula}) = params(generatorof(CT))

########## Archimedean 1-par fitting (itau / irho / ibeta / mle) ##########
# ========== :itau ==========
function _fit(CT::Type{<:ArchimedeanCopula}, U::AbstractMatrix, ::Val{:itau};
              eps::Real=1e-10)
    d = size(U,1)
    d ≥ 2 || throw(ArgumentError("itau requiere d≥2"))
    τmat = StatsBase.corkendall(U')              # d×d
    GT   = generatorof(CT)
    θs   = map(v -> τ⁻¹(GT, clamp(v, -1+eps, 1-eps)), _uppertriangle_stats(τmat))
    θ    = StatsBase.mean(θs)
    lo, hi = _θ_bounds(GT, d)
    if isfinite(lo) && θ ≤ lo; θ = nextfloat(float(lo)); end
    if isfinite(hi) && θ ≥ hi; θ = prevfloat(float(hi)); end
    return CT(d, θ), (; estimator=:itau, eps)
end

# ========== :irho ==========
function _fit(CT::Type{<:ArchimedeanCopula}, U::AbstractMatrix, ::Val{:irho};
              eps::Real=1e-10)
    d = size(U,1)
    d ≥ 2 || throw(ArgumentError("irho requiere d≥2"))
    ρmat = StatsBase.corspearman(U')             # d×d
    GT   = generatorof(CT)
    θs   = map(v -> ρ⁻¹(GT, clamp(v, -1+eps, 1-eps)), _uppertriangle_stats(ρmat))
    θ    = StatsBase.mean(θs)
    lo, hi = _θ_bounds(GT, d)
    if isfinite(lo) && θ ≤ lo; θ = nextfloat(float(lo)); end
    if isfinite(hi) && θ ≥ hi; θ = prevfloat(float(hi)); end
    return CT(d, θ), (; estimator=:irho, eps)
end

# ========== :ibeta (root en 1D con Brent) ==========
function _fit(CT::Type{<:ArchimedeanCopula}, U::AbstractMatrix, ::Val{:ibeta};
              epsβ::Real=1e-10, max_expand::Int=20)
    d = size(U,1)
    d ≥ 2 || throw(ArgumentError("ibeta requiere d≥2"))
    β̂ = clamp(blomqvist_beta(U), -1 + epsβ, 1 - epsβ)

    GT = generatorof(CT)
    θlo, θhi = _θ_bounds(GT, d)
    a = isfinite(θlo) ? nextfloat(float(θlo)) : -5.0
    b = isfinite(θhi) ? prevfloat(float(θhi)) :  5.0
    if !(a < b); a, b = b, a; end

    f(θ) = begin
        Cθ = CT(d, θ)
        β(Cθ) - β̂
    end

    fa, fb = f(a), f(b)

    # we expand if there is an infinite bound and there is no sign change
    if ( !isfinite(θlo) || !isfinite(θhi) ) && sign(fa) == sign(fb)
        k = 0
        while sign(fa) == sign(fb) && k < max_expand
            if !isfinite(θhi); b *= 2; fb = f(b); end
            if sign(fa) != sign(fb); break; end
            if !isfinite(θlo); a *= 2; fa = f(a); end
            k += 1
        end
    end

    # if no bracket yet → β̂ out of range → nearest end
    if sign(fa) == sign(fb)
        θstar = (abs(fa) ≤ abs(fb)) ? a : b
        return CT(d, θstar), (; estimator=:ibeta, epsβ)
    end

    θ = Roots.find_zero(f, (a, b), Roots.Brent(); xatol=1e-10, rtol=0.0)
    if isfinite(θlo) && θ ≤ θlo; θ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ ≥ θhi; θ = prevfloat(float(θhi)); end
    return CT(d, θ), (; estimator=:ibeta, epsβ)
end

# ========== :mle (Brent en 1D; no vcov by default) ==========
@inline function _finite_box1(lo::Float64, hi::Float64, θ0::Float64; width::Float64=50.0)
    a = isfinite(lo) ? lo : (θ0 - width)
    b = isfinite(hi) ? hi : (θ0 + width)
    if !(a < b); a, b = min(θ0 - 1.0, θ0), max(θ0, θ0 + 1.0); end
    return (a, b)
end

function _fit(CT::Type{<:ArchimedeanCopula}, U::AbstractMatrix, ::Val{:mle};
              start::Union{Symbol,Real}=:itau, xtol::Real=1e-8)
    d = size(U,1)
    d ≥ 2 || throw(ArgumentError("mle requiere d≥2"))
    GT = generatorof(CT)
    lo, hi = map(float, _θ_bounds(GT, d))

    # seed
    θ0 = if start === :itau
        τmat = StatsBase.corkendall(U')
        vals = collect(_uppertriangle_stats(τmat))
        StatsBase.mean(map(v -> τ⁻¹(GT, clamp(v, -0.9999999999, 0.9999999999)), vals))
    elseif start === :irho
        ρmat = StatsBase.corspearman(U')
        vals = collect(_uppertriangle_stats(ρmat))
        StatsBase.mean(map(v -> ρ⁻¹(GT, clamp(v, -0.9999999999, 0.9999999999)), vals))
    elseif start isa Real
        float(start)
    else
        error("start ∈ {:itau,:irho} or numeric")
    end
    if isfinite(lo) && θ0 ≤ lo; θ0 = nextfloat(lo); end
    if isfinite(hi) && θ0 ≥ hi; θ0 = prevfloat(hi); end

    f(θ) = begin
        Cθ = CT(d, θ)
        -Distributions.loglikelihood(Cθ, U)
    end

    t = @elapsed begin
        if isfinite(lo) && isfinite(hi)
            res = Optim.optimize(f, lo, hi; abs_tol=xtol)
            global θ̂ = Optim.minimizer(res); global nll = Optim.minimum(res)
            global _conv = Optim.converged(res); global _it = Optim.iterations(res)
        else
            a,b = _finite_box1(lo, hi, θ0; width=50.0)
            res = Optim.optimize(f, a, b; abs_tol=xtol)
            global θ̂ = Optim.minimizer(res); global nll = Optim.minimum(res)
            global _conv = Optim.converged(res); global _it = Optim.iterations(res)
        end
    end

    if isfinite(lo) && θ̂ ≤ lo; θ̂ = nextfloat(lo); end
    if isfinite(hi) && θ̂ ≥ hi; θ̂ = prevfloat(hi); end

    return CT(d, θ̂), (; estimator=:mle, θ̂=θ̂, optimizer=:Brent,
                        xtol=xtol, converged=_conv, iterations=_it, elapsed_sec=t)
end

############################################################################
# These theta bounds are not really true, you should match them with the 
# max monotony 
# in fact they are bascially "inverses" of the max monotony functions. 
# They should be moved to the geenrator files would be easier. 
_θ_bounds(::Type{<:ClaytonGenerator},      d::Integer) = (-1/(d-1),  Inf)
_θ_bounds(::Type{<:AMHGenerator},          d::Integer) = (-1,  1)
_θ_bounds(::Type{<:GumbelGenerator},        ::Integer) = (1.0,       Inf)
_θ_bounds(::Type{<:JoeGenerator},           ::Integer) = (1.0,       Inf)
_θ_bounds(::Type{<:FrankGenerator},        d::Integer) = d ≥ 3 ? (nextfloat(0.0),  Inf) : (-Inf, Inf)
_θ_bounds(::Type{<:GumbelBarnettGenerator}, ::Integer) = (0.0, 1.0)