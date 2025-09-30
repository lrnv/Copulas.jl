# helper
params(::Type{T}) where {T<:Copula} = throw("No params() function defined for type T = $T...")
params(CT::Type{<:ArchimedeanCopula}) = params(generatorof(CT))

function _fit(CT::Type{<:ArchimedeanCopula{d,<:UnivariateGenerator} where d}, U, method::Union{Val{:itau},Val{:irho}}; kwargs...)
    d = size(U, 1)
    GT   = generatorof(CT)
    if method == Val{:itau}()
        θs = Base.Fix1(τ⁻¹, GT).(_uppertriangle_gen(StatsBase.corkendall(U')))
    elseif method == Val{:irho}()
        θs = Base.Fix1(ρ⁻¹, GT).(_uppertriangle_gen(StatsBase.corspearman(U')))
    else
        throw("This should never happen.")
    end

    θ = StatsBase.mean(θs)
    θlo, θhi = _θ_bounds(GT, d) ######### Yes this kind of things is good, we should add bounds. 
    θ = clamp(θ, θlo, θhi)

    C = CT(d, θ) # so here we assume that any archimedean with univaraite parameter can be constructed this way. 
    return C, (;) # i found no extra things to give here ? 
end

# ============================= :IBETA =======================================
function _fit(CT::Type{<:ArchimedeanCopula{d,<:UnivariateGenerator} where d}, U, ::Val{:ibeta}; epsβ::Real=1e-10, max_expand::Int=20)
    # data
    d, n = size(U)
    d ≥ 2 || throw(ArgumentError("fit_ibeta(Archimedean) requires d≥2."))

    # β̂ multivariante (Hofert–Mächler–McNeil, ec. (7))
    β̂ = clamp(blomqvist_beta(U), -1 + epsβ, 1 - epsβ)

    GT = generatorof(CT)
    θlo, θhi = _θ_bounds(GT, d)
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
        return ArchimedeanCopula(d, GT(θstar)), (; epsβ)
    end

    # Root by Brent (uses Roots.jl; if you've already imported it, you can leave Roots.find_zero)
    θ = Roots.find_zero(f, (a, b), Roots.Brent(); xatol=1e-10, rtol=0.0)

    if isfinite(θlo) && θ ≤ θlo; θ = nextfloat(float(θlo)); end
    if isfinite(θhi) && θ ≥ θhi; θ = prevfloat(float(θhi)); end

    return ArchimedeanCopula(d, GT(θ)), (; epsβ)
end
function _fit(CT::Type{<:ArchimedeanCopula{d,<:UnivariateGenerator} where d}, U::AbstractMatrix;
                      start::Union{Symbol,Real} = :itau, xtol::Real = 1e-8)

    d, n = size(U)
    GT = generatorof(CT)
    θlo, θhi = _θ_bounds(GT, d)
    lo, hi = float(θlo), float(θhi)

    θ0 = start isa Symbol ? only(Distributions.params(fit(CT,U, Val{quickstart}))) : start
    if isfinite(lo) && θ0 ≤ lo; θ0 = nextfloat(lo); end
    if isfinite(hi) && θ0 ≥ hi; θ0 = prevfloat(hi); end

    f(θ) = Distributions.loglikelihood(CT(d, θ), U)

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

    C = CT(d, θ̂ ), 
    meta = (; estimator=:mle, θ̂=θ̂, ll=-fmin, optimizer=:Brent,
            maxiter=maxiter, xtol=xtol, #vcov=V, vcov_method = isnothing(V) ? :none : :hessian,
            converged=true, iterations=0, elapsed_sec=t)
    return C, meta

    # I did not look at this its probably completely nonfunctioning now, sorry about that. 
    # d  = length(Ĉ)
    # θ̂ = Float64(only(Distributions.params(Ĉ)))
    # I  = _obsinfo1(CT, d, U, θ̂)
    # V = (isfinite(I) && I > 0) ? [1 / I;;] : nothing
end

@inline function _finite_box1(lo::Float64, hi::Float64, θ0::Float64; width::Float64=50.0)
    a = isfinite(lo) ? lo : (θ0 - width)
    b = isfinite(hi) ? hi : (θ0 + width)
    if !(a < b)
        a, b = min(θ0 - 1.0, θ0), max(θ0, θ0 + 1.0)
    end
    return (a, b)
end

# Second numerical derivative of ℓ(θ) in θ̂ ⇒ I_obs(θ̂) = -ℓ''(θ̂)
# ℓ''(θ̂) robust (1D) with scaling and fallback; returns I_obs = -ℓ''(θ̂)
@inline function _obsinfo1(::Type{CT}, d::Integer, U::AbstractMatrix, θ̂::Float64;
                           scale::Float64 = 1.0) where {CT<:ArchimedeanCopula}
    GT = generatorof(CT)
    θlo, θhi = θ_bounds(GT, d)
    lo, hi = float(θlo), float(θhi)

    f(θ) = Distributions.loglikelihood(CT(d, θ), U)

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