# =========================== src/Fitting/Traits.jl ===========================
const _N01 = Distributions.Normal()
const _δθ = 1e-8
const _EULER_GAMMA = Base.MathConstants.eulergamma
# ---- Framily Domain θ (minimum viable) --------
θ_bounds(::Type{<:ClaytonGenerator}, d::Integer) = (-1/(d-1),  Inf)
θ_bounds(::Type{<:AMHGenerator}, d::Integer) = (-1,  1)
θ_bounds(::Type{<:GumbelGenerator},  ::Integer)  = (1.0,       Inf)
θ_bounds(::Type{<:JoeGenerator},     ::Integer)  = (1.0,       Inf)
θ_bounds(::Type{<:FrankGenerator}, d::Integer) = d ≥ 3 ? (nextfloat(0.0),  Inf) : (-Inf, Inf)
θ_bounds(::Type{<:GumbelBarnettGenerator}, ::Integer) = (0.0, 1.0)

θ_bounds(::Type, ::Integer) = (-Inf, Inf)  # fallback

param_length(::Type{GT}) where {GT} = 1
param_bounds(::Type{GT}, d::Integer) where {GT} = begin
    lo, hi = θ_bounds(GT, d)
    (Float64[lo], Float64[hi])
end
# ---- practical bounds ----------------
# ============================ family bounds ============================

# Each method returns (lo, hi) as Float64 vectors for (θ, δ)
_bounds(::Type{BB1Copula}) = (Float64[nextfloat(0.0), 1.0],     Float64[7.0, 7.0])            # θ > 0, δ ≥ 1
_bounds(::Type{BB6Copula}) = (Float64[1.0+eps(), 1.0],          Float64[6.0, 8.0])            # θ ≥ 1, δ ≥ 1
_bounds(::Type{BB7Copula}) = (Float64[1.0, 0.01],               Float64[6.0, 25.0])           # θ ≥ 1, δ > 0
_bounds(::Type{BB8Copula}) = (Float64[1.0, 0.01],               Float64[8.0, 1.0])            # θ ≥ 1, 0<δ≤1
_bounds(::Type{BB9Copula}) = (Float64[1.0, 0.01],               Float64[8.0, 25.0])           # θ ≥ 1, δ > 0
_bounds(::Type{BB10Copula})= (Float64[nextfloat(0.0), 1e-6],    Float64[8.0, 1.0])          # θ > 0, 0≤δ≤1

# If requested for another unsupported family, throw a clear error
function _bounds(::Type{T}) where {T<:Copula}
    throw(ArgumentError("Bounds not defined for $(T)."))
end
# --- Finite number box for the optimizer (replaces ±Inf)
function _finite_box(lo::AbstractVector, hi::AbstractVector, θ0::AbstractVector; width::Real=50.0)
    lo2 = similar(lo, Float64); hi2 = similar(hi, Float64)
    @inbounds for i in eachindex(lo)
        lo2[i] = isfinite(lo[i]) ? float(lo[i]) : float(θ0[i] - width)
        hi2[i] = isfinite(hi[i]) ? float(hi[i]) : float(θ0[i] + width)
        if !(lo2[i] < hi2[i])
            lo2[i] = hi2[i] - 1.0
        end
    end
    return lo2, hi2
end
# -------------------------- Numerical Helpers --------------------------------

function _root1d(f, target, lo, hi; tol=1e-10, maxiter=80)
    # 1) Brent (Roots.jl)
    try
        return Roots.find_zero(θ -> f(θ) - target, (lo, hi), Roots.Brent(); xatol=tol, rtol=0)
    catch
        # 2) Fallback: minimum bisecction
        a, b = lo, hi
        fa, fb = f(a) - target, f(b) - target
        (sign(fa) != sign(fb)) || error("raíz no acotada en [$lo,$hi]")
        for _ in 1:maxiter
            c = (a + b)/2
            fc = f(c) - target
            (abs(fc) ≤ tol || abs(b-a) ≤ 2tol) && return c
            if sign(fa) != sign(fc)
                b, fb = c, fc
            else
                a, fa = c, fc
            end
        end
        return (a + b)/2
    end
end

function _to_param_vec(x, k::Int)
    v = if x isa AbstractVector
        Float64.(x)
    elseif x isa Tuple
        Float64[ x... ]     # <— importante: no existe Vector{Float64}(::Tuple)
    elseif x isa Real
        Float64[x]
    elseif x isa AbstractArray && ndims(x) == 0
        Float64[x[]]
    else
        error("Inicial start no convertible a vector de parámetros.")
    end
    (length(v) == k) || error("dim(start)=$(length(v))≠$k")
    return v
end

# --- Kendall distribution K_C(t) ---
function kendall_C(C::ArchimedeanCopula{d,TG}, t::Real) where {d,TG}
   # t must be at (0,1); light numeric clipping
    tt = clamp(t, eps(Float64), 1 - eps(Float64))
    x  = ϕ⁻¹(C.G, tt)       # x = ψ^{-1}(t)
    s  = zero(x)
    @inbounds for k in 0:d-1
       # Note: factorial(k) is small for usual d; if d is large, may can we caches.???
        s += ϕ⁽ᵏ⁾(C.G, Val{k}(), x) * ((-x)^k) / factorial(k)
    end
    return s
end
# --- Automatic criterion: d > 5 use reduced version; otherwise, complete ---
_use_reduced(d::Integer) = d > 5

_up_len(d::Integer)      = _use_reduced(d) ? d - 1 : d  # size of vector U'

function _Uprime_reduced!(up::AbstractVector, Cθ::ArchimedeanCopula, u::AbstractVector)
    d = length(u); @assert length(up) == d-1
    x = similar(u)
    @inbounds for k in 1:d
        uk = clamp(u[k], 1e-12, 1 - 1e-12)
        x[k] = ϕ⁻¹(Cθ.G, uk)
    end
    s = zero(eltype(u))
    @inbounds for j in 1:d-1
        s += x[j]
        denom = s + x[j+1]
        ratio = denom == 0 ? 0.5 : s/denom
        up[j] = ratio^j
    end
    return up
end

function _Uprime_full!(up::AbstractVector, Cθ::ArchimedeanCopula, u::AbstractVector)
    d = length(u); @assert length(up) == d
    _Uprime_reduced!(view(up, 1:d-1), Cθ, u)
    Cu = Distributions.cdf(Cθ, u)              # C(U) ∈ (0,1)
    up[d] = kendall_C(Cθ, Cu)                  # U'_d = K(C(U))
    return up
end

function _Uprime!(up::AbstractVector, Cθ::ArchimedeanCopula, u::AbstractVector)
    d = length(u); @assert length(up) == _up_len(d)
    if _use_reduced(d)
        _Uprime_reduced!(up, Cθ, u)
    else
        _Uprime_full!(up, Cθ, u)
    end
    return up
end

# Recommended heuristic (Hering, 2011)
_default_reduced(d::Integer) = d > 3  # Low: d≤3 → complete; High: d>3 → reduced

# ============================ Traits EVC ============================

# ========== API base EV ==========
# Para EV, todo se deriva de _bounds(CT) (práctico/numérico).
# Define _bounds(::Type{FamiliaEV}) para TODAS las familias EV (ya tienes varios).

# Fallbacks explícitos (si te olvidas de definirlos en una familia)
function _bounds(::Type{CT}) where {CT<:ExtremeValueCopula}
    throw(ArgumentError("_bounds not defined for $(CT). Define practical bounds (lo, hi)."))
end

# Param length/bounds definidos a partir de _bounds(CT)
param_length(::Type{CT}) where {CT<:ExtremeValueCopula} = length(first(_bounds(CT)))

param_bounds(::Type{CT}) where {CT<:ExtremeValueCopula} = begin
    lo, hi = _bounds(CT)
    # Asegurar vectores Float64
    (Float64[lo...], Float64[hi...])
end

# Alias público (si quieres un nombre “bonito”)
bounds(::Type{CT}) where {CT<:ExtremeValueCopula} = param_bounds(CT)

# Construcción genérica desde vector θ (longitud 1,2,3) usando tus builders
function make(::Type{CT}, θ::AbstractVector) where {CT<:ExtremeValueCopula}
    k = length(θ)
    if k == 1
        return CT(θ[1])
    elseif k == 2
        return _build_twoparam(CT, θ[1], θ[2])
    elseif k == 3
        return _build_threeparam(CT, θ[1], θ[2], θ[3])
    else
        throw(ArgumentError("make($(CT), θ): soportado sólo para k∈{1,2,3}; recibido k=$(k)."))
    end
end

# Ejemplos EV (prácticos)
_bounds(::Type{GalambosCopula})     = (Float64[0.0],       Float64[50.0])
_bounds(::Type{LogCopula})          = (Float64[1.0],       Float64[50.0])
_bounds(::Type{CuadrasAugeCopula})  = (Float64[0.0],       Float64[1.0])
_bounds(::Type{MixedCopula})        = (Float64[0.0],       Float64[1.0])
_bounds(::Type{HuslerReissCopula})  = (Float64[0.0],       Float64[50.0])

_bounds(::Type{AsymGalambosCopula}) = (Float64[nextfloat(0.0), 0.0, 0.0],
                                       Float64[20.0,              1.0, 1.0])

_bounds(::Type{AsymLogCopula})      = (Float64[1.0 + eps(), 0.0, 0.0],
                                       Float64[20.0,          1.0, 1.0])

_bounds(::Type{BC2Copula})          = (Float64[nextfloat(0.0), nextfloat(0.0)],
                                       Float64[1.0,             1.0])

_bounds(::Type{AsymMixedCopula})    = (Float64[0.0, 0.0], Float64[1.5, 1.0])

# ============================ NLL (3 params) ============================
_build_threeparam(::Type{CT}, a::Float64, b::Float64, c::Float64) where {CT<:ExtremeValueCopula} = CT(a,b,c)

# CT(a, [b,c])
_build_threeparam(::Type{AsymGalambosCopula}, a::Float64, b::Float64, c::Float64) =
    AsymGalambosCopula(a, [b, c])

_build_threeparam(::Type{AsymLogCopula}, a::Float64, b::Float64, c::Float64) =
    AsymLogCopula(a, [b, c])

_build_twoparam(::Type{CT}, x1::Float64, x2::Float64) where {CT<:Copula} = CT(x1, x2)

_build_twoparam(::Type{BC2Copula}, a::Float64, b::Float64) = BC2Copula(a, b)

function _build_twoparam(::Type{AsymMixedCopula}, p::Float64, s::Float64)
    θ1 = p
    L  = -θ1/3
    U  = (θ1 ≤ 1.0) ? (1 - θ1)/2 : (1 - θ1)
    θ2 = L + s*(U - L)
    return AsymMixedCopula([θ1, θ2])
end

# ============================ NLL (2 parámetros, genérico) ============================
@inline function _nll_two_params(::Type{CT}, U::AbstractMatrix,
                                 x1::Float64, x2::Float64) where {CT<:Copula}
    C = try
        _build_twoparam(CT, x1, x2)  # por defecto CT(x1,x2); con overrides donde aplique
    catch
        return Inf
    end
    Up = _as_pxn(U)
    ll = Distributions.loglikelihood(C, Up)
    return isfinite(ll) ? -ll : Inf
end

# ============================ NLL (3 parámetros, genérico) ============================
@inline function _nll_three_param(::Type{CT}, U::AbstractMatrix,
                                  a::Float64, b::Float64, c::Float64) where {CT<:ExtremeValueCopula}
    C = try
        _build_threeparam(CT, a, b, c)
    catch
        return Inf
    end
    Up = _as_pxn(U)
    ll = Distributions.loglikelihood(C, Up)
    return isfinite(ll) ? -ll : Inf
end
