"""
    BB5Copula{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB5Copula(θ, δ)

The BB4 copula in dimension ``d = 2`` is parameterized by ``\\theta \\in [1,\\infty)`` and ``\\delta \\in (0,\\infty). It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp(-[\\delta^{-1}\\log(1 + t)]^{\\frac{1}{\\theta}}),
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.199-200
"""
struct BB5Copula{Tθ,Tδ} <: Copula{2}
    θ::Tθ   # θ ≥ 1
    δ::Tδ   # δ > 0
    function BB5Copula(θ, δ)
        (θ ≥ 1) || throw(ArgumentError("θ must be ≥ 1"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        new{typeof(θ),typeof(δ)}(θ, δ)
    end
end

Distributions.params(C::BB5Copula) = (C.θ, C.δ)

# ---------- CDF ----------
function _cdf(C::BB5Copula, u)
    θ, δ = C.θ, C.δ
    u1, u2 = u
    x = -log(u1);  y = -log(u2) 
    logB = LogExpFunctions.logaddexp(-θ*δ*log(x), -θ*δ*log(y))
    H    = exp(-logB/δ)
    s    = exp(θ*log(x)) + exp(θ*log(y)) - H 
    f    = exp((1/θ)*log(s))
    return exp(-f)
end

# ---------- log-PDF ----------
function Distributions._logpdf(C::BB5Copula, u)
    Tret = promote_type(Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    θ, δ   = C.θ, C.δ
    invθ   = inv(θ);    invδ = inv(δ)
    x = -log(u1); y = -log(u2)

    xθ  = exp( θ*log(x) );     yθ  = exp( θ*log(y) )
    logB = LogExpFunctions.logaddexp(-θ*δ*log(x), -θ*δ*log(y))
    B    = exp(logB)
    H    = exp(-invδ*logB)                        
    s    = xθ + yθ - H
    f    = exp(invθ*log(s))                   
    logC = -f                                   

    Ax   = θ*exp((θ-1)*log(x));      Ay   = θ*exp((θ-1)*log(y))    
    Axx  = θ*(θ-1)*exp((θ-2)*log(x))
    Ayy  = θ*(θ-1)*exp((θ-2)*log(y))

    Bx   = -θ*δ*exp(-(θ*δ+1)*log(x))              
    By   = -θ*δ*exp(-(θ*δ+1)*log(y))
    Bxx  = θ*δ*(θ*δ+1)*exp(-(θ*δ+2)*log(x)) 
    Byy  = θ*δ*(θ*δ+1)*exp(-(θ*δ+2)*log(y))

    H_over_B  = H / B
    H_over_B2 = H / (B*B)

    Hx  = (-invδ) * H_over_B  * Bx
    Hy  = (-invδ) * H_over_B  * By
    Hxx = (-invδ) * ( (-(invδ+1)) * H_over_B2 * Bx*Bx + H_over_B * Bxx )
    Hyy = (-invδ) * ( (-(invδ+1)) * H_over_B2 * By*By + H_over_B * Byy )
    Hxy = (-invδ) * ( (-(invδ+1)) * H_over_B2 * Bx*By )                 # B_xy=0

    sx  = Ax - Hx
    sy  = Ay - Hy
    sxy = -Hxy

    fs  = invθ * exp( (invθ-1)*log(s) )                                 # d(s^{1/θ})/ds
    fss = invθ*(invθ-1) * exp( (invθ-2)*log(s) )

    fx  = fs * sx
    fy  = fs * sy
    fxy = fs * sxy + fss * sx * sy

    tmp = fx*fy - fxy
    tmp > 0 || return Tret(-Inf)

    logc = logC - log(u1) - log(u2) + log(tmp)
    return Tret(logc)
end

archimax_view(C::BB5Copula) = ArchimaxCopula(GumbelCopula(2, C.θ), GalambosCopula(C.δ))

function Distributions._rand!(rng::Distributions.AbstractRNG, C::BB5Copula, x::AbstractVector{T}) where {T<:Real}
    Distributions._rand!(rng, archimax_view(C), x)
end

τ(C::BB5Copula) = τ(archimax_view(C))
