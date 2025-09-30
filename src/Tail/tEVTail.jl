"""
    tEVTail{Tdf,Tρ}

Fields:
  - ν::Real — degrees of freedom (ν > 0)
  - ρ::Real — correlation parameter (ρ ∈ (-1,1])

Constructor

    tEVCopula(ν, ρ)
    ExtremeValueCopula(2, tEVTail(ν, ρ))

The (bivariate) extreme-t copula is parameterized by ``\\nu > 0`` and \\rho \\in (-1,1]``.  
Its Pickands dependence function is

```math
A(x) = xt_{\\nu+1}(Z_x) +(1-x)t_{\\nu+1}(Z_{1-x})
```
Where ``t_{\\nu + 1}`` is the cumulative distribution function (CDF) of the standard t distribution with ``\\nu + 1`` degrees of freedom and

```math
Z_x = \\frac{(1+\\nu)^{1/2}{\\sqrt{1-\\theta^2}}\\left[ \\left(\\frac{x}{1-x} \\right)^{1/\\nu} - \\theta \\right]
```

Special cases:

* ρ = 0 ⇒ IndependentCopula
* ρ = 1 ⇒ M Copula (upper Fréchet-Hoeffding bound)

References:

* [nikoloulopoulos2009extreme](@cite) Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of multivariate t copulas. Extremes, 12, 129-148.
"""
struct tEVTail{T} <: Tail2
    ν::T
    ρ::T
    function tEVTail(ν::Real, ρ::Real)
        (ν > 0)     || throw(ArgumentError("ν must be > 0"))
        (-1 < ρ ≤ 1)|| throw(ArgumentError("ρ must be in (-1,1]"))
        ρ == 0 && return NoTail()
        ρ == 1 && return MTail()
        νT, ρT = promote(ν, ρ)
        return new{typeof(ρT)}(νT, ρT)
    end
end

const tEVCopula{T} = ExtremeValueCopula{2, tEVTail{T}}
tEVCopula(ν, ρ) = ExtremeValueCopula(2, tEVTail(ν, ρ))
tEVCopula(d::Integer, ν, ρ) = ExtremeValueCopula(2, tEVTail(ν, ρ))
Distributions.params(tail::tEVTail) = (ν = tail.ν, ρ = tail.ρ)

function A(tail::tEVTail, t::Real)
    ρ, ν = tail.ρ, tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    # log-ratios for stability
    log_t  = log(tt)
    log_om = log1p(-tt) # = log(1 - t)
    log_r  = log_t - log_om           # log(t/(1-t))
    log_s  = log_om - log_t           # log((1-t)/t)

    rα = exp(α * log_r)
    sα = exp(α * log_s)

    Z1 = C * (rα - ρ)
    Z2 = C * (sα - ρ)

    D = Distributions.TDist(ν + 1)
    F1 = Distributions.cdf(D, Z1)
    F2 = Distributions.cdf(D, Z2)

    return tt * F1 + om * F2
end

# Fitting helpers for EV copulas using extreme-t tail
_example(::Type{<:tEVCopula}, d) = ExtremeValueCopula(2, tEVTail(4.0, 0.5))
_unbound_params(::Type{<:tEVCopula}, d, θ) = [log(θ.ν), atanh(clamp(θ.ρ, -0.999999, 0.999999))]
_rebound_params(::Type{<:tEVCopula}, d, α) = (; ν = exp(α[1]), ρ = tanh(α[2]))
function dA(tail::tEVTail, t::Real)
    ρ, ν = tail.ρ, tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv(om)^2

    Z2  = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv(tt)^2)

    D = Distributions.TDist(ν + 1)
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)

    DB1 = tt * f1 * DZ1 + F1
    DB2 = om * f2 * DZ2 - F2
    return DB1 + DB2
end
function d²A(tail::tEVTail, t::Real)
    ρ, ν = tail.ρ, tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    rαm2  = exp((α - 2) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)
    sαm2  = exp((α - 2) * log_s)

    inv_om  = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t   = inv(tt)
    inv_t2  = inv_t^2
    inv_t3  = inv_t2 * inv_t
    inv_t4  = inv_t2^2

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    # d²Z1/dt² using product rule on r^(α-1) * (1-t)^(-2)
    DDZ1 = C * α * ( 2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4 )

    Z2  = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    # d²Z2/dt² with s = (1-t)/t, s'=-1/t², s''=2/t³
    DDZ2 = C * α * ( (α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3 )

    D = Distributions.TDist(ν + 1)
    f1 = Distributions.pdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)

    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2
    return DDB1 + DDB2
end
function _A_dA_d²A(tail::tEVTail, t::Real)
    ρ, ν = tail.ρ, tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    rαm2  = exp((α - 2) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)
    sαm2  = exp((α - 2) * log_s)

    inv_om  = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t   = inv(tt)
    inv_t2  = inv_t^2
    inv_t3  = inv_t2 * inv_t
    inv_t4  = inv_t2^2

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    DDZ1 = C * α * ( 2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4 )

    Z2  = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    DDZ2 = C * α * ( (α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3 )

    D = Distributions.TDist(ν + 1)
    
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)
    
    B1  = tt * F1
    DB1 = tt * f1 * DZ1 + F1
    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)
    
    B2  = om * F2
    DB2 = om * f2 * DZ2 - F2
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2

    A  = B1 + B2
    DA = DB1 + DB2
    DDA = DDB1 + DDB2
    return A, DA, DDA
end