"""
    tEVTail{Tdf,Tρ}

Fields:
  - ν::Real — degrees of freedom (ν > 0)
  - ρ::Real — correlation parameter (ρ ∈ (-1,1])

Constructor

    tEVCopula(ν, ρ)
    ExtremeValueCopula(tEVTail(ν, ρ))

The (bivariate) extreme-t copula is parameterized by ``\\nu > 0`` and \\rho \\in (-1,1]``.  
Its Pickands dependence function is

```math
A(x) = xt_{\\nu+1}(Z_x) +(1-x)t_{\\nu+1}(Z_{1-x})
```
Where ``t_{\\nu + 1}`` is the cumulative distribution function (CDF) of the standard t distribution with ``\\nu + 1`` degrees of freedom and

```math
Z_x = \\frac{(1+\\nu)^{1/2}{\\sqrt{1-\\theta^2}}\\left [ \\left (\\frac{x}{1-x} \\right )^{1/\\nu} - \\theta \\right ]
```

Special cases:

* ρ = 0 ⇒ IndependentCopula
* ρ = 1 ⇒ M Copula (upper Fréchet-Hoeffding bound)

References:

* [nikoloulopoulos2009extreme](@cite) Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of multivariate t copulas. Extremes, 12, 129-148.
"""
struct tEVTail{Tdf,Tρ} <: Tail{2}
    ν::Tdf
    ρ::Tρ
    function tEVTail(ν::Tdf, ρ::Tρ) where {Tdf<:Real, Tρ<:Real}
        (ν > 0)     || throw(ArgumentError("ν must be > 0"))
        (-1 < ρ ≤ 1)|| throw(ArgumentError("ρ must be in (-1,1]"))
        TdfT = promote_type(Tdf)
        TρT  = promote_type(Tρ)
        return new{TdfT,TρT}(TdfT(ν), TρT(ρ))
    end
end

const tEVCopula{Tdf,Tρ} = ExtremeValueCopula{2, tEVTail{Tdf,Tρ}}

function tEVCopula(ν::Real, ρ::Real)
    if ρ == 0
        return IndependentCopula(2)
    elseif ρ == 1
        return MCopula(2)
    else
        return ExtremeValueCopula(tEVTail(ν, ρ))
    end
end

Distributions.params(C::ExtremeValueCopula{2,tEVTail{Tdf,Tρ}}) where {Tdf,Tρ} = (C.E.ν, C.E.ρ)

function A(T::tEVTail, t::Real)
    ρ, ν = T.ρ, T.ν
    C = sqrt((1+ν)/(1-ρ^2))
    α = 1/ν

    Z1 =  C * ((t/(1-t))^α - ρ)
    Z2 = C * ((1/t - 1)^α - ρ)
    
    D = Distributions.TDist(T.ν + 1)
    F1 = Distributions.cdf(D, Z1)
    F2 = Distributions.cdf(D, Z2)
    
    B1 = t * F1
    B2 = (1-t) * F2
    A = B1 + B2
    return A
end
function dA(T::tEVTail, t::Real)
    ρ, ν = T.ρ, T.ν
    C = sqrt((1+ν)/(1-ρ^2))
    α = 1/ν

    Z1 =  C * ((t/(1-t))^α - ρ)
    DZ1 = C * α * (t/(1-t))^(α - 1) * (1/(1-t))^2
    
    Z2 = C * ((1/t - 1)^α - ρ)
    DZ2 = C * α * (1/t - 1)^(α - 1) * (-1/t^2)
    
    D = Distributions.TDist(T.ν + 1)
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)
    
    DB1 = t * f1 * DZ1 + F1
    DB2 = (1-t) * f2 * DZ2 - F2
    DA = DB1 + DB2
    return DA
end
function d²A(T::tEVTail, t::Real)
    ρ, ν = T.ρ, T.ν
    C = sqrt((1+ν)/(1-ρ^2))
    α = 1/ν

    Z1 =  C * ((t/(1-t))^α - ρ)
    DZ1 = C * α * (t/(1-t))^(α - 1) * (1/(1-t))^2
    DDZ1 = C * α * ((α-1)*t^(α-2) + 2t^(α-1)) / (1-t)^(α+2) 
    
    Z2 = C * ((1/t - 1)^α - ρ)
    DZ2 = C * α * (1/t - 1)^(α - 1) * (-1/t^2)
    DDZ2 = C * α * (
          (1/t - 1)^(α-2) * (α-1) * (1/t^4)
        + (1/t - 1)^(α-1) * (2/t^3)
    )
    
    D = Distributions.TDist(T.ν + 1)    
    f1 = Distributions.pdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)

    DDB1 = f1 * DZ1 + t * f1 * DDZ1 + t * g1 * f1 * DZ1^2 + f1 * DZ1
    DDB2 = (1-t) * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2
    DDA = DDB1 + DDB2
    return DDA
end
function _A_dA_d²A(T::tEVTail, t::Real)
    ρ, ν = T.ρ, T.ν
    C = sqrt((1+ν)/(1-ρ^2))
    α = 1/ν

    Z1 =  C * ((t/(1-t))^α - ρ)
    DZ1 = C * α * (t/(1-t))^(α - 1) * (1/(1-t))^2
    DDZ1 = C * α * ((α-1)*t^(α-2) + 2t^(α-1)) / (1-t)^(α+2)
    
    Z2 = C * ((1/t - 1)^α - ρ)
    DZ2 = C * α * (1/t - 1)^(α - 1) * (-1/t^2)
    DDZ2 = C * α * (
          (1/t - 1)^(α-2) * (α-1) * (1/t^4)
        + (1/t - 1)^(α-1) * (2/t^3)
    )
    
    D = Distributions.TDist(T.ν + 1)
    
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)
    
    
    B1 = t * F1
    DB1 = t * f1 * DZ1 + F1
    DDB1 = f1 * DZ1 + t * f1 * DDZ1 + t * g1 * f1 * DZ1^2 + f1 * DZ1
    
    B2 = (1-t) * F2
    DB2 = (1-t) * f2 * DZ2 - F2
    DDB2 = (1-t) * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2

    A = B1 + B2
    DA = DB1 + DB2
    DDA = DDB1 + DDB2
    return A, DA, DDA
end