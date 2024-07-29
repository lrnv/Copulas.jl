"""
    tEVCopula{P}

Fields:
    - ν::Real - paremeter
    - θ::Real - Parameter 
    
Constructor

    tEVCopula(ν, θ)

The bivariate extreme t copula is parameterized by ``\\nu \\in [0,\\infty)`` and \\theta \\in (-1,1]. It is an Extreme value copula with Pickands dependence function: 

```math
A(x) = xt_{\\nu+1}(Z_x) +(1-x)t_{\\nu+1}(Z_{1-x})
```
Where ``t_{\\nu + 1}``is the cumulative distribution function (CDF) of the standard t distribution with \\nu + 1 degrees of freedom and

```math
Z_x = \\frac{(1+\\nu)^{1/2}{\\sqrt{1-\\theta^2}}\\left [ \\left (\\frac{x}{1-x}  \\right )^{1/\\nu} - \\theta \\right ]
```

It has a few special cases:

- When θ = 0, it is the Independent Copula
- When θ = ∞, it is the M Copula (Upper Frechet-Hoeffding bound)

References:
* Extreme value properties of multivariate t copulas. Springer. 2008.
"""
struct tEVCopula{df, P} <: ExtremeValueCopula{P}
    ρ::P   # correlation paremeter
    ν::df  # degree of freedom

    function tEVCopula(ν::df, ρ::P) where {df<:Real, P<:Real}
        if ν <= 0
            throw(ArgumentError("The degrees of freedom ν must be positive real"))
        end
        if !(-1 < ρ <= 1)
            throw(ArgumentError("The correlation parameter ρ must be in (-1, 1]"))
        elseif ρ == 0
            return IndependentCopula(2)
        elseif ρ == 1
            return MCopula(2)
        end
        return new{df, typeof(ρ)}(ρ, ν)
    end
end
# #  specific ℓ funcion of Extreme t Copula
function ℓ(T::tEVCopula{P}, t::Vector) where P
    ρ = T.ρ
    ν = T.ν
    t₁, t₂ = t
    b = sqrt(ν + 1) / sqrt(1 - ρ^2)
    term1 = t₁ * StatsFuns.tdistcdf(ν + 1, b * ((t₁ / t₂)^(1 / ν) - ρ))
    term2 = t₂ * StatsFuns.tdistcdf(ν + 1, b * ((t₂ / t₁)^(1 / ν) - ρ))
    return term1 + term2
end
function z(T::tEVCopula, t)
    ρ = T.ρ
    ν = T.ν
    return ((1+ν)^(1/2))*((t/(1-t))^(1/ν) - ρ)*(1-ρ^2)^(-1/2)
end
# specific A funcion of Extreme t Copula
function A(T::tEVCopula, t::Real)
    ρ = T.ρ
    ν = T.ν
    t = clamp(t, 0, 1)
    zt = z(T,t)
    tt_minus = z(T,1-t)
    term1 = t * StatsFuns.tdistcdf(ν + 1, zt)
    term2 = (1-t) * StatsFuns.tdistcdf(ν + 1, tt_minus)
    return term1 + term2
end

function dA(C::tEVCopula, t::Real)
    h = 1e-5
    t_h_clamped = clamp(t - h, 0, 1)
    t_h_clamped_plus = clamp(t + h, 0, 1)
    dA_minus = A(C, t_h_clamped)
    dA_plus = A(C, t_h_clamped_plus)
    da = (dA_plus - dA_minus) / (2 * h)
    return da
end

# Approximation of the second derivative of A because the t distribution is not compatible with ForwarDiff
function d²A(C::tEVCopula, t::Real)
    h = 1e-5
    t_h_clamped = clamp(t - h, 0, 1)
    t_h_clamped_plus = clamp(t + h, 0, 1)
    dA_minus = dA(C, t_h_clamped)
    dA_plus = dA(C, t_h_clamped_plus)
    d2A = (dA_plus - dA_minus) / (2 * h)
    return d2A
end

# PDF function for ExtremeValueCopula using ℓ
function _pdf(C::tEVCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return c * (-D12 + D1 * D2) / (u[1] * u[2])
end
function Distributions._logpdf(C::tEVCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return log(c) + log(-D12 + D1 * D2) - log(u[1] * u[2])
end

function D_B_ℓ(C::tEVCopula, t::Vector{Float64}, B::Vector{Int})
    h = 1e-5
    if length(B) == 1
        # First partial derivative
        return partial_derivative_1(C, t, B[1], h)
    elseif length(B) == 2
        # Second partial derivative or mixed derivative
        return partial_derivative_2(C, t, B[1], B[2], h)
    else
        throw(ArgumentError("Higher order partial derivatives are not required for bivariate case"))
    end
end

function partial_derivative_1(C::tEVCopula, t::Vector{Float64}, i::Int, h::Float64)
    t_plus = copy(t)
    t_minus = copy(t)
    t_plus[i] += h
    t_minus[i] -= h
    
    return (ℓ(C, t_plus) - ℓ(C, t_minus)) / (2 * h)
end

function partial_derivative_2(C::tEVCopula, t::Vector{Float64}, i::Int, j::Int, h::Float64)
    if i == j
        # Second partial derivative
        t_plus = copy(t)
        t_minus = copy(t)
        t_plus[i] += h
        t_minus[i] -= h
        
        d_plus = partial_derivative_1(C, t_plus, i, h)
        d_minus = partial_derivative_1(C, t_minus, i, h)
        
        return (d_plus - d_minus) / (2 * h)
    else
        # Mixed derivative
        t_plus_plus = copy(t)
        t_plus_minus = copy(t)
        t_minus_plus = copy(t)
        t_minus_minus = copy(t)
        
        t_plus_plus[i] += h
        t_plus_plus[j] += h
        t_plus_minus[i] += h
        t_plus_minus[j] -= h
        t_minus_plus[i] -= h
        t_minus_plus[j] += h
        t_minus_minus[i] -= h
        t_minus_minus[j] -= h
        
        return (ℓ(C, t_plus_plus) - ℓ(C, t_plus_minus) - ℓ(C, t_minus_plus) + ℓ(C, t_minus_minus)) / (4 * h^2)
    end
end