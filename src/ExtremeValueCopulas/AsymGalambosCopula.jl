"""
    AsymGalambosCopula{P}

Fields:

  - α::Real - Dependency parameter
  - θ::Vector - Asymmetry parameters (size 2)

Constructor

    AsymGalambosCopula(α, θ)

The Asymmetric bivariate Galambos copula is parameterized by one dependence parameter ``\\alpha \\in [0, \\infty]`` and two asymmetry parameters ``\\theta_{i} \\in [0,1], i=1,2``. It is an Extreme value copula with Pickands function: 

```math
\\A(t) = 1 - ((\\theta_1t)^{-\\alpha}+(\\theta_2(1-t))^{-\\alpha})^{-\\frac{1}{\\alpha}} 
```

It has a few special cases:

- When α = 0, it is the Independent Copula
- When θ₁ = θ₂ = 0, it is the Independent Copula
- When θ₁ = θ₂ = 1, it is the Galambos Copula

References:
* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
struct AsymGalambosCopula{P} <: ExtremeValueCopula{P}
    α::P  # Dependence parameter
    θ::Vector{P}  # Asymmetry parameters (size 2)
    function AsymGalambosCopula(α::P, θ::Vector{P}) where {P}
        if length(θ) != 2
            throw(ArgumentError("The vector θ must have 2 elements for the bivariate case"))
        elseif !(0 <= α)
            throw(ArgumentError("The parameter α must be greater than or equal to 0"))
        elseif  !(0 <= θ[1] <= 1)  || !(0 <= θ[2] <= 1)  
            throw(ArgumentError("All parameters θ must be in the interval [0, 1]"))
        elseif α == 0 || (θ[1] == 0 && θ[2] == 0)
            return IndependentCopula(2)
        elseif θ[1] == 1 && θ[2] == 1
            return GalambosCopula(α)
        else
            T = promote_type(Float64, typeof(α), eltype(θ))
            return new{T}(T(α), T.(θ))
        end
    end
end

function A(C::AsymGalambosCopula, t::Real)
    x₁ = - C.α * log(C.θ[1] * t)
    x₂ = - C.α * log(C.θ[2] * (1-t))
    return -expm1(-LogExpFunctions.logaddexp(x₁,x₂) / C.α)
end

# First derivative of Pickands A(t) for Asymmetric Galambos (hand-derived, stable)
function dA(C::AsymGalambosCopula, t::Real)
    a = C.α
    θ1, θ2 = C.θ
    # Endpoint limits: A(t) ~ 1 - θ1 t near 0, and ~ 1 - θ2 (1 - t) near 1
    if t <= 0
        return -θ1
    elseif t >= 1
        return θ2
    end
    # Stable computation via log-sum-exp
    x1 = -a * log(θ1 * t)
    x2 = -a * log(θ2 * (1 - t))
    L = LogExpFunctions.logaddexp(x1, x2)           # log S with S = e^{x1} + e^{x2}
    e1 = exp(x1)
    e2 = exp(x2)
    invt = inv(t)
    inv1mt = inv(1 - t)
    N = e2 * inv1mt - e1 * invt                     # numerator
    F = exp(-(1 + 1/a) * L)                         # S^{-(1 + 1/a)}
    return F * N
end

# Second derivative of Pickands A(t) (hand-derived, stable)
function d²A(C::AsymGalambosCopula, t::Real)
    a = C.α
    θ1, θ2 = C.θ
    # Endpoint limits: second derivative tends to 0
    if t <= 0 || t >= 1
        return zero(promote_type(typeof(a), eltype(C.θ)))
    end
    x1 = -a * log(θ1 * t)
    x2 = -a * log(θ2 * (1 - t))
    L = LogExpFunctions.logaddexp(x1, x2)           # log S
    e1 = exp(x1)
    e2 = exp(x2)
    invt = inv(t)
    inv1mt = inv(1 - t)
    N = e2 * inv1mt - e1 * invt
    FinvS = exp(-L)                                  # 1/S
    F = exp(-(1 + 1/a) * L)                          # S^{-(1 + 1/a)}
    ap1 = a + 1
    term = e2 * inv1mt^2 + e1 * invt^2
    return F * ap1 * (term - N^2 * FinvS)
end