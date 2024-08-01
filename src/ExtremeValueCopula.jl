"""
    ExtremeValueCopula{P}

Fields:
    - P::Parameters: Parameters that define the copula.

Constructor:
    ExtremeValueCopula(P)

Represents a bivariate extreme value copula parameterized by `P`. Extreme value copulas are used to model the dependence structure between two random variables in the tails of their distribution, making them particularly useful in risk management, environmental studies, and finance.

In the bivariate case, an extreme value copula can be expressed as:

``math
C(u, v) = \\exp(-\\ell(\\log(u), \\log(v))).
``
where ``\\ell(\\cdot)`` is a tail dependence function associated with the bivariate extreme value copula. Furthermore, ``A(t)`` is a function ``A: [0, 1] \\to [0.5, 1] `` that is convex on the interval [0,1] and satisfies the boundary conditions ``A(0) = A(1) = 1``. This is denominated Pickands representation or Pickands function.

It is possible to relate these functions in the following way

``math
\\ell(u, v) = \\frac{u}{u+v}A\\left(\\frac{u}{u+v}\\right).
``


In this way, in order to define a bivariate copula of extreme values, it is only necessary to introduce the function \( A\).

A generic bivariate Extreme Values ​​copula can be constructed as follows:

```julia
struct GalambosCopula{P} <: ExtremeValueCopula{P}
A(C::GalambosCopula, t::Real) = 1 - (t^(-C.θ) + (1 - t)^(-C.θ))^(-1/C.θ) # You can define your own Pickands representation
param = 2.5
C = GalambosCopula(param)
```
The obtained model can be used as follows: 
```julia
samples = rand(C,1000)   # sampling
cdf(C,samples)           # cdf
pdf(C,samples)           # pdf
```

References:

* [gudendorf2010extreme](@cite) G., & Segers, J. (2010). Extreme-value copulas. In Copula Theory and Its Applications (pp. 127-145). Springer.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
* [mai2014financial](@cite) Mai, J. F., & Scherer, M. (2014). Financial engineering with copulas explained (p. 168). London: Palgrave Macmillan.
"""

abstract type ExtremeValueCopula{P} <: Copula{2} end

# Función genérica para A
function A(C::ExtremeValueCopula, t::Real)
    throw(ArgumentError("Function A must be defined for specific copula"))
end

function dA(C::ExtremeValueCopula, t::Real)
    ForwardDiff.derivative(t -> A(C, t), t)
end

function d²A(C::ExtremeValueCopula, t::Real)
    ForwardDiff.derivative(t -> dA(C, t), t)
end

function ℓ(C::ExtremeValueCopula, t::Vector)
    sumu = sum(t)
    vectw = t[1] / sumu
    return sumu * A(C, vectw)
end

# Función CDF para ExtremeValueCopula
function _cdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = abs.(log.(u)) # 0 <= u <= 1 so abs == neg, but return corectly 0 instead of -0 when u = 1. 
    return exp(-ℓ(C, t))
end

# Función genérica para calcular derivadas parciales de ℓ
function D_B_ℓ(C::ExtremeValueCopula, t::Vector{Float64}, B::Vector{Int})
    f = x -> ℓ(C, x)

    if length(B) == 1
        return ForwardDiff.gradient(f, t)[B[1]]
    elseif length(B) == 2
        return ForwardDiff.hessian(f, t)[B[1], B[2]]
    else
        throw(ArgumentError("Higher order partial derivatives are not required for bivariate case"))
    end
end

# Función PDF para ExtremeValueCopula usando ℓ
function _pdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return c * (-D12 + D1 * D2) / (u[1] * u[2])
end
function Distributions._logpdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = -log.(u)
    c = exp(-ℓ(C, t))
    D1 = D_B_ℓ(C, t, [1])
    D2 = D_B_ℓ(C, t, [2])
    D12 = D_B_ℓ(C, t, [1, 2])
    return log(c) + log(-D12 + D1 * D2) - log(u[1] * u[2])
end
# Definir la función para calcular τ
function τ(C::ExtremeValueCopula)
    integrand(x) = begin
        a = A(C, x)
        da = dA(C, x)
        return (x * (1 - x) / a) * da
    end
    
    integrate, _ = QuadGK.quadgk(integrand, 0.0, 1.0)
    return integrate
end

function ρₛ(C::ExtremeValueCopula)
    integrand(x) = 1 / (1 + A(C, x))^2
    
    integral, _ = QuadGK.quadgk(integrand, 0, 1)
    
    ρs = 12 * integral - 3
    return ρs
end
# Función para calcular el coeficiente de dependencia en el límite superior
function λᵤ(C::ExtremeValueCopula)
    return 2(1 - A(C, 0.5))
end

function λₗ(C::ExtremeValueCopula)
    if A(C, 0.5) > 0.5
        return 0
    else
        return 1
    end
end

function probability_z(C::ExtremeValueCopula, z)
    num = z*(1 - z)*d²A(C, z)
    dem = A(C, z)*_pdf(ExtremeDist(C), z)
    p = num / dem
    return clamp(p, 0.0, 1.0)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula, x::AbstractVector{T}) where {T<:Real}
    u1, u2 = rand(rng, Distributions.Uniform(0,1), 2)
    z = rand(rng, ExtremeDist(C))
    p = probability_z(C, z)
    if p < -eps() || p > eps()
        p = 0
    end
    c = rand(rng, Distributions.Bernoulli(p))
    w = 0
    if c == 1
        w = u1
    else
        w = u1*u2
    end
    a = A(C, z)
    x[1] = w^(z/a)
    x[2] = w^((1-z)/a)
    return x
end