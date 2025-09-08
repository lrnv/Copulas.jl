"""
    ExtremeValueCopula{P}

Fields:
    - P::Parameters: Parameters that define the copula.

Constructor:
    ExtremeValueCopula(P)

Represents a bivariate extreme value copula parameterized by `P`. Extreme value copulas are used to model the dependence structure between two random variables in the tails of their distribution, making them particularly useful in risk management, environmental studies, and finance.

In the bivariate case, an extreme value copula can be expressed as:

```math
C(u, v) = \\exp(-\\ell(\\log(u), \\log(v))).
```

where ``\\ell(\\cdot)`` is a tail dependence function associated with the bivariate extreme value copula. Furthermore, ``A(t)`` is a function ``A: [0, 1] \\to [0.5, 1] `` that is convex on the interval [0,1] and satisfies the boundary conditions ``A(0) = A(1) = 1``. This is denominated Pickands representation or Pickands function.

It is possible to relate these functions in the following way

```math
\\ell(u, v) = \\frac{u}{u+v}A\\left(\\frac{u}{u+v}\\right).
```


In this way, in order to define a bivariate copula of extreme values, it is only necessary to introduce the function ``A``.

A generic bivariate Extreme Values copula can be constructed as follows:

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
needs_binary_search(::ExtremeValueCopula) = false
A(C::ExtremeValueCopula, t::Real) = throw(ArgumentError("Function A must be defined for specific copula"))
dA(C::ExtremeValueCopula, t::Real) = ForwardDiff.derivative(t -> A(C,t), t)
d²A(C::ExtremeValueCopula, t::Real) = ForwardDiff.derivative(t -> dA(C,t), t)
_A_dA_d²A(C::ExtremeValueCopula, t::Real) = (A(C,t), dA(C,t), d²A(C,t)) #WilliamsonTransforms.taylor(x -> A(C, x), t, ::Val{2}())
ℓ(C::ExtremeValueCopula, t₁, t₂) = (t₁ +t₂) * A(C, t₁ /(t₁+t₂))
function _der_ℓ(C::ExtremeValueCopula, u, v) 
    x, y = u/(u+v), v/(u+v)
    a, da, d2a = _A_dA_d²A(C, x)
    val = (u+v)*a
    du = a + da * y
    dv = a - x * da
    dudv = - x * y * d2a / (u+v)
    return val, du, dv, dudv
end
# Función CDF para ExtremeValueCopula
function _cdf(C::ExtremeValueCopula, u::AbstractArray{<:Real})
    t = abs.(log.(u)) # 0 <= u <= 1 so abs == neg, but return corectly 0 instead of -0 when u = 1. 
    return exp(-ℓ(C, t...))
end
function Distributions._logpdf(C::ExtremeValueCopula, t::AbstractArray{<:Real})
    u, v = -log.(t)
    val, du, dv, dudv = _der_ℓ(C, u, v)
    return - val + log(max(-dudv + du*dv,0)) + u + v
end

# Definir la función para calcular τ and ρ
# Warning: the τ function can be veeeery unstable... it is actually very hard to compute correctly. 
# In some case, it simply fails by a lot. 
τ(C::ExtremeValueCopula) = QuadGK.quadgk(x -> d²A(C, x) * x * (1 - x) / A(C, x), 0.0, 1.0)[1]
ρ(C::ExtremeValueCopula) = 12 *  QuadGK.quadgk(x -> 1 / (1 + A(C, x))^2, 0, 1)[1] - 3

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
DistortionFromCop(C::ExtremeValueCopula, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) = BivEVDistortion(C, js[1], uⱼₛ[1])
