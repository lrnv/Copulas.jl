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
struct ExtremeValueCopula{d, TT<:Tail{d}} <: Copula{d}
    E::TT
    function ExtremeValueCopula(E::Tail{d}) where {d}
        return new{d, typeof(E)}(E)
    end
end

@inline _δ(t) = oftype(t, 1e-12)
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))

A(C::ExtremeValueCopula{2}, t::Real)  = A(C.E, t)
dA(C::ExtremeValueCopula{2}, t::Real) = ForwardDiff.derivative(z -> A(C.E, z), t)
d²A(C::ExtremeValueCopula{2}, t::Real) = ForwardDiff.derivative(z -> dA(C, z), t)

_A_dA_d²A(C::ExtremeValueCopula{2}, t::Real) = begin
    tt = _safett(t)
    (A(C, tt), dA(C, tt), d²A(C, tt))
end

ℓ(C::ExtremeValueCopula{2}, t₁::Real, t₂::Real) = begin
    s = t₁ + t₂
    s == 0 ? zero(promote_type(typeof(t₁), typeof(t₂))) : s * A(C, t₁ / s)
end

function _der_ℓ(C::ExtremeValueCopula{2}, u::Real, v::Real)
    s  = u + v
    x  = u / s
    y  = v / s
    a, da, d2a = _A_dA_d²A(C, x)
    val  = s * a
    du   = a + da * y
    dv   = a - x * da
    dudv = - x * y * d2a / s
    return val, du, dv, dudv
end

function Distributions.cdf(C::ExtremeValueCopula{2}, u1::Real, u2::Real)
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) ||
        throw(ArgumentError("u_i must be in (0,1]"))
    t1, t2 = -log(u1), -log(u2)
    return exp(-ℓ(C, t1, t2))
end
Distributions.cdf(C::ExtremeValueCopula{2}, u::NTuple{2,Real}) = Distributions.cdf(C, u[1], u[2])

function Distributions._logpdf(C::ExtremeValueCopula{2}, u::NTuple{2,Real})
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return -Inf
    # Borde: densidad en el límite es 0 → logpdf = -Inf
    (u1 == 1.0 || u2 == 1.0) && return -Inf
    x, y = -log(u1), -log(u2)
    val, du, dv, dudv = _der_ℓ(C, x, y)
    core = -dudv + du*dv
    core ≤ 0 && return -Inf
    return -val + log(core) + x + y
end

τ(C::ExtremeValueCopula{2}) = QuadGK.quadgk(t -> d²A(C, t) * t * (1 - t) / max(A(C, t), _δ(t)), 0.0, 1.0)[1]
ρ(C::ExtremeValueCopula{2}) = 12 * QuadGK.quadgk(t -> 1 / (1 + A(C, t))^2, 0.0, 1.0)[1] - 3

λᵤ(C::ExtremeValueCopula{2}) = 2 * (1 - A(C, 0.5))
function λₗ(C::ExtremeValueCopula{2})
    A(C, 0.5) > 0.5 ? 0.0 : 1.0
end

needs_binary_search(::ExtremeValueCopula{2}) = false

function probability_z(C::ExtremeValueCopula{2}, z::Real) 
    # p(z) = z(1-z) A''(z) / [ A(z) g_Z(z) ] 
    num = z * (1 - z) * d²A(C, z) 
    dem = A(C, z) * _pdf(ExtremeDist(C), z) # usa pdf, no _pdf 
    p = num / dem 
    return clamp(p, 0.0, 1.0) 
end

function Distributions._rand!(rng::Distributions.AbstractRNG,
                              C::ExtremeValueCopula{2},
                              x::AbstractVector{T}) where {T<:Real}
    @boundscheck length(x) ≥ 2 || throw(ArgumentError("x must have length ≥ 2"))
    u1, u2 = rand(rng), rand(rng)
    z  = rand(rng, ExtremeDist(C))
    p  = probability_z(C, z)
    w  = (rand(rng) < p) ? u1 : (u1*u2)
    a  = A(C, z)
    x[1] = w^( z / a )
    x[2] = w^((1 - z) / a)
    return x
end

DistortionFromCop(C::ExtremeValueCopula{2}, js::NTuple{1,Int},
                  uⱼₛ::NTuple{1,Float64}, ::Int) =
    BivEVDistortion(C, Int8(js[1]), float(uⱼₛ[1]))
