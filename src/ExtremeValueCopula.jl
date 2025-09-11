"""
    ExtremeValueCopula{d, TT}

Constructor

    ExtremeValueCopula(d, tail::Tail)

Extreme-value copulas model tail dependence via a stable tail dependence function (STDF) ``\\ell`` or, equivalently,
via a Pickands dependence function ``A``. In any dimension ``d``, the copula cdf is

```math
\\displaystyle C(u) = \\\exp\\!\\left(-\\, \\\ell(-\\log u_1,\\ldots,-\\log u_d) \\right).
```

For ``d=2``, write ``x=-\\log u``, ``y=-\\log v``, ``s=x+y``, and ``t = x/s``. The relation between ``\\ell`` and ``A`` is

```math
\\ell(x,y) = s\\, A(t), \\qquad A:[0,1]\\to[1/2,1], \\quad A(0)=A(1)=1, \\ A \\text{ convex}.
```

Usage
- Provide any valid tail `tail::Tail` (which implements `A` and/or `ℓ`) to construct the copula.
- Sampling, cdf, and logpdf follow the standard `Distributions.jl` API.

Example
```julia
C = ExtremeValueCopula(2, GalambosTail(θ))
U = rand(C, 1000)
logpdf.(Ref(C), eachcol(U))
```

References:

* [gudendorf2010extreme](@cite) G., & Segers, J. (2010). Extreme-value copulas. In Copula Theory and Its Applications (pp. 127-145). Springer.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press.
* [mai2014financial](@cite) Mai, J. F., & Scherer, M. (2014). Financial engineering with copulas explained (p. 168). London: Palgrave Macmillan.
"""
struct ExtremeValueCopula{d, TT<:Tail} <: Copula{d}
    tail::TT
    function ExtremeValueCopula(d, tail::Tail)
        @assert _is_valid_in_dim(tail, d)
        return new{d, typeof(tail)}(tail)
    end
end
_cdf(C::ExtremeValueCopula{d, TT}, u) where {d, TT} = exp(-ℓ(C.tail, .- log.(u)))
Distributions.params(C::ExtremeValueCopula) = Distributions.params(C.tail)

#### Restriction to bivariate cases of the following methods: 
function Distributions._logpdf(C::ExtremeValueCopula{2, TT}, u) where {TT}
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return -Inf 
    # On the broder, the limit of the pdf is 0 and thus logpdf tends to -Inf
    (u1 == 1.0 || u2 == 1.0) && return -Inf
    x, y = -log(u1), -log(u2)
    val, du, dv, dudv = _biv_der_ℓ(C.tail, (x, y))
    core = -dudv + du*dv
    core ≤ 0 && return -Inf
    return -val + log(core) + x + y
end
τ(C::ExtremeValueCopula{2}) = QuadGK.quadgk(t -> d²A(C.tail, t) * t * (1 - t) / max(A(C.tail, t), _δ(t)), 0.0, 1.0)[1]
ρ(C::ExtremeValueCopula{2}) = 12 * QuadGK.quadgk(t -> 1 / (1 + A(C.tail, t))^2, 0.0, 1.0)[1] - 3
λᵤ(C::ExtremeValueCopula{2}) = 2 * (1 - A(C.tail, 0.5))
λₗ(C::ExtremeValueCopula{2}) =  A(C.tail, 0.5) > 0.5 ? 0.0 : 1.0

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, TT}, X::DenseMatrix{T}) where {T<:Real, TT}
    # More efficient Matrix sampler: 
    d,n = size(X)
    @assert d==2
    E = ExtremeDist(C.tail)
    for i in 1:n
        z = rand(rng, E)
        w = rand(rng) < _probability_z(C.tail, z) ? rand(rng) : rand(rng) * rand(rng)
        a = A(C.tail, z)
        X[1,i] = exp(log(w)*z/a)
        X[2,i] = exp(log(w)*(1-z)/a)
    end
    return X 
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, TT},
                              x::AbstractVector{T}) where {T<:Real, TT}
    u1, u2 = rand(rng), rand(rng)
    z  = rand(rng, ExtremeDist(C.tail))
    w  = (rand(rng) < _probability_z(C.tail, z)) ? u1 : (u1*u2)
    a  = A(C.tail, z)
    x[1] = exp(log(w)*z/a)
    x[2] = exp(log(w)*(1-z)/a)
    return x
end
DistortionFromCop(C::ExtremeValueCopula{2, TT}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) where TT = BivEVDistortion(C.tail, Int8(js[1]), float(uⱼₛ[1]))
