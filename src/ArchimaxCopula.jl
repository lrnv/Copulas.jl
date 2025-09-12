"""
    ArchimaxCopula{d, TG, TT}

Fields
- `gen::TG`  Archimedean generator ``\\phi`` (implements `ϕ`, `ϕ⁻¹`, derivatives)
- `tail::TT` Extreme-value tail (implements Pickands `A` / STDF `ℓ`)

Constructor

    ArchimaxCopula(d, gen::Generator, tail::Tail)

Definition (bivariate shown). Let ``x_i = ϕ^{-1}(u_i)`` and denote the STDF by ``\\ell``. The cdf is

```math
C(u_1,u_2) = ϕ\big( \\ell(x_1, x_2) \big).
```

Reductions
- If ``\\ell(x) = x_1 + x_2`` (i.e., `tail = NoTail()`), this is the Archimedean copula with generator `gen`.
- If ``ϕ(s) = e^{-s}`` (i.e., `gen = IndependentGenerator()`), this is the extreme-value copula with tail `tail`.

`params(::ArchimaxCopula)` concatenates the parameter tuples of `gen` and `tail`.

References:

* [caperaa2000](@cite) Capéraà, Fougères & Genest (2000), Bivariate Distributions with Given Extreme Value Attractor.
* [charpentier2014](@cite) Charpentier, Fougères & Genest (2014), Multivariate Archimax Copulas.
* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications.
"""
struct ArchimaxCopula{d, TG, TT} <: Copula{d}
    gen::TG
    tail::TT
    function ArchimaxCopula(d, gen::Generator, tail::Tail)
        @assert max_monotony(gen) >= d
        @assert _is_valid_in_dim(tail, d)
        return new{d, typeof(gen), typeof(tail)}(gen, tail)
    end
end
ArchimaxCopula(d, gen::Generator, ::NoTail) = ArchimedeanCopula(d, gen)
ArchimaxCopula(d, ::IndependentGenerator, tail::Tail) = ExtremeValueCopula(d, tail) 
Distributions.params(C::ArchimaxCopula) = (_as_tuple(Distributions.params(C.gen))..., _as_tuple(Distributions.params(C.tail))...)

function _cdf(C::ArchimaxCopula{d}, u) where {d}
    # Basic support checks
    T = eltype(u)
    any(iszero, u) && return T(0)
    all(isone, u) && return T(1)
    # Compute x_i = ϕ⁻¹(u_i), S = ∑ x_i, ω = x/S, then ϕ(S·A(ω))
    x = ϕ⁻¹.(C.gen, u)
    S = sum(x)
    S == 0 && return T(1)
    ω = ntuple(i -> x[i] / S, d)
    return ϕ(C.gen, S * A(C.tail, ω))
end
function Distributions._logpdf(C::ArchimaxCopula{d, TG, TT}, u) where {d, TG, TT}
    @inbounds for ui in u
        (0.0 < ui < 1.0) || return -Inf
    end
    val = _der(v -> Distributions.cdf(C, v), collect(u), ntuple(identity, d))
    return log(max(val, 0))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{d, TG, TT}, X::AbstractMatrix{T}) where {T<:Real, d, TG, TT}
    d == 2 && return @invoke Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, X)
    @assert size(X, 1) == d
    U = rand(rng, Distributions.Uniform(), d, size(X, 2))
    X .= inverse_rosenblatt(C, U)
    return X
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{d, TG, TT}, x::AbstractVector{T}) where {T<:Real, d, TG, TT}
    d == 2 && return @invoke Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, x)
    u = rand(rng, Distributions.Uniform(), d)
    x .= inverse_rosenblatt(C, u)
    return x
end






###### Special methods for the bivariate cases
function _cdf(C::ArchimaxCopula{2}, u)
    u1, u2 = u
    (0.0 ≤ u1 ≤ 1.0 && 0.0 ≤ u2 ≤ 1.0) || return 0.0
    (u1 == 0.0 || u2 == 0.0) && return 0.0
    (u1 == 1.0 && u2 == 1.0) && return 1.0

    x = ϕ⁻¹(C.gen, u1)
    y = ϕ⁻¹(C.gen, u2)
    S = x + y
    S == 0 && return one(eltype(u))
    t = _safett(y / S)                 # protect t≈0,1
    return ϕ(C.gen, S * A(C.tail, t))
end
function Distributions._logpdf(C::ArchimaxCopula{2, TG, TT}, u) where {TG, TT}
    T = promote_type(Float64, eltype(u))
    @assert length(u) == 2
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return T(-Inf)

    x = ϕ⁻¹(C.gen, u1)
    y = ϕ⁻¹(C.gen, u2)
    S = x + y
    S > 0 || return T(-Inf)

    t   = _safett(y / S)
    A0  = A(C.tail,  t)
    A1  = dA(C.tail, t)
    A2  = d²A(C.tail,t)

    xu  = ϕ⁻¹⁽¹⁾(C.gen, u1)          # < 0
    yv  = ϕ⁻¹⁽¹⁾(C.gen, u2)          # < 0

    su  = xu * (A0 - t*A1)
    sv  = yv * (A0 + (1 - t)*A1)
    suv = - (xu*yv) * (t*(1 - t)/S) * A2

    s    = S * A0
    φp   = ϕ⁽¹⁾(C.gen, s)            # < 0
    φpp  = ϕ⁽ᵏ⁾(C.gen, Val(2), s)            # > 0

    base = su*sv + (φp/φpp)*suv
    base > 0 || return T(-Inf)
    return T(log(φpp) + log(base))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, A::DenseMatrix{T}) where {T<:Real, TG, TT}
    evcop, frail = ExtremeValueCopula(2, C.tail), frailty(C.gen)
    Distributions._rand!(rng, evcop, A)
    F = zeros(eltype(A), size(A, 2))
    Distributions.rand!(rng, frail, F)
    A .= ϕ.(C.gen, -log.(A) ./ F') 
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, x::AbstractVector{T}) where {T<:Real, TG, TT}
    v1, v2 = rand(rng, ExtremeValueCopula(2, C.tail))
    M  = rand(rng, frailty(C.gen))
    x[1] = ϕ(C.gen, -log(v1)/M)
    x[2] = ϕ(C.gen, -log(v2)/M)
    return x
end
DistortionFromCop(C::ArchimaxCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) = BivArchimaxDistortion(C.gen, C.tail, Int8(js[1]), float(uⱼₛ[1]))
τ(C::ArchimaxCopula{2, TG, TT})  where {TG, TT} = begin
    τA = τ(ExtremeValueCopula(2, C.tail))
    τψ = τ(C.gen)
    τA + (1 - τA) * τψ
end



"""
    BB4Copula{T}

Fields:
    - θ::Real - dependence parameter (θ ≥ 0)
    - δ::Real - shape parameter (δ > 0)

Constructor

    BB4Copula(θ, δ)

The BB4 copula is a two-parameter [Archimax](@ref ArchimaxCopula) copula constructed from the Galambos tail and the Clayton generator. Its distribution function is

```math
C(u,v; \\theta, \\delta) = \\left( u^{-\\theta} + v^{-\\theta} - 1 - \\left[ (u^{-\\theta} - 1)^{-\\delta} + (v^{-\\theta} - 1)^{-\\delta} \\right]^{-1/\\delta} \\right)^{-1/\\theta}, \\quad \\theta ≥ 0, \\; \\delta > 0.
```
for ``0 ≤ u,v ≤ 1``.

Special cases:

* As δ → 0+, reduces to the Clayton Copula (Archimedean).
* As θ → 0+, reduces to the Galambos copula (extreme-value).
* As θ → ∞ or δ → ∞, approaches the M Copula.

References:

* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.197-198
"""
const BB4Copula{T} = ArchimaxCopula{2, ClaytonGenerator{T}, GalambosTail{T}}
BB4Copula(θ, δ) = ArchimaxCopula(2, ClaytonGenerator(θ), GalambosTail(δ))

function _cdf(C::BB4Copula{T}, u) where T
    θ, δ = C.gen.θ, C.tail.θ
    θ == 0 && return u1*u2
    
    u1, u2 = u
    uθ = exp(-θ*log(u1))
    vθ = exp(-θ*log(u2))
    a  = expm1(-θ*log(u1))              # = u1^{-θ} - 1  ≥ 0
    b  = expm1(-θ*log(u2))              # = u2^{-θ} - 1  ≥ 0
    x  = a^(-δ)
    y  = b^(-δ)
    s  = (x + y)^(-1/δ)
    r  = uθ + vθ - 1 - s                # [1 + a + b - (x+y)^{-1/δ}]
    return r^(-1/θ)
end
function Distributions._logpdf(C::BB4Copula{T}, u) where T
    Tret = promote_type(T, Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    θ, δ = C.gen.θ, C.tail.θ
    θ == 0 && return Tret(log(u1) + log(u2))

    uθ = exp(-θ*log(u1))
    vθ = exp(-θ*log(u2))
    a  = expm1(-θ*log(u1))              # u1^{-θ} - 1
    b  = expm1(-θ*log(u2))              # u2^{-θ} - 1
    x  = a^(-δ)
    y  = b^(-δ)
    S  = x + y
    sS = S^(-1/δ)                       # (x+y)^{-1/δ}
    Tm = uθ + vθ - 1 - sS
    (Tm > 0) || return Tret(-Inf)

    invδ = inv(δ)
    log_fac1 = (-1/θ - 2) * log(Tm)

    log_fac2 = (1 + invδ) * (log(x) + log(y)) + (-θ - 1) * (log(u1) + log(u2))

    invx, invy, invS = inv(x), inv(y), inv(S)
    p = a*invx - sS*invS
    q = b*invy - sS*invS
    term1 = (θ + 1) * p * q
    term2 = θ * (1 + δ) * (Tm) * S^(-invδ - 2)
    bracket = term1 + term2
    (bracket > 0) || return Tret(-Inf)

    logc = log_fac1 + log_fac2 + log(bracket)
    return Tret(logc)
end


"""
    BB5Copula{T}

Fields:
    - θ::Real - dependence parameter (θ ≥ 1)
    - δ::Real - shape parameter (δ > 0)

Constructor

    BB5Copula(θ, δ)

The BB5 copula is a two-parameter [Archimax](@ref ArchimaxCopula) copula, constructed from the Galambos tail and the Gumbel generator. Its distribution function is

```math
C(u,v; \\theta, \\delta) = \\exp\\Big\\{ -\\big[ x^{\\theta} + y^{\\theta} - (x^{-\\theta\\delta} + y^{-\\theta\\delta})^{-1/\\delta} \\big]^{1/\\theta} \\Big\\}, \\quad \\theta ≥ 1, \\; \\delta > 0,
```
where ``x = -\\log(u)`` and ``y = -\\log(v)``.

Special cases:

* As δ → 0⁺, reduces to the Gumbel copula (extreme-value and Archimedean).
* As θ = 1, reduces to the Galambos copula.
* As θ → ∞ or δ → ∞, converges to the M copula.

References:

* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.197-198
"""
const BB5Copula{T} = ArchimaxCopula{2, GumbelGenerator{T}, GalambosTail{T}}
BB5Copula(θ, δ) = ArchimaxCopula(2, GumbelGenerator(θ), GalambosTail(δ))
function _cdf(C::BB5Copula{T}, u) where T
    θ, δ = C.gen.θ, C.tail.θ
    u1, u2 = u
    x = -log(u1);  y = -log(u2) 
    logB = LogExpFunctions.logaddexp(-θ*δ*log(x), -θ*δ*log(y))
    H    = exp(-logB/δ)
    s    = exp(θ*log(x)) + exp(θ*log(y)) - H 
    f    = exp((1/θ)*log(s))
    return exp(-f)
end
function Distributions._logpdf(C::BB5Copula{T}, u) where T
    Tret = promote_type(T, Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    θ, δ = C.gen.θ, C.tail.θ
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