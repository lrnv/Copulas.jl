"""
    ArchimaxCopula{G<:Generator, E<:ExtremeValueCopula}

Fields:
    - gen::G  : Archimedean generator ϕ (with methods `ϕ(gen, s)` and `ϕ⁻¹(gen, u)`)
    - evd::E  : Extreme-value copula (Pickands dependence function A)

Constructor

    ArchimaxCopula(A::ArchimedeanCopula{2,G}, E::ExtremeValueCopula) where {G<:Generator}
    ArchimaxCopula(gen::G, E::ExtremeValueCopula) where {G<:Generator}

Bivariate Archimax copula (convention used in `ArchimedeanCopula`). It is defined by

```math
C_{\\phi,A}(u_1,u_2)= \\phi\\!\\Big( (x+y)\\, A\\!\\big(\\tfrac{y}{x+y}\\big) \\Big), \\qquad x=\\phi^{-1}(u_1),\\; y=\\phi^{-1}(u_2).
```

Equivalent to evaluating the stable queuing function ``\\ell_A(x,y)=(x+y)A(x/(x+y))`` at ``(x,y)=(\\phi^{-1}(u_1), \\phi^{-1}(u_2))`` and apply ``\\phi``:

```julia
x = ϕ⁻¹(gen, u1);  y = ϕ⁻¹(gen, u2)
S = x + y
t = y / S
C(u1,u2) = ϕ(gen, S * A(evd, t))
```

Notes:

* If `A(t) ≡ 1`, it reduces to the Archimedean copula with generator `ϕ`.
* If `ϕ(s) = exp(-s)`, it reduces to the extreme value copula with Pickands function `A`.

`params(::ArchimaxCopula)` returns the concatenated tuple of parameters from `gen` and `evd`.

References:

* [caperaa2000](@cite) Capéraà, Fougères & Genest (2000), Bivariate Distributions with Given Extreme Value Attractor.
* [charpentier2014](@cite) Charpentier, Fougères & Genest (2014), Multivariate Archimax Copulas.
* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications.
"""
struct ArchimaxCopula{G<:Generator,E<:ExtremeValueCopula} <: Copula{2}
    gen::G            # archimedean ψ generator
    evd::E            # EVC (Pickands A maybe ℓ...)
end

ArchimaxCopula(A::ArchimedeanCopula{2,G}, E::ExtremeValueCopula) where {G<:Generator} =
    ArchimaxCopula(A.G, E)

_as_tuple(x) = x isa Tuple ? x : (x,)

Distributions.params(C::ArchimaxCopula) = (_as_tuple(Distributions.params(C.gen))..., _as_tuple(Distributions.params(C.evd))...)
# --- CDF ---
function Distributions.cdf(C::ArchimaxCopula, u::AbstractVector)
    @assert length(u) == 2
    u1, u2 = u
    (0.0 ≤ u1 ≤ 1.0 && 0.0 ≤ u2 ≤ 1.0) || return 0.0
    (u1 == 0.0 || u2 == 0.0) && return 0.0
    (u1 == 1.0 && u2 == 1.0) && return 1.0

    G = C.gen
    x = ϕ⁻¹(G, u1);  y = ϕ⁻¹(G, u2)
    S = x + y
    S == 0 && return 1.0
    t = _safett(y / S)                 # protect t≈0,1
    return ϕ(G, S * A(C.evd, t))
end

# --- log-PDF stable ---
function Distributions._logpdf(C::ArchimaxCopula, u::AbstractVector)
    T = promote_type(Float64, eltype(u))
    @assert length(u) == 2
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return T(-Inf)

    G = C.gen
    x = ϕ⁻¹(G, u1);  y = ϕ⁻¹(G, u2)
    S = x + y
    S > 0 || return T(-Inf)

    t   = _safett(y / S)
    A0  = A(C.evd,  t)
    A1  = dA(C.evd, t)
    A2  = d²A(C.evd,t)

    xu  = ϕ⁻¹⁽¹⁾(G, u1)          # < 0
    yv  = ϕ⁻¹⁽¹⁾(G, u2)          # < 0

    su  = xu * (A0 - t*A1)
    sv  = yv * (A0 + (1 - t)*A1)
    suv = - (xu*yv) * (t*(1 - t)/S) * A2

    s    = S * A0
    φp   = ϕ⁽¹⁾(G, s)            # < 0
    φpp  = ϕ⁽ᵏ⁾(G, Val(2), s)            # > 0

    base = su*sv + (φp/φpp)*suv
    base > 0 || return T(-Inf)
    return T(log(φpp) + log(base))
end

# --- Kendall τ: τ = τ_A + (1 - τ_A) τ_ψ ---
τ(C::ArchimaxCopula) = begin
    τA = τ(C.evd)
    τψ = τ(C.gen)
    τA + (1 - τA) * τψ
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula, x::AbstractVector{T}) where {T<:Real}
    v1, v2 = rand(rng, C.evd)

    M  = rand(rng, frailty(C.gen)) # frailty with LT = ϕ

    G = C.gen
    x[1] = ϕ(G, -log(v1)/M)
    x[2] = ϕ(G, -log(v2)/M)
    return x
end