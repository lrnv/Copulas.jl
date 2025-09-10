"""
    ArchimaxCopula{G<:Generator, E<:Tail}

Fields:
    - gen::G  : Archimedean generator ϕ (with methods `ϕ(gen, s)` and `ϕ⁻¹(gen, u)`)
    - tail::E  : Extreme-value copula (Pickands dependence function A)

Constructor

    ArchimaxCopula(gen::Generator, tail::Tail)

Bivariate Archimax copula (convention used in `ArchimedeanCopula`). It is defined by

```math
C_{\\phi,\\ell}(u_1,u_2)= \\phi\\!\\Big( \\ell\\!\\Big( \\phi^{-1}(u_i), i \\in 1,...d \\Big) \\Big).
```

Notes:

* If `ℓ(x) = sum(x)`, it reduces to the Archimedean copula with generator `ϕ`.
* If `ϕ(s) = exp(-s)`, it reduces to the extreme value copula with stable tail dependence function `ℓ`

`params(::ArchimaxCopula)` returns the concatenated tuple of parameters from `gen` and `tail`.

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

# --- CDF ---
function Distributions.cdf(C::ArchimaxCopula{2}, u::AbstractVector)
    @assert length(u) == 2
    u1, u2 = u
    (0.0 ≤ u1 ≤ 1.0 && 0.0 ≤ u2 ≤ 1.0) || return 0.0
    (u1 == 0.0 || u2 == 0.0) && return 0.0
    (u1 == 1.0 && u2 == 1.0) && return 1.0

    x = ϕ⁻¹(C.gen, u1)
    y = ϕ⁻¹(C.gen, u2)
    S = x + y
    S == 0 && return one(eltype(u))
    t = _safett(y / S)                 # protect t≈0,1
    return ϕ(C.gen, S * A(C.evd, t))
end

# --- log-PDF stable ---
function Distributions._logpdf(C::ArchimaxCopula{2}, u::AbstractVector)
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

# --- Kendall τ: τ = τ_A + (1 - τ_A) τ_ψ ---
τ(C::ArchimaxCopula) = begin
    τA = τ(C.tail)
    τψ = τ(C.gen)
    τA + (1 - τA) * τψ
end


# Use the matrix sampler for better efficiency
# (if not working, maybe uncomment the vetor version ?)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, x::A::DenseMatrix{T}) where {T<:Real, TG, TT}
    evd, frail = ExtremeValueCopula(C.tail), frailty(C.gen)
    Distributions._rand!(rng, evd, x)
    F = rand(rng, frail, size(A, 2))
    x ./= F'
    return x
end
# function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimaxCopula{2, TG, TT}, x::AbstractVector{T}) where {T<:Real, TG, TT}
#     v1, v2 = rand(rng, ExtremeValueCopula(C.tail))
#     M  = rand(rng, frailty(C.gen))
#     x[1] = ϕ(C.gen, -log(v1)/M)
#     x[2] = ϕ(C.gen, -log(v2)/M)
#     return x
# end
