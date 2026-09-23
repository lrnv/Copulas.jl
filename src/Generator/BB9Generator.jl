"""
    BB9Generator(ϑ, δ)
    BB9Copula{d}(ϑ, δ)
    BB9Copula(d, ϑ, δ)

The BB9 copula has parameters ``\\vartheta \\in [1,\\infty)`` and ``\\delta \\in (0, \\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = \\exp(-(\\delta^{-\\vartheta} + t)^{\\frac{1}{\\vartheta}} + \\delta^{-1}),
```

The corner `ϑ = 1` is independence, while `δ → ∞` approaches Gumbel with
parameter `ϑ`. A large fitted `δ` can therefore leave the second parameter
weakly identified; log-scale evaluation is preferable for extreme values.

See also: [`GumbelGenerator`](@ref), [`ArchimedeanCopula`](@ref),
[Distributions `logpdf`](@extref Distributions Probability-evaluation),
[`Distributions.fit`](@ref).

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.205-206
"""
BB9Generator, BB9Copula

Paramorph.@paramorph T struct BB9Generator{T<:Real} <: AbstractFrailtyGenerator
    θ::T ~ closed_lower(one(T))
    δ::T ~ asℝ₊
end
function BB9Generator(θ::Real, δ::Real)
    T = promote_type(typeof(float(θ)), typeof(float(δ)))
    return BB9Generator{T}(T(θ), T(δ))
end

const BB9Copula{d, T} = ArchimedeanCopula{d, BB9Generator{T}}
function (::Type{BB9Copula{d}})(args...; kwargs...) where {d}
    return _wrap_archimedean(Val(d), BB9Generator(args...; kwargs...))
end
(::Type{BB9Copula})(d::Int, args...; kwargs...) = BB9Copula{d}(args...; kwargs...)

ϕ(  G::BB9Generator, s) = begin
    a  = inv(G.θ)
    c  = G.δ^(-G.θ)
    exp(inv(G.δ) - (s + c)^a)
end
ϕ⁻¹(G::BB9Generator, t) = (inv(G.δ) - log(t))^(G.θ) - G.δ^(-G.θ)

function ϕ⁽¹⁾(G::BB9Generator, s)
    a  = inv(G.θ);  c = G.δ^(-G.θ)
    ϕ(G,s) * ( -a * (s + c)^(a-1) )
end
function ϕ⁽ᵏ⁾(G::BB9Generator, k::Int, s::Real)
    if k==2
        a  = inv(G.θ);  c = G.δ^(-G.θ)
        φ  = ϕ(G,s)
        t  = s + c
        φ * ( a^2 * t^(2a-2) - a*(a-1) * t^(a-2) )
    end
    return @invoke ϕ⁽ᵏ⁾(G::Generator, k, s)
    # k == 0 && return ϕ(G, s)
    # a, c = inv(G.θ), G.δ^(-G.θ)
    # T = promote_type(typeof(a), typeof(s))
    # xs = [-prod(a - i for i in 0:j-1) * (s + c)^(a - j) for j in 1:k]
    # B = ones(T, k + 1)
    # for n in 1:k
    #     B[n + 1] = sum(binomial(n - 1, j - 1) * xs[j] * B[n - j + 1] for j in 1:n)
    # end    
    # return ϕ(G, s) * B[end]
end
function ϕ⁽ᵏ⁾⁻¹(G::BB9Generator, k::Int, t; start_at=t)
    k == 1 || return @invoke ϕ⁽ᵏ⁾⁻¹(G::Generator, k, t; start_at=start_at)
    T = float(promote_type(typeof(t), typeof(G.θ), typeof(G.δ)))
    target = T(t)
    iszero(target) && return T(Inf)
    θ = T(G.θ)
    θ == one(T) && return -log(-target)

    δ = T(G.δ)
    θm1 = θ - one(T)
    logscaled = log(-θ * target) - inv(δ)
    logarg = -logscaled / θm1 - log(θm1)
    z = θm1 * _lambertw_exp(logarg)
    return max(zero(T), exp(θ * log(z)) - δ^(-θ))
end
ϕ⁻¹⁽¹⁾(G::BB9Generator, t) = -G.θ * (inv(G.δ) - log(t))^(G.θ - 1) / t

frailty(G::BB9Generator) =  TiltedPositiveStable(inv(G.θ), G.δ^(-G.θ))
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB9Generator}
    θ, δ = C.G.θ, C.G.δ
    x = inv(δ) - log(u[1])
    y = inv(δ) - log(u[2])
    c = δ^(-θ)
    A = (x^θ + y^θ - c)^(1/θ)
    return exp(inv(δ) - A)
end

function Distributions._logpdf(C::ArchimedeanCopula{2,BB9Generator{TF}}, u) where {TF}
    T = promote_type(TF, eltype(u))
    (0.0 < u[1] ≤ 1.0 && 0.0 < u[2] ≤ 1.0) || return T(-Inf)

    θ, δ = C.G.θ, C.G.δ
    x = inv(δ) - log(u[1])
    y = inv(δ) - log(u[2])
    S = x^θ + y^θ - δ^(-θ)
    S ≤ 0 && return T(-Inf)

    A = S^(1/θ)
    logGbar = inv(δ) - A
    logc = logGbar +
           (1/θ - 2)*log(S) +
           log(A + θ - 1) +
           (θ - 1)*(log(x) + log(y)) -
           (log(u[1]) + log(u[2]))

    return T(logc)
end
