"""
    InvGaussianGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    InvGaussianGenerator(θ)
    InvGaussianCopula(d,θ)

The Inverse Gaussian copula in dimension ``d`` is parameterized by ``\\theta \\in [0,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = \\exp\\left( \\frac{1 - \\sqrt{1 + 2\\theta^{2} t}}{\\theta} \\right).
```

More details about Inverse Gaussian Archimedean copula are found in :

    Mai, Jan-Frederik, and Matthias Scherer. Simulating copulas: stochastic models, sampling algorithms, and applications. Vol. 6. # N/A, 2017. Page 74.

Special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct InvGaussianGenerator{T} <: AbstractUnivariateFrailtyGenerator
    θ::T
    function InvGaussianGenerator(θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative."))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const InvGaussianCopula{d, T}   = ArchimedeanCopula{d, InvGaussianGenerator{T}}
Distributions.params(G::InvGaussianGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:InvGaussianGenerator}, d, θ) = [log(θ.θ)]
_rebound_params(::Type{<:InvGaussianGenerator}, d, α) = (; θ = exp(α[1]))
_θ_bounds(::Type{<:InvGaussianGenerator}, d) = (0, Inf)

ϕ(  G::InvGaussianGenerator, t) = isinf(G.θ) ? exp(-sqrt(2*t)) : exp((1-sqrt(1+2*((G.θ)^(2))*t))/G.θ)
ϕ⁻¹(G::InvGaussianGenerator, t) = isinf(G.θ) ? log(t)^2/2 : ((1-G.θ*log(t))^(2)-1)/(2*(G.θ)^(2))
function ϕ⁽¹⁾(G::InvGaussianGenerator, t)
    if isinf(G.θ)
        s = sqrt(2*t)
        return -exp(-s) / s
    else
        θ = G.θ
        s = sqrt(1 + 2*(θ^2)*t)
        return -(θ / s) * ϕ(G, t)
    end
end

function ϕ⁽ᵏ⁾(G::InvGaussianGenerator, k::Int, t)
    k == 0 && return ϕ(G, t)
    k == 1 && return ϕ⁽¹⁾(G, t)
    # Closed-form via Faà di Bruno: ϕ^{(k)} = ϕ * Y_k(f', f'', ..., f^{(k)})
    # where f(t) = (1 - sqrt(1 + 2θ^2 t))/θ for finite θ, and f(t) = -sqrt(2t) for θ=∞.
    # f^{(n)}(t) = (-1)^n (2n-3)!! * A_n / S^{2n-1}, with
    #   finite θ: S = sqrt(1 + 2θ^2 t), A_n = θ^{2n-1}
    #   θ = ∞:    S = sqrt(2t),          A_n = 1

    # helper: double factorial (odd), with (-1)!! = 1
    ddfact(m::Int) = m <= 0 ? 1 : m * ddfact(m-2)

    if isinf(G.θ)
        S = sqrt(2*t)
        A = one(t) # placeholder; power handled per n
        # build x_m = f^{(m)}
        x = Vector{typeof(t)}(undef, k)
        @inbounds for n in 1:k
            coef = (-1)^n * ddfact(2n-3)
            x[n] = coef / (S^(2n-1))
        end
        # Bell polynomial Y_k via recurrence
        Y = zeros(eltype(x), k+1); Y[1] = one(t)  # Y_0 = 1
        @inbounds for n in 1:k
            acc = zero(t)
            for m in 1:n
                acc += binomial(n-1, m-1) * x[m] * Y[n-m+1]
            end
            Y[n+1] = acc
        end
        return ϕ(G, t) * Y[k+1]
    else
        θ = G.θ
        S = sqrt(1 + 2*(θ^2)*t)
        x = Vector{typeof(t)}(undef, k)
        @inbounds for n in 1:k
            coef = (-1)^n * ddfact(2n-3)
            x[n] = coef * (θ^(2n-1)) / (S^(2n-1))
        end
        Y = zeros(eltype(x), k+1); Y[1] = one(t)
        @inbounds for n in 1:k
            acc = zero(t)
            for m in 1:n
                acc += binomial(n-1, m-1) * x[m] * Y[n-m+1]
            end
            Y[n+1] = acc
        end
        return ϕ(G, t) * Y[k+1]
    end
end

function ϕ⁻¹⁽¹⁾(G::InvGaussianGenerator, t)
    if isinf(G.θ)
        return log(t) / t
    else
        θ = G.θ
        y = 1 - θ*log(t)
        return - y / (θ * t)
    end
end



function _invgaussian_tau(θ)
    T = promote_type(typeof(θ),Float64)
    θ == 0 && return zero(θ)
    θ > 1e153 && return 1/2
    θ < sqrt(eps(T)) && return zero(θ)
    function _integrand(x,θ)
        y = 1-θ*log(x)
        ret = - x*(y^2-1)/(2θ*y)
        return ret
    end
    rez, _ = QuadGK.quadgk(x -> _integrand(x,θ),zero(θ),one(θ))
    return 1+4*rez
end
τ(G::InvGaussianGenerator) = _invgaussian_tau(G.θ)
function τ⁻¹(::Type{<:InvGaussianGenerator}, τ)
    τ ≤ 0 && return zero(τ)
    τ ≥ 1/2 && return τ * Inf
    return Roots.find_zero(x -> _invgaussian_tau(x) - τ, (0, Inf))
end
frailty(G::InvGaussianGenerator) = Distributions.InverseGaussian(G.θ,1)


function _rho_invgaussian(θ; rtol=1e-7, atol=1e-9, maxevals=10^6)
    θeff = clamp(θ, 1e-12, Inf)
    Cθ   = Copulas.ArchimedeanCopula(2, InvGaussianGenerator(θeff))
    f(x) = _cdf(Cθ, (x[1], x[2]))  # <- tu _cdf
    I = HCubature.hcubature(f, (0.0,0.0), (1.0,1.0); rtol=rtol, atol=atol, maxevals=maxevals)[1]
    return 12I - 3
end

ρ(G::InvGaussianGenerator) = _rho_invgaussian(G.θ)
function ρ⁻¹(::Type{<:InvGaussianGenerator}, rho)
    # Numerically inverts _rho_invgaussian using Brent's method.
    # Spearman's rho for InvGaussian: [0, 1/2)
    rho ≤ 0 && return zero(rho)
    rho ≥ log(2) && return Inf * rho
    xhat = Roots.find_zero(x -> _rho_invgaussian(-log(x)) - rho, (0, 1))
    return -log(xhat)
end