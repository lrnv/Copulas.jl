"""
    GumbelBarnettGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelBarnettGenerator(θ)
    GumbelBarnettCopula(d,θ)

The Gumbel-Barnett copula is an archimdean copula with generator:

```math
\\phi(t) = \\exp{θ^{-1}(1-e^{t})}, 0 \\leq \\theta \\leq 1.
```

It has a few special cases:
- When θ = 0, it is the IndependentCopula

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.437
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GumbelBarnettGenerator{T} <: Generator
    θ::T
    function GumbelBarnettGenerator(θ)
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const GumbelBarnettCopula{d, T} = ArchimedeanCopula{d, GumbelBarnettGenerator{T}}
GumbelBarnettCopula(d, θ) = ArchimedeanCopula(d, GumbelBarnettGenerator(θ))
Distributions.params(G::GumbelBarnettGenerator) = (G.θ,)

function max_monotony(G::GumbelBarnettGenerator)
    G.θ == 0 && return Inf
    G.θ > 0.380 && return 2
    G.θ > 0.216 && return 3
    G.θ > 0.145 && return 4
    G.θ > 0.106 && return 5
    G.θ > 0.082 && return 6
    G.θ > 0.066 && return 7
    G.θ > 0.055 && return 8
    G.θ > 0.046 && return 9

    n = 10
    
    # if more is needed, this value can be increased. 
    MAX = 1e3 # we look until 1000, with E(1000) \approx 20 000 so \theta < 0.00005
    while n <= MAX
        C2 = binomial(n, 2)
        C3 = binomial(n, 3)
        C4 = binomial(n, 4)
        term1 = C2 / n
        discr = C2^2 - (2n / (n - 1)) * (C3 + 3*C4)
        term2 = (n - 1)/n * sqrt(discr)
        lower_bound = (term1 + term2) #lowerbound of the leftmost root of touchards polynomials. 
        if G.θ > 1/lower_bound
            return n
        else
            n+=1
        end
    end
    return MAX 
end

ϕ(G::GumbelBarnettGenerator, t) = exp((1 - exp(t)) / G.θ)
ϕ⁽¹⁾(G::GumbelBarnettGenerator, t) = -exp((1 - exp(t)) / G.θ) * exp(t) / G.θ
ϕ⁻¹(G::GumbelBarnettGenerator, t) = log1p(-G.θ * log(t))
ϕ⁻¹⁽¹⁾(G::GumbelBarnettGenerator, t) = -G.θ / (t - G.θ * t * log(t))
function ϕ⁽ᵏ⁾(G::GumbelBarnettGenerator, ::Val{k}, t) where k
    α = 1/G.θ    
    C = -α*exp(t)
    R = C * exp(α + C)
    k == 1 && return R
    return evalpoly(C, ntuple(i->Combinatorics.stirlings2(k, i), k)) * R
end


# See this htread ;: https://discourse.julialang.org/t/solving-for-transcendental-equation/131229/16


# function lower_bound_on_leftmost_root(::Val{n}) where n
#     n==2  && return - 1.0
#     n==3  && return - 1 / 0.380
#     n==4  && return - 1 / 0.216
#     n==5  && return - 1 / 0.145
#     n==6  && return - 1 / 0.106
#     n==7  && return - 1 / 0.082
#     n==8  && return - 1 / 0.066
#     n==9  && return - 1 / 0.055
#     n==10 && return - 1 / 0.046
#     C2 = binomial(n, 2)
#     C3 = binomial(n, 3)
#     C4 = binomial(n, 4)
#     term1 = C2 / n
#     discr = C2^2 - (2n / (n - 1)) * (C3 + 3*C4)
#     term2 = (n - 1)/n * sqrt(discr)
#     lower_bound = (term1 + term2)
#     return -lower_bound
# end
# leftmost_critical_point(::Val{n}) where n = lower_bound_on_leftmost_root(Val{n+1}())
# starting_point(::Val{n}) where n = leftmost_critical_point(Val{n+1}())
# last_summit(::Val{n}) where n = _fₙ(leftmost_critical_point(Val{n}()), Val{n}())

# function _fₙ(x, ::Val{n}, s2::NTuple{n, Int}=ntuple(i->stirlings2(n, i), n)) where n
#     n==1 && return exp(x)
#     return x*evalpoly(x, s2) * exp(x)
# end
# function _inv_fₙ(x, ::Val{n}) where n
#     n==1 && return log(x)
#     x₀ = starting_point(Val{n}())
#     return find_zero(
#       let s2 = ntuple(i->stirlings2(n, i), n)
#           t -> _fₙ(t, Val{n}(), s2)-x
#       end, 
#       x₀
#       )
# end
# function ϕ⁽ᵏ⁾⁻¹(G::GumbelBarnettGenerator, ::Val{k}, t; start_at=t) where k
#     @show k, t
#     return log(- G.θ * _inv_fₙ(t * exp(-1/G.θ), Val{k}()))
# end


function _gumbelbarnett_tau(θ)
    iszero(θ) && return θ
    r, _ = QuadGK.quadgk(x -> (1-θ*log(x))  * log1p(-θ*log(x)) * x, 0, 1)
    return 1-4*r/θ
end
τ(G::GumbelBarnettGenerator) = _gumbelbarnett_tau(G.θ)
function τ⁻¹(::Type{T}, tau) where T<:GumbelBarnettGenerator
    if tau == 0
        return zero(tau)
    elseif tau > 0
        @info "GumbelBarnettCopula cannot handle τ > 0."
        return zero(tau)
    elseif tau < -0.3612
        @info "GumbelBarnettCopula cannot handle τ <≈ -0.3613."
        return one(tau)
    end
    # Use the bisection method to find the root
    return Roots.find_zero(θ -> _gumbelbarnett_tau(θ) - tau, (0.0, 1.0))
end