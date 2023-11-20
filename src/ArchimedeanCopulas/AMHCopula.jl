"""
    AMHCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    AMHCopula(d, θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t}-\\theta}
```

It has a few special cases: 
- When θ = 0, it is the IndependentCopula
"""
struct AMHCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function AMHCopula(d,θ)
        if (θ < -1) || (θ >= 1)
            throw(ArgumentError("Theta must be in [-1,1)"))
        elseif θ == 0
            return IndependentCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::AMHCopula,t) = (1-C.θ)/(exp(t)-C.θ)
ϕ⁻¹(  C::AMHCopula,t) = log(C.θ + (1-C.θ)/t)

τ(C::AMHCopula) = _amh_tau_f(C.θ) # no closed form inverse...

function _amh_tau_f(θ)

    # unstable around zero, we instead cut its taylor expansion: 
    if abs(θ) < 0.01
        return 2/9  * θ
            + 1/18  * θ^2 
            + 1/45  * θ^3
            + 1/90  * θ^4
            + 2/315 * θ^5
            + 1/252 * θ^6
            + 1/378 * θ^7
            + 1/540 * θ^8
            + 2/1485 * θ^9
            + 1/990 * θ^10
    end
    if iszero(θ)
        return zero(θ)
    end
    u = isone(θ) ? θ : θ + (1-θ)^2 * log1p(-θ)
    return 1 - (2/3)*u/θ^2
end
function τ⁻¹(::Type{AMHCopula},tau)
    if tau == zero(tau)
        return tau
    elseif tau > 1/3
        @warn "AMHCopula cannot handle kendall tau's greater than 1/3. We capped it to 1/3."
        return one(τ)
    elseif tau < (5 - 8*log(2))/3
        @warn "AMHCopula cannot handle kendall tau's smaller than (5- 8ln(2))/3 (approx -0.1817). We capped it to this value."
        return -one(tau)
    end
    search_range = tau > 0 ? (0,1) : (-1,0)
    return Roots.find_zero(θ -> tau - _amh_tau_f(θ), search_range)
end
williamson_dist(C::AMHCopula{d,T}) where {d,T} = C.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-C.θ),d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(C,t),d)


