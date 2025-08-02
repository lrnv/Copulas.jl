"""
    AMHGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    AMHGenerator(θ)
    AMHCopula(d,θ)

The [AMH](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1,1)``. It is an Archimedean copula with generator :

```math
\\phi(t) = 1 - \\frac{1-\\theta}{e^{-t}-\\theta}
```

It has a few special cases:
- When θ = 0, it is the IndependentCopula

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct AMHGenerator{T} <: UnivariateGenerator
    θ::T
    function AMHGenerator(θ)
        if (θ < -1) || (θ > 1)
            throw(ArgumentError("Theta must be in [-1,1), you provided $θ."))
        elseif θ == 0
            return IndependentGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end

function _find_critical_value_amh(k; step=1e-7)
    # this function was used to define things in max_monotony below. 
    x = 0.0
    while x > -1
        if PolyLog.reli.(-k, x) <= 0
            x -= step
        else
            break
        end
    end
    return x
end
function max_monotony(G::AMHGenerator)
    G.θ >= 0 && return Inf        
    G.θ < sqrt(3)-2                && return 2  
    G.θ < -5+2sqrt(6)              && return 3   
    G.θ < -13/2 -sqrt(105)/2 +sqrt(2)/2 * sqrt(13sqrt(105)+135)     && return 4   
    G.θ < -14 - 3 * sqrt(15) + sqrt(6) * sqrt(14 * sqrt(15) + 55)     && return 5   
    G.θ < -0.00914869999999993     && return 6   
    G.θ < -0.004376199999998468    && return 7    
    G.θ < -0.002121400000000042    && return 8    
    G.θ < -0.0010375999999997928   && return 9     
    G.θ < -0.0005105999999999994   && return 10     
    G.θ < -0.00025240000000000527  && return 11     
    G.θ < -0.0001252000000000022   && return 12     
    G.θ < -6.220000000000067e-5    && return 13    
    G.θ < -3.099999999999991e-5    && return 14    
    G.θ < -1.5500000000000048e-5   && return 15     
    G.θ < -7.699999999999994e-6    && return 16    
    G.θ < -3.839999999999973e-6    && return 17
    G.θ < -1.9199999999999918e-6   && return 18
    G.θ < -9.600000000000008e-7    && return 19
    for k in 21:100
        G.θ < _find_critical_value_amh(k, step=1e-7) && return k-1
    end
    return 100
end

ϕ(  G::AMHGenerator, t) = (1-G.θ)/(exp(t)-G.θ)
ϕ⁻¹(G::AMHGenerator, t) = log(G.θ + (1-G.θ)/t)
ϕ⁽¹⁾(G::AMHGenerator, t) = -((1-G.θ) * exp(t)) / (exp(t) - G.θ)^2
ϕ⁽ᵏ⁾(G::AMHGenerator, ::Val{k}, t) where k = (-1)^k * (1 - G.θ) / G.θ * PolyLog.reli(-k, G.θ * exp(-t))
ϕ⁻¹⁽¹⁾(G::AMHGenerator, t) = (G.θ - 1) / (G.θ * (t - 1) * t + t)
williamson_dist(G::AMHGenerator, ::Val{d}) where d = G.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-G.θ),Val{d}()) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),Val{d}())

function τ(G::AMHGenerator)
    θ = G.θ
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
function τ⁻¹(::Type{T},tau) where T<:AMHGenerator
    if tau == zero(tau)
        return tau
    elseif tau > 1/3
        @info "AMHCopula cannot handle κ > 1/3."
        return one(tau)
    elseif tau < (5 - 8*log(2))/3
        @info "AMHCopula cannot handle κ < 5 - 8ln(2))/3 (approx -0.1817)."
        return -one(tau)
    end
    search_range = tau > 0 ? (0,1) : (-1,0)
    return Roots.find_zero(θ -> tau - τ(AMHGenerator(θ)), search_range)
end

function ρ(G::AMHGenerator)
    # Taken from https://cran.r-project.org/web/packages/copula/vignettes/rhoAMH-dilog.pdf
    a = G.θ
    if isnan(a)
        return a
    end
    aa = abs(a)
    if aa < 7e-16
        return a / 3
    elseif aa < 1e-4
        return a / 3 * (1 + a / 4)
    elseif aa < 0.002
        return a * (1/3 + a * (1/12 + a * 3/100))
    elseif aa < 0.007
        return a * (1/3 + a * (1/12 + a * (3/100 + a / 75)))
    elseif aa < 0.016
        return a * (1/3 + a * (1/12 + a * (3/100 + a * (1/75 + a / 147))))
    else
        term1 = 3 / a * (4 * (1 + 1 / a) * SpecialFunctions.spence(a))
        term2 = if a < 1
            8 * (1 / a - 1) * log1p(-a)
        else
            0.0
        end
        return term1 - term2 - (a + 12)
    end
end