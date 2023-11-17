"""
GumbelBarnettCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelBarnettCopula(d, θ)

The Gumbel-Barnett copula is an archimdean copula with generator:

```math
\\phi(t) = \\exp{θ^{-1}(1-e^{t})}, 0 \\leq \\theta \\leq 1.
```

More details about Gumbel-Barnett copula are found in:

    Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.437

It has a few special cases: 
- When θ = 0, it is the IndependentCopula
"""
struct GumbelBarnettCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function GumbelBarnettCopula(d,θ)
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end 
    end
end

ϕ(  C::GumbelBarnettCopula,       t) = exp((1-exp(t))/C.θ)
ϕ⁻¹(C::GumbelBarnettCopula,       t) = log(1-C.θ*log(t))
function τ(C::GumbelBarnettCopula)
    # Use a numerical integration method to obtain tau
    # result, _ = quadgk(x -> -((x-C.θ*x*log(x))*log(1-C.θ*ln(x))/C.θ), 0, 1)
    
    return 1+4*result
end
function τ⁻¹(::Type{GumbelBarnettCopula}, tau)
    if tau == zero(tau)
        return tau
    end
    
    # Define an anonymous function that takes a value x and computes τ 
    #for a GumbelBarnettCopula with θ = x
    τ_func(x) = τ(GumbelBarnettCopula(2,x))
    
    # Use the bisection method to find the root
    x = Roots.find_zero(x -> τ_func(x) - tau, (0.0, 1.0))    
    return x
end
