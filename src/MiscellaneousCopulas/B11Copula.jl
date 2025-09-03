"""
    B11Copula{P}

Fields:

    - θ::Real - parameter
    
Constructor

    B11Copula(θ)

The bivariate B11 copula is parameterized by ``\\theta \\in [0,1]``. It is constructed as: 

```math
C(u_1, u_2) = \\theta \\min\\{u_1,u_2\\} + (1-\\theta)u_1u_2
```

It has a few special cases: 
- When θ = 0, it is the IndependentCopula
- When θ = 1, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [Joe1997] Joe, Harry, Multivariate Models and Multivariate Dependence Concepts, Chapman & Hall. 1997.
"""
struct B11Copula{P} <: Copula{2}
    θ::P  # Copula parameter
    function B11Copula(θ)
        if !(0 <= θ <= 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentCopula()
        elseif θ == 1
            return MCopula()
        else
            return new{typeof(θ)}(θ)
        end
    end
end

τ(C::B11Copula) = C.θ

function Distributions.cdf(C::B11Copula, x::AbstractVector)
    θ = C.θ
    u1, u2 = x
    return θ*(min(u1,u2)) + (1-θ)*u1*u2
end

function Distributions.pdf(C::B11Copula, x::AbstractVector)
    return (1 - C.θ)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::B11Copula, x::AbstractVector{T}) where {T<:Real}
    θ = C.θ
    v, w = rand(rng, Distributions.Uniform(0,1),2)
    b = rand(rng, Distributions.Bernoulli(θ))
    if b == 0
        x[1] = v
        x[2] = w
    else
        x[1] = min(v, w)
        x[2] = min(v, w)
    end
    return x
end