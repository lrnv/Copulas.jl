"""
    RafteryCopula{d, P}

Fields:
  - θ::Real - parameter

Constructor

    RafteryCopula(d, θ)

The Multivariate Raftery Copula of dimension d is arameterized by ``\\theta \\in [0,1]`` 

```math
C_{\\theta}(\\mathbf{u}) = u_{(1)} + \\frac{(1 - \\theta)(1 - d)}{1 - \\theta - d} \\left(\\prod_{j=1}^{d} u_j\\right)^{\\frac{1}{1-\\theta}} - \\sum_{i=2}^{d} \\frac{\\theta(1-\\theta)}{(1-\\theta-i)(2-\\theta-i)} \\left(\\prod_{j=1}^{i-1}u_{(j)}\\right)^{\\frac{1}{1-\\theta}}u_{(i)}^{\\frac{2-\\theta-i}{1-\\theta}}
```


where ``u_{(1)}, \\ldots , u_{(d)}`` denote the order statistics of ``u_1, \\ldots ,u_d``.

More details about Multivariate Raftery Copula are found in :

    Saali, T., M. Mesfioui, and A. Shabri, 2023: Multivariate Extension of Raftery Copula. Mathematics, 11, 414, https://doi.org/10.3390/math11020414.    
    
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.6.

It has a few special cases:
- When θ = 0, it is the IndependentCopula.
- When θ = 1, it is the the Fréchet upper bound
"""
struct RafteryCopula{d, P} <: Copula{d}
    θ::P  # Copula parameter
    function RafteryCopula(d,θ)
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ == 1
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
function _cdf(R::RafteryCopula, u::Vector{T}) where {T}
    if length(u) != R.d
        throw(ArgumentError("Dimension mismatch"))
    end
    
    # Order the vector u
    u_ordered = sort(u)
    
    term1 = u_ordered[1]
    term2 = (1 - R.θ) * (1 - R.d) / (1 - R.θ - R.d) * prod(u).^(1/(1 - R.θ))

    term3 = 0.0
    for i in 2:R.d
        prod_prev = prod(u_ordered[1:i-1])
        term3 += R.θ * (1 - R.θ) / ((1 - R.θ - i) * (2 - R.θ - i)) * prod_prev^(1/(1 - R.θ)) * u_ordered[i]^((2 - R.θ - i) / (1 - R.θ))
    end
    # Combine the terms to get the cumulative distribution function
    cdf_value = term1 + term2 - term3
    
    return cdf_value
end

function Distributions._logpdf(R::RafteryCopula, u::Vector{T}) where {T}
    if length(u) != R.d
        throw(ArgumentError("Dimension mismatch"))
    end
    
    # Order the vector u 
    u_ordered = sort(u)
    
    term_denominator = (1 - R.θ)^(R.d - 1) * (1 - R.θ - R.d)
    term_numerator = 1 - R.d - R.θ * u_ordered[R.d]^((1 - R.θ - R.d) / (1 - R.θ))
    term_product = prod(u)^((R.θ) / (1 - R.θ))
    
    logpdf_value = log(term_numerator) - log(term_denominator) + log(term_product)
    
    return logpdf_value
end

function Distributions._rand!(rng::Distributions.AbstractRNG, R::RafteryCopula{d,P}, x::AbstractVector{T}) where {d,P,T <: Real}
    
    dim = length(x)
    
    # Step 1: Generate independent values u, u_1, ..., u_d from a uniform distribution [0, 1]
    u = rand(rng, dim+1)
    
    # Step 2: Generate j from a Bernoulli distribution with parameter θ
    j = rand(Bernoulli(R.θ),1)
    uj = u[1].^j
    # Step 3: Calculate v_1, ..., v_d
    for i in 2:dim+1
        x[i] = u[i]^(1 - R.θ) * uj
    end
    
    return x
end

function ρ(R::RafteryCopula{d,R}) where {d, P}
    term1 = (d+1)*(2^d-(2-R.θ)^d)-(2^d*R.θ*d)
    term2 = (2-R.θ)^d*(2^d-d-1) 
    return term1/term2
end