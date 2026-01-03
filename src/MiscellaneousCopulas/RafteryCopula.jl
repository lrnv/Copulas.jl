"""
    RafteryCopula{d, P}

Fields:
    - θ::Real - parameter

Constructor

    RafteryCopula(d, θ)

The multivariate Raftery copula of dimension d is parameterized by ``\\theta \\in [0,1]``.

```math
C_{\\theta}(\\mathbf{u}) = u_{(1)} + \\frac{(1 - \\theta)(1 - d)}{1 - \\theta - d} \\left(\\prod_{j=1}^{d} u_j\\right)^{\\frac{1}{1-\\theta}} - \\sum_{i=2}^{d} \\frac{\\theta(1-\\theta)}{(1-\\theta-i)(2-\\theta-i)} \\left(\\prod_{j=1}^{i-1}u_{(j)}\\right)^{\\frac{1}{1-\\theta}}u_{(i)}^{\\frac{2-\\theta-i}{1-\\theta}}
```

where ``u_{(1)}, \\ldots , u_{(d)}`` denote the order statistics of ``u_1, \\ldots ,u_d``. More details about Multivariate Raftery Copula are found in the references below.

Special cases:
- When θ = 0, it is the IndependentCopula.
- When θ = 1, it is the the Fréchet upper bound

References: 
* [Raftery2023](@cite) Saali, T., M. Mesfioui, and A. Shabri, 2023: Multivariate Extension of Raftery Copula. Mathematics, 11, 414, https://doi.org/10.3390/math11020414. 
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.6. 
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
            θ, _ = promote(θ, 1.0)
            return new{d,typeof(θ)}(θ)
        end
    end
    RafteryCopula{D, P}(d,θ) where {D, P} = RafteryCopula(d,θ)
end
Base.eltype(R::RafteryCopula) = eltype(R.θ)
Distributions.params(R::RafteryCopula) = (θ = R.θ,)
_example(::Type{<:RafteryCopula}, d) = RafteryCopula(d, 0.5)
_unbound_params(::Type{<:RafteryCopula}, d, θ) = [log(θ.θ) - log1p(-θ.θ)]
_rebound_params(::Type{<:RafteryCopula}, d, α) = (; θ = 1 / (1 + exp(-α[1])))
function _cdf(R::RafteryCopula{d,P}, u) where {d,P}
    # Order the vector u
    u_ordered = sort(u)
    term1 = u_ordered[1]
    term2 = (1 - R.θ) * (1 - d) / (1 - R.θ - d) * prod(u).^(1/(1 - R.θ))
    term3 = 0.0
    for i in 2:d
        prod_prev = one(eltype(u))
        for j in 1:i-1
            prod_prev *= u_ordered[j]
        end
        term3_part = R.θ * (1 - R.θ) / ((1 - R.θ - i) * (2 - R.θ - i)) * prod_prev^(1/(1 - R.θ)) * u_ordered[i]^((2 - R.θ - i) / (1 - R.θ))
        term3 += term3_part
    end
    return term1 + term2 - term3
end
function Distributions._logpdf(R::RafteryCopula{d,P}, u) where {d,P}
    u==zeros(d) && return eltype(u)(Inf)
    u==ones(d) && return (1-d) * log(1-R.θ)
    # Order the vector u
    u_ordered = sort(u)
    l_den = (d-1) * log(1-R.θ) + log(d + R.θ -1)
    l_num = log(d - 1 + R.θ * u_ordered[d]^((1 - R.θ - d) / (1 - R.θ)))
    l_prd = (R.θ) / (1 - R.θ) * log(prod(u))
    return l_num - l_den + l_prd
end
function Distributions._rand!(rng::Distributions.AbstractRNG, R::RafteryCopula{d,P}, x::AbstractVector{T}) where {d,P,T <: Real}
    # Step 1: Generate independent values u, u_1, ..., u_d from a uniform distribution [0, 1]
    u = rand(rng, d)

    # Step 2: Generate j from a Bernoulli distribution with parameter θ
    uj = (rand(rng)<R.θ) ? rand(rng) : 1

    # Step 3: Calculate v_1, ..., v_d
    for i in 1:d
        x[i] = u[i]^(1 - R.θ) * uj
    end
    
    return x
end
function ρ(R::RafteryCopula{d,P}) where {d, P}
    term1 = (d+1)*(2^d-(2-R.θ)^d)-(2^d*R.θ*d)
    term2 = (2-R.θ)^d*(2^d-d-1) 
    return term1/term2
end
function τ(R::RafteryCopula{d, P}) where {d, P}
    term1 = (2^(d-1) * factorial(d)) / ((2^(d-1)-1) * prod(i+1-R.θ for i in 2:d))
    term2 = ((1 - R.θ)^2 * (d^2 - 1)) / ((d-1+R.θ) * (d+1-R.θ) * (2^(d-1)-1))
    term3_sum = 0.0
    for k in 2:d
        term3_sum += (R.θ * (1-R.θ) * (2-R.θ)) / (2^k * factorial(k-1) * (1-R.θ-k) * (2-R.θ-k) * prod(i+1-R.θ for i in k:d))
    end
    term3 = (2^d * factorial(d)) / (2^(d-1)-1) * term3_sum
    
    term4 = 1 / (2^(d-1)-1)
    
    return term1 + term2 - term3 - term4
end

