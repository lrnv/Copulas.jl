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
    function RafteryCopula{d}(θ) where {d}
        if (θ < 0) || (θ > 1)
            throw(ArgumentError("Theta must be in [0,1]"))
        elseif θ == 0
            return IndependentCopula{d}()
        elseif θ == 1
            return MCopula{d}()
        else
            θ, _ = promote(θ, 1.0)
            return new{d,typeof(θ)}(θ)
        end
    end
end
RafteryCopula(d, θ) = RafteryCopula{d}(θ)
(::Type{<:RafteryCopula{D,P}})(d::Int, θ) where {D,P} = RafteryCopula{d}(θ)
Base.eltype(R::RafteryCopula) = eltype(R.θ)
Distributions.params(R::RafteryCopula) = (θ = R.θ,)
_example(::Type{<:RafteryCopula}, d) = RafteryCopula(d, 0.5)
_unbound_params(::Type{<:RafteryCopula}, d, θ) = [LogExpFunctions.logit(θ.θ)]
_rebound_params(::Type{<:RafteryCopula}, d, α) = (; θ = LogExpFunctions.logistic(α[1]))
function _cdf(R::RafteryCopula{d,P}, u) where {d,P}
    # Order the vector u
    u_ordered = sort(u)
    term1 = u_ordered[1]
    term2 = (1 - R.θ) * (1 - d) / (1 - R.θ - d) * prod(u).^(1/(1 - R.θ))
    term3 = 0.0
    for i in 2:d
        prod_prev = prod(u_ordered[1:i-1])
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
function Distributions._rand!(rng::Distributions.AbstractRNG, R::RafteryCopula{d,P}, A::AbstractMatrix{T}) where {d,P,T <: Real}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    Random.rand!(rng, A)
    common = rand(rng, T, size(A, 2))
    selectors = rand(rng, T, size(A, 2))
    @inbounds for (j, col) in enumerate(axes(A, 2))
        uj = selectors[j] < R.θ ? common[j] : one(T)
        for row in axes(A, 1)
            A[row, col] = A[row, col]^(one(T) - R.θ) * uj
        end
    end
    return A
end
function ρ(R::RafteryCopula{d,P}) where {d, P}
    T = typeof(float(R.θ))
    θ = T(R.θ)
    inv2d = exp2(-T(d))
    scaled_power = (one(T) - θ / 2)^d
    numerator = T(d + 1) * (one(T) - scaled_power) - θ * d
    denominator = scaled_power * (one(T) - T(d + 1) * inv2d)
    return numerator * inv2d / denominator
end
function τ(R::RafteryCopula{d, P}) where {d, P}
    T = typeof(float(R.θ))
    θ = T(R.θ)
    pow2 = exp2(T(1 - d))
    normalization = inv(one(T) - pow2)
    common = θ * (one(T) - θ) * (T(2) - θ)

    ratio = one(T)
    term3 = zero(T)
    for k in d:-1:2
        ratio *= T(k) / (T(k + 1) - θ)
        term3 += common * exp2(T(1 - k)) * normalization * ratio /
                 ((one(T) - θ - k) * (T(2) - θ - k))
    end

    term1 = normalization * ratio
    term2 = (one(T) - θ)^2 * (T(d)^2 - one(T)) * pow2 * normalization /
            ((T(d - 1) + θ) * (T(d + 1) - θ))
    term4 = pow2 * normalization
    return term1 + term2 - term3 - term4
end
