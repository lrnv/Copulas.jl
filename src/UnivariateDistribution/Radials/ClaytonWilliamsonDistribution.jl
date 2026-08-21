struct ClaytonWilliamsonDistribution{T<:Real} <: Distributions.ContinuousUnivariateDistribution
    θ::T # theta is negative here. 
    d::Int # d is an integer, at least 2. 
    function ClaytonWilliamsonDistribution(θ, d)
        if θ == -1/(d-1)
            return Distributions.Dirac(1)
        end
        @assert θ < 0
        @assert d > 1
        new{typeof(θ)}(θ, d)
    end
    ClaytonWilliamsonDistribution{T}(θ, d) where T = ClaytonWilliamsonDistribution(T(θ), d)
end
Base.minimum(D::ClaytonWilliamsonDistribution) = zero(D.θ)
Base.maximum(D::ClaytonWilliamsonDistribution) = -1/D.θ
@inline function _clayton_logchoose(a, k::Integer)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    result = zero(float(a))
    @inbounds for j in 0:(k - 1)
        result += log(a - j) - log(j + 1)
    end
    return result
end
function Distributions.cdf(D::ClaytonWilliamsonDistribution, x::Real)
    θ = D.θ
    d = D.d
    x <= 0 && return zero(x)
    α = -1/θ
    x >= α && return one(x)
    rez = zero(x)
    y = x/α
    for k in 0:(d-1)
        logterm = _clayton_logchoose(α, k) + k*log(y) + (α-k)*log1p(-y)
        rez += exp(logterm)
    end
    return 1-rez
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::ClaytonWilliamsonDistribution)
    u = rand(rng)
    Roots.find_zero(x -> (Distributions.cdf(d,x) - u), (0.0, Inf))
end
function Distributions.pdf(D::ClaytonWilliamsonDistribution, x::Real)
    return exp(Distributions.logpdf(D, x))
end
function Distributions.logpdf(D::ClaytonWilliamsonDistribution, x::Real)
    d = D.d
    α = -1/D.θ
    (0 ≥ x || x ≥ α) && return -Inf
    y = x/α
    return _clayton_logchoose(α - 1, d - 1) + (d-1)*log(y) + (α - d)*log1p(-y)
end
