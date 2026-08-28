struct ExtremeDist{C} <: Distributions.ContinuousUnivariateDistribution
    tail::C
end

Base.minimum(::ExtremeDist) = 0
Base.maximum(::ExtremeDist) = 1

function Distributions.cdf(d::ExtremeDist, z::Real)
    z <= 0 && return zero(float(z))
    z >= 1 && return one(float(z))
    return z + z * (1 - z) * (dA(d.tail, z) / A(d.tail, z))
end

function Distributions.pdf(d::ExtremeDist, z::Real)
    (z <= zero(z) || z >= one(z)) && return zero(float(z))
    A, A1, A2 = _A_dA_d²A(d.tail, z)
    return 1 + (1 - 2z) * A1 / A + z * (1 - z) * (A2 * A - A1^2) / A^2
end

function Distributions.logpdf(d::ExtremeDist, z::Real)
    f = Distributions.pdf(d, z)
    f > zero(f) || return oftype(f, -Inf)
    return log(f)
end

function Distributions.quantile(d::ExtremeDist, p)
    return _unit_quantile(d, p)
end

# Generate random samples from the radial distribution using the quantile function
Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist) = Distributions.quantile(d, rand(rng))
