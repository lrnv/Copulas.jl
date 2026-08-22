struct ExtremeDist{C} <: Distributions.ContinuousUnivariateDistribution
    tail::C
end

function Distributions.cdf(d::ExtremeDist, z::Real)
    z <= zero(z) && return zero(float(z))
    z >= one(z)  && return one(float(z))
    return z + z * (1 - z) * (dA(d.tail, z) / A(d.tail, z))
end

function _pdf(d::ExtremeDist, z::Real)
    (z <= zero(z) || z >= one(z)) && return zero(float(z))
    Aval, A1, A2 = _A_dA_d²A(d.tail, z)
    return 1 + (1 - 2z) * A1 / Aval +
           z * (1 - z) * (A2 * Aval - A1^2) / Aval^2
end

function Distributions.logpdf(d::ExtremeDist, z::Real)
    f = _pdf(d, z)
    f > zero(f) || return oftype(f, -Inf)
    return log(f)
end

function Distributions.quantile(d::ExtremeDist, p)
    return _unit_quantile(d, p)
end

# Generate random samples from the radial distribution using the quantile function
Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist) = Distributions.quantile(d, rand(rng))
