struct ExtremeDist{C} <: Distributions.ContinuousUnivariateDistribution
    tail::C
end

function Distributions.cdf(d::ExtremeDist, z)
    z < 0 || z > 1 && return 0.0
    return z + z * (1 - z) * (dA(d.tail, z)/A(d.tail, z))
end

function _pdf(d::ExtremeDist, z)
    z < 0 || z > 1 && return 0.0
    A, A1, A2 = _A_dA_d²A(d.tail, z)
    return 1 + (1 - 2z) * A1 / A + z * (1 - z) * (A2 * A - A1^2) / A^2
end

function Distributions.quantile(d::ExtremeDist, p)
    return _unit_quantile(d, p)
end

# Generate random samples from the radial distribution using the quantile function
Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist) = Distributions.quantile(d, rand(rng))
