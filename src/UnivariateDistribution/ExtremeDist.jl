struct ExtremeDist{C} <: Distributions.ContinuousUnivariateDistribution
    G::C
    function ExtremeDist(G)
        return new{typeof(G)}(G)
    end
end

function Distributions.cdf(d::ExtremeDist, z)
    copula = d.G
    return z + z*(1 - z)*(d𝘈(copula, z)/𝘈(copula, z)) 
end

function _pdf(d::ExtremeDist, z)
    copula = d.G
    A = 𝘈(copula, z)
    A_prime = d𝘈(copula, z)
    A_double_prime = d²𝘈(copula, z)
    return 1 + (1 - 2z) * A_prime / A + z * (1 - z) * (A_double_prime * A - A_prime^2) / A^2
end

function Distributions.quantile(d::ExtremeDist, p)
    cdf_func(x) = Distributions.cdf(d, x) - p
    return Roots.find_zero(cdf_func, (eps(), 1-eps()), Roots.Brent())
end

# Generate random samples from the radial distribution using the quantile function
function Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist)
    u = rand(rng, Distributions.Uniform(0,1))
    return Distributions.quantile(d, u)
end