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
    p < 0 || p > 1 && throw(ArgumentError("p must be between 0 and 1"))
    # Automatically decide whether to use binary search or Brent
    if needs_binary_search(d.tail)
        # Use binary search for copulas with large parameters     
        lower_bound = eps()
        upper_bound = 1.0 - eps()
        mid_point = (lower_bound + upper_bound) / 2

        while upper_bound - lower_bound > 1e-6  # Accuracy threshold
            mid_value = Distributions.cdf(d, mid_point) - p

            if abs(mid_value) < 1e-6
                return mid_point
            elseif mid_value > 0
                upper_bound = mid_point
            else
                lower_bound = mid_point
            end

            mid_point = (lower_bound + upper_bound) / 2
        end

        return mid_point
    else
        # Use Brent for other copulations or if there are no problems
        return Roots.find_zero(x -> Distributions.cdf(d, x) - p, (eps(), 1.0 - eps()), Roots.Brent())
    end
end

# Generate random samples from the radial distribution using the quantile function
Distributions.rand(rng::Distributions.AbstractRNG, d::ExtremeDist) = Distributions.quantile(d, rand(rng))
