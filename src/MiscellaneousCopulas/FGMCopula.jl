"""
    FGMCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    FGMCopula(d, θ)

The Multivariate Farlie-Gumbel-Morgenstern (FGM) copula of dimension d has ``2^d-d-1`` parameters ``\\theta`` and function

```math
C(\\boldsymbol{u})=\\prod_{i=1}^{d}u_i \\left[1+ \\sum_{k=2}^{d}\\sum_{1 \\leq j_1 < \\cdots < j_k \\leq d} \\theta_{j_1 \\cdots j_k} \\bar{u}_{j_1}\\cdots \\bar{u}_{j_k} \\right],
```

where `` \\bar{u}=1-u``.

More details about Farlie-Gumbel-Morgenstern (FGM) copula are found in :
    
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.38.

We use the stochastic representation of the copula to obtain random samples.
    
    Blier-Wong, C., Cossette, H., & Marceau, E. (2022). Stochastic representation of FGM copulas using multivariate Bernoulli random variables. Computational Statistics & Data Analysis, 173, 107506.

It has a few special cases:
- When d=2 and θ = 0, it is the IndependentCopula.
"""
struct FGMCopula{d, Tθ} <: Copula{d}
    θ::Tθ
    function FGMCopula(d, θ)
        vθ = typeof(θ)<:Vector ? θ : [θ]
        if  all(θ .== 0)
            return IndependentCopula(d)
        end
        # Check first restrictions on parameters
        any(abs.(vθ) .> 1) && throw(ArgumentError("Each component of the parameter vector must satisfy that |θᵢ| ≤ 1"))
        length(vθ) != 2^d - d - 1 && throw(ArgumentError("Number of parameters (θ) must match the dimension ($d): 2ᵈ-d-1"))
        # Last check: 
        rez = new{d, typeof(vθ)}(vθ)
        for epsilon in Base.product(fill([-1, 1], d)...)
            if 1 + _reduce_over_combinations(rez,epsilon,prod) < 0
                throw(ArgumentError("Invalid parameters. The parameters do not meet the condition to be an FGM copula"))
            end
        end
        return rez
    end
end
Base.eltype(C::FGMCopula) = eltype(C.θ)
function _reduce_over_combinations(C::FGMCopula{d,Tθ}, vector_to_combine, reducer_function) where {d,Tθ}
    # This version of the reductor is non-allocative, which is much better in terms of performance. 
    # Moreover, since $d$ is a type parameter the loop will fold out at compile time :)
    rez = zero(eltype(vector_to_combine))
    # Iterate over all possible combinations of k elements, for k = 2, 3, ..., d
    i = 1
    for k in 2:d
      for indices in Combinatorics.combinations(1:d, k)
        rez += C.θ[i] * reducer_function(vector_to_combine[indices])
        i = i+1
      end
    end
    return rez
end
function _cdf(fgm::FGMCopula, u::Vector{T}) where {T}
    return prod(u) * (1 + _reduce_over_combinations(fgm, 1 .-u, prod))
end
function Distributions._logpdf(fgm::FGMCopula, u::Vector{T}) where {T}
    return log1p(_reduce_over_combinations(fgm, 1 .-2u, prod))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, fgm::FGMCopula{d,Tθ}, x::AbstractVector{T}) where {d,Tθ, T <: Real}
    if d == 2
        u = rand(rng, T)
        t = rand(rng, T)
        a = 1.0 .+ fgm.θ .* (1.0-2.0*u)
        b = sqrt.(a.^2 .-4.0 .*(a .-1.0).*t)
        v = (2.0 .*t) ./(b .+ a)
        x[1] = u
        x[2] = v[1]
        return x
    elseif d > 2
        I = zeros(T,d)
        for i in 1:d
            term = _reduce_over_combinations(fgm, I, x -> (-1)^sum(x))
            I[i] = rand(rng) < (1 / 2^d) * (1 + term)
        end
        V0 = rand(rng, d)
        V1 = rand(rng, d)
        for j in 1:d
            U_j = 1-sqrt(1-V0[j])*(1-V1[j])^(I[j])
            x[j] = U_j
        end
        return x
    end
end
τ(fgm::FGMCopula) = (2*fgm.θ[1])/9
function τ⁻¹(::Type{FGMCopula}, τ)
    if any(τ .< -2/9 .|| τ .> 2/9)
        throw(ArgumentError("For the FGM copula, tau must be in [-2/9, 2/9]."))
    end
    return max.(min.(9 * τ / 2, 1), -1)
end
ρ(fgm::FGMCopula) = (1*fgm.θ)/3 # this is weird as it will return a vector ? 
function ρ⁻¹(::Type{FGMCopula}, ρ)
    if any(ρ .< -1/3 .|| ρ .> 1/9)
        throw(ArgumentError("For the FGM copula, rho must be in [-1/3, 1/3]."))
    end
    return max.(min.(3 * ρ, 1), -1)
end