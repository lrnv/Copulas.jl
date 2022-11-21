abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.length(::Copula{d}) where d = d

# The potential functions to code: 
# Distributions._logpdf
# Distributions.cdf
# Distributions.fit(::Type{CT},u) where CT<:Mycopula
# Distributions._rand!
# Base.rand
# Base.eltype
# τ, τ⁻¹
# Base.eltype 

function Distributions.cdf(C::CT,u) where {CT<:Copula}
    f(x) = pdf(C,x)
    z = zeros(eltype(u),length(C))
    return Cubature.pcubature(f,z,u,reltol=sqrt(eps()))[1]
end

using Copulas

function measure(C::CT, u,v) where {CT<:Copula}

    # Computes the value of the cdf at each corner of the hypercube [u,v]
    # To obtain the C-volume of the box.
    # This assumes u[i] < v[i] for all i
    # Based on Computing the {{Volume}} of {\emph{n}} -{{Dimensional Copulas}}, Cherubini & Romagnoli 2009

    eval_pt = similar(u)
    d = length(C)
    r = zero(eltype(u))

    for i in 1:2^d
        parity = 0
        for k in 0:(d-1)
            if (i - 1) & (1 << k) != 0
                eval_pt[k+1] = u[k+1]
            else
                eval_pt[k+1] = v[k+1]
                parity += 1
            end
        end
        r += (-1)^parity * Distributions.cdf(C,eval_pt)
    end
    return r
end
