abstract type Copula{d} <: Distributions.ContinuousMultivariateDistribution end
Base.length(::Copula{d}) where d = d

# The potential functions to code: 
# Distributions._logpdf
# Distributions.cdf
# Distributions.fit(::Type{CT},u) where CT<:Mycopula
# Distributions._rand!
# Base.eltype
# τ, τ⁻¹
# Base.eltype 

function Distributions.cdf(C::CT,u) where {CT<:Copula}
    f(x) = Distributions.pdf(C,x)
    z = zeros(eltype(u),length(C))
    return Cubature.pcubature(f,z,u,reltol=sqrt(eps()))[1]
end
function ρ(C::Copula{d}) where d
    F(x) = Distributions.cdf(C,x)
    z = zeros(d)
    i = ones(d)
    r = Cubature.pcubature(F,z,i,reltop=sqrt(eps()))[1]
    return 12*r-3
end
function τ(C::Copula)
    F(x) = Distributions.cdf(C,x)
    z = zeros(d)
    i = ones(d)
    r = Distributions.expectation(F,C)
    return 4*r-1
end

function measure(C::CT, u,v) where {CT<:Copula}

    # Computes the value of the cdf at each corner of the hypercube [u,v]
    # To obtain the C-volume of the box.
    # This assumes u[i] < v[i] for all i
    # Based on Computing the {{Volume}} of {\emph{n}} -{{Dimensional Copulas}}, Cherubini & Romagnoli 2009

    # We use a gray code according to the proposal at https://discourse.julialang.org/t/looping-through-binary-numbers/90597/6

    eval_pt = copy(u)
    d = length(C) # is known at compile time 
    r = zero(eltype(u))
    graycode = 0    # use a gray code to flip one element at a time
    which = fill(false, d) # false/true to use u/v for each component (so false here)
    r += Distributions.cdf(C,eval_pt) # the sign is always 0. 
    for s = 1:(1<<d)-1
        graycode′ = s ⊻ (s >> 1)
        graycomp = trailing_zeros(graycode ⊻ graycode′) + 1
        graycode = graycode′
        eval_pt[graycomp] = (which[graycomp] = !which[graycomp]) ? v[graycomp] : u[graycomp]
        r += (-1)^(s+d) * Distributions.cdf(C,eval_pt)
    end
    return r
end
function check_dim(C,u)
    if length(u) != length(C)
        throw(ArgumentError("Dimension mismatch between copula and input vector"))
    end
end