#Using the convention that generator satisfies ϕ(0) = 1 (this is opposite to e.g. https://en.wikipedia.org/wiki/Copula_(probability_theory))


# We should follow closely
# https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/preprintmariushofert.pdf
# for the implementation of sampling methods. 



abstract type ArchimedeanCopula{d} <: Copula{d} end
function Distributions.cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    @assert length(C) == length(u) 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C,us)
    end
    return ϕ(C,sum_ϕ⁻¹u)
end
function ϕ⁽ᵈ⁾(C::ArchimedeanCopula{d},t) where d
    X = Taylor1(eltype(t),d)
    taylor_expansion = ϕ(C,t+X)
    coef = getcoeff(taylor_expansion,d) # gets the dth coef. 
    der = coef * factorial(d) # gets the dth derivative of $\phi$ taken in t. 
    return der
end
function Distributions._logpdf(C::CT, u) where {CT<:ArchimedeanCopula}
    d = length(C)
    # @assert d == length(u) "Dimension mismatch"
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C,ϕ⁻¹u)) #log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    
    numer = ϕ⁽ᵈ⁾(C, sum_ϕ⁻¹u)
    dimension_sign = iseven(d) ? 1.0 : -1.0 #need this for log since (-1.0)ᵈ ϕ⁽ᵈ⁾ ≥ 0.0
    return log(dimension_sign*numer) - sum_logϕ⁽¹⁾ϕ⁻¹u
end

ϕ(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
ϕ⁻¹(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
radial_dist(C::ArchimedeanCopula{d}) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
τ(C::ArchimedeanCopula{d}) where d  = @error "Archimedean interface not implemented for $(typeof(C)) yet."
τ⁻¹(::ArchimedeanCopula{d},τ) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
# radial_dist(C::ArchimedeanCopula) = laplace_transform(t -> ϕ(C,t))

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    r = rand(rng,radial_dist(C))
    Random.rand!(rng,x)
    for i in 1:length(C)
        x[i] = ϕ(C,-log(x[i])/r)
    end
    return x
end
function Base.rand(rng::Distributions.AbstractRNG,C::CT) where CT<: ArchimedeanCopula
    x = rand(rng,length(C))
    r = rand(rng,radial_dist(C))
    for i in 1:length(C)
        x[i] = ϕ(C,-log(x[i])/r)
    end
    return x
end
function Distributions.fit(::Type{CT},u) where {CT <: ArchimedeanCopula}
    # @info "Archimedean fits are by default through inverse kendall tau."
    d = size(u,1)
    τ = StatsBase.corkendall(u')
    # Then the off-diagonal elements of the matrix should be averaged: 
    avgτ = (sum(τ) .- d) / (d^2-d)
    θ = τ⁻¹(CT,avgτ)
    return CT(d,θ)
end
