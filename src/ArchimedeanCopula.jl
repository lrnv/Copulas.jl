#Using the convention that generator satisfies ϕ(0) = 1 (this is opposite to e.g. https://en.wikipedia.org/wiki/Copula_(probability_theory))
abstract type ArchimedeanCopula{d} <: Copula{d} end
function Distributions.cdf(C::CT,u::Vector) where {CT<:ArchimedeanCopula} 
    @assert length(C) == length(u) 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C,us)
    end
    return ϕ(C,sum_ϕ⁻¹u)
end
function Distributions.pdf(C::CT,u::Vector) where {CT<:ArchimedeanCopula} 
    d = length(C)
    @assert d == length(u) 
    sum_ϕ⁻¹u = 0.0
    prod_ϕ⁽¹⁾ϕ⁻¹u = 1.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        prod_ϕ⁽¹⁾ϕ⁻¹u *= ϕ⁽¹⁾(C,ϕ⁻¹u)
    end

    numer = if d == 2
        ϕ⁽²⁾(C, sum_ϕ⁻¹u)
    elseif d == 3
        ϕ⁽³⁾(C, sum_ϕ⁻¹u)
    elseif d == 4
        ϕ⁽⁴⁾(C, sum_ϕ⁻¹u)
    elseif d == 5
        ϕ⁽⁵⁾(C, sum_ϕ⁻¹u)
    elseif d == 6
        ϕ⁽⁶⁾(C, sum_ϕ⁻¹u)
    elseif d == 7
        ϕ⁽⁷⁾(C, sum_ϕ⁻¹u)
    elseif d == 8
        ϕ⁽⁸⁾(C, sum_ϕ⁻¹u)
    elseif d == 9
        ϕ⁽⁹⁾(C, sum_ϕ⁻¹u)
    end

    return  numer/prod_ϕ⁽¹⁾ϕ⁻¹u
end
function Distributions.logpdf(C::CT, u::Vector) where CT<:ArchimedeanCopula
    d = length(C)
    @assert d == length(u) "Dimension mismatch"
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C,ϕ⁻¹u)) #log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    
    numer = if d == 2
        ϕ⁽²⁾(C, sum_ϕ⁻¹u)
    elseif d == 3
        ϕ⁽³⁾(C, sum_ϕ⁻¹u)
    elseif d == 4
        ϕ⁽⁴⁾(C, sum_ϕ⁻¹u)
    elseif d == 5
        ϕ⁽⁵⁾(C, sum_ϕ⁻¹u)
    elseif d == 6
        ϕ⁽⁶⁾(C, sum_ϕ⁻¹u)
    elseif d == 7
        ϕ⁽⁷⁾(C, sum_ϕ⁻¹u)
    elseif d == 8
        ϕ⁽⁸⁾(C, sum_ϕ⁻¹u)
    elseif d == 9
        ϕ⁽⁹⁾(C, sum_ϕ⁻¹u)
    end

    dimension_sign = iseven(d) ? 1.0 : -1.0 #need this for log since (-1.0)ᵈ ϕ⁽ᵈ⁾ ≥ 0.0

    return log(dimension_sign*numer) - sum_logϕ⁽¹⁾ϕ⁻¹u
end

function Distributions.cdf(C::CT,u::Matrix) where {CT<:ArchimedeanCopula} 
    @assert length(C) == size(u,1)
    return [cdf(C,u[:,i]) for i in 1:size(u,2)]
end

function Distributions.pdf(C::CT,u::Matrix) where {CT<:ArchimedeanCopula} 
    @assert length(C) == size(u,1)
    return [pdf(C,u[:,i]) for i in 1:size(u,2)]
end

function Distributions.logpdf(C::CT,u::Matrix) where {CT<:ArchimedeanCopula} 
    @assert length(C) == size(u,1)
    return [logpdf(C,u[:,i]) for i in 1:size(u,2)]
end

ϕ(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet"
ϕ⁻¹(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet"
radial_dist(C::ArchimedeanCopula{d}) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet"
τ(C::ArchimedeanCopula{d}) where d  = @error "Archimedean interface not implemented for $(typeof(C)) yet"
τ⁻¹(::ArchimedeanCopula{d},τ) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet"
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

# We should follow closely
# https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/preprintmariushofert.pdf
# for the implementation of sampling methods. 

# using ForwardDiff.derivative for derivatives of the generator
# maybe a smarter way to do this for general order derivative instead of stopping at 9
ϕ⁽¹⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ(C,x), t)
ϕ⁽²⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽¹⁾(C,x), t)
ϕ⁽³⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽²⁾(C,x), t)
ϕ⁽⁴⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽³⁾(C,x), t)
ϕ⁽⁵⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽⁴⁾(C,x), t)
ϕ⁽⁶⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽⁵⁾(C,x), t)
ϕ⁽⁷⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽⁶⁾(C,x), t)
ϕ⁽⁸⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽⁷⁾(C,x), t)
ϕ⁽⁹⁾(C::CT, t::Real) where {CT<:ArchimedeanCopula} = derivative(x -> ϕ⁽⁸⁾(C,x), t)