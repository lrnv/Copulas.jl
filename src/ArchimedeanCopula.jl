abstract type ArchimedeanCopula{d} <: Copula{d} end
function Distributions.cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    return ϕ(C,sum(ϕ⁻¹.(u)))
end
function Distributions.pdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    @error "Not implemented yet (derivatives of generator needed...)"
end
function Distributions._logpdf(C::CT, u) where CT<:ArchimedeanCopula
    @error "Not implemented yet (derivatives of generator needed...)"
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
    for i in 1:C.d
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