##########
########## Copied from https://github.com/org-arl/AlphaStableDistributions.jl
########## To avoid too many dependencies and fasten the package. 
########## Probably we can do better by integrating with them... but they use too much. 
##########

Base.@kwdef struct AlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T = 1.5
    β::T = zero(α)
    scale::T = one(α)
    location::T = zero(α)
    function AlphaStable(α,β,scale,location)
        αT,βT,scaleT,locationT =  promote(α,β,scale,location)
        new{typeof(αT)}(αT,βT,scaleT,locationT)
    end
    AlphaStable{T}(pars...) where T = AlphaStable(pars...)
end
function Distributions.rand(rng::Distributions.AbstractRNG, d::AlphaStable{T}) where {T<:AbstractFloat} 
# """
# Generate independent stable random numbers.

# :param α: characteristic exponent (0.1 to 2.0)
# :param β: skew (-1 to +1)
# :param scale: scale parameter
# :param loc: location parameter (mean for α > 1, median/mode when β=0)


# This implementation is based on the method in J.M. Chambers, C.L. Mallows
# and B.W. Stuck, "A Method for Simulating Stable Random Variables," JASA 71 (1976): 340-4.
# McCulloch's MATLAB implementation (1996) served as a reference in developing this code.
# """
    α=d.α; β=d.β; sc=d.scale; loc=d.location
    (α <= 0 || α > 2) && throw(DomainError(α, "α must be in the range 0.1 to 2"))
    abs(β) > 1 && throw(DomainError(β, "β must be in the range -1 to 1"))
    # added eps(T) to prevent DomainError: x ^ y where x < 0
    ϕ = (rand(rng, T) - T(0.5)) * π * (one(T) - eps(T))
    if α == one(T) && β == zero(T)
        return loc + sc * tan(ϕ)
    end
    w = -log(rand(rng, T))
    α == 2 && (return loc + 2*sc*sqrt(w)*sin(ϕ))
    β == zero(T) && (return loc + sc * ((cos((one(T)-α)*ϕ) / w)^(one(T)/α - one(T)) * sin(α * ϕ) / cos(ϕ)^(one(T)/α)))
    cosϕ = cos(ϕ)
    if abs(α - one(T)) > 1e-8
        ζ = β * tan(π * α / 2)
        aϕ = α * ϕ
        a1ϕ = (one(T) - α) * ϕ
        return loc + sc * ((((sin(aϕ) + ζ * cos(aϕ))/cosϕ) * ((cos(a1ϕ) + ζ*sin(a1ϕ)) / (w*cosϕ))^((one(T)-α)/α)))
    end
    bϕ = π/2 + β*ϕ
    x = 2/π * (bϕ * tan(ϕ) - β * log(π/2*w*cosϕ/bϕ))
    α == one(T) || (x += β * tan(π*α/2))
    return loc + sc * x
end

Base.eltype(::Type{<:AlphaStable{T}}) where {T<:AbstractFloat} = T
