struct AlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T
    β::T
    scale::T
    location::T
end
Base.eltype(::Type{<:AlphaStable{T}}) where {T<:AbstractFloat} = T
function Base.rand(rng::Distributions.AbstractRNG, d::AlphaStable{T}) where {T<:AbstractFloat}
    # Shamelessly taken from https://github.com/org-arl/AlphaStableDistributions.jl/blob/master/src/AlphaStableDistributions.jl
    α=d.α; β=d.β; scale=d.scale; loc=d.location
    (α < 0 || α > 2) && throw(DomainError(α, "α must be in the range 0 to 2"))
    abs(β) > 1 && throw(DomainError(β, "β must be in the range -1 to 1"))
    ϕ = (rand(rng, T) - 0.5) * π
    if α == one(T) && β == zero(T)
        return loc + scale * tan(ϕ)
    end
    w = -log(rand(rng, T))
    α == 2 && (return loc + 2*scale*sqrt(w)*sin(ϕ))
    β == zero(T) && (return loc + scale * ((cos((1-α)*ϕ) / w)^(one(T)/α - one(T)) * sin(α * ϕ) / cos(ϕ)^(one(T)/α)))
    cosϕ = cos(ϕ)
    if abs(α - one(T)) > 1e-8
        ζ = β * tan(π * α / 2)
        aϕ = α * ϕ
        a1ϕ = (one(T) - α) * ϕ
        return loc + scale * (( (sin(aϕ) + ζ * cos(aϕ))/cosϕ * ((cos(a1ϕ) + ζ*sin(a1ϕ))) / ((w*cosϕ)^((1-α)/α)) ))
    end
    bϕ = π/2 + β*ϕ
    x = 2/π * (bϕ * tan(ϕ) - β * log(π/2*w*cosϕ/bϕ))
    α == one(T) || (x += β * tan(π*α/2))
    return loc + scale * x
end


