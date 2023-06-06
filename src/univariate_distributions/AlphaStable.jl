# Following https://cran.r-project.org/web/packages/copula/vignettes/nacopula-pkg.pdf
# and then Th 1.19 (b) p 21 in Nolan: https://edspace.american.edu/jpnolan/wp-content/uploads/sites/1720/2020/09/Chap1.pdf

# See also there: https://discourse.julialang.org/t/bug-in-alphastabledistributions/99868

struct RestrictedAlphaStable{T} <: Distributions.ContinuousUnivariateDistribution
    α::T
    β::T
    scale::T
    location::T
end
RestrictedAlphaStable(α::T) where T = RestrictedAlphaStable(α,T(1),T(cos(α * π/2)^(1/α)),T(α == 1 ? 1 : 0))
Base.eltype(::Type{<:RestrictedAlphaStable{T}}) where {T<:AbstractFloat} = T
function Base.rand(rng::Distributions.AbstractRNG, d::RestrictedAlphaStable{T}) where {T<:AbstractFloat}

    d.α != one(T) || error("special case missing")
    # if alpha is one, 
    t₀ = atan(d.β * tan(d.α * π / 2))  # numerical problems if alpha near 1
    Θ = π * (rand(rng,T) - 0.5)
    W = -log(rand(rng,T))
    x = sin(t₀ + d.α * Θ) / (cos(t₀) * cos(Θ))^(1/d.α) * ((cos(t₀ + (d.α - 1) * Θ)) / W)^((1 - d.α)/d.α)
    return d.location + d.scale * x
end


