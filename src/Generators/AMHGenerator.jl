struct AMHGenerator{T} <: Generator
    θ::T
    function AMHGenerator(θ)
        if (θ < -1) || (θ >= 1)
            throw(ArgumentError("Theta must be in [-1,1)"))
        elseif θ == 0
            return IndependentGenrator()
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
max_monotony(G::AMHGenerator) = Inf
ϕ(  G::AMHGenerator, t) = (1-G.θ)/(exp(t)-G.θ)
ϕ⁻¹(G::AMHGenerator, t) = log(G.θ + (1-G.θ)/t)
# ϕ⁽¹⁾(G::AMHGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::AMHGenerator, k, t) = kth derivative of ϕ
williamson_dist(G::AMHGenerator, d) = G.θ >= 0 ? WilliamsonFromFrailty(1 + Distributions.Geometric(1-G.θ),d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)