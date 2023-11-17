struct ClaytonGenerator{T} <: Generator
    θ::T
    function ClaytonGenerator(θ)
        if θ < -1
            throw(ArgumentError("Theta must be greater than -1"))
        elseif θ == -1
            return WGenerator()
        elseif θ == 0
            return IndependentGenrator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::ClaytonGenerator) = Int(floor(1 - 1/G.θ))
ϕ(  G::ClaytonGenerator, t) = max(1+G.θ*t,zero(t))^(-1/G.θ)
ϕ⁻¹(G::ClaytonGenerator, t) = (t^(-G.θ)-1)/G.θ
# ϕ⁽¹⁾(G::ClaytonGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::ClaytonGenerator, k, t) = kth derivative of ϕ
williamson_dist(G::ClaytonGenerator, d) = G.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,C.θ),d) : ClaytonWilliamsonDistribution(C.θ,d)