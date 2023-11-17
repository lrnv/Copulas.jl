struct FrankGenerator{T} <: Generator
    θ::T
    function FrankGenerator(θ)
        if θ == -Inf
            return WGenerator()
        elseif θ == 0
            return IndependentGenrator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
max_monotony(G::FrankGenerator) = G.θ < 0 ? 2 : Inf
ϕ(  G::FrankGenerator, t) = G.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ)-t)/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ(  G::FrankGenerator, t::TaylorSeries.Taylor1) = G.θ > 0 ? -log(-expm1(LogExpFunctions.log1mexp(-G.θ)-t))/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ⁻¹(G::FrankGenerator, t) = G.θ > 0 ? LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t*G.θ) : -log(expm1(-t*G.θ)/expm1(-G.θ))
# ϕ⁽¹⁾(G::FrankGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::FrankGenerator, k, t) = kth derivative of ϕ
williamson_dist(G::FrankGenerator, d) = G.θ > 0 ?  WilliamsonFromFrailty(Logarithmic(-G.θ), d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)