abstract type Generator end

max_monotony(G::Generator) = throw("This generator does not have a defined max monotony")
ϕ(   G::Generator, t) = throw("This generator has not been defined correctly.")
ϕ⁻¹( G::Generator, x) = Roots.find_zero(t -> ϕ(G,t) - x, (0.0, Inf))
ϕ⁽¹⁾(G::Generator, t) = ForwardDiff.derivative(x -> ϕ(G,x), t)
function ϕ⁽ᵏ⁾(G::Generator, k, t)
    X = TaylorSeries.Taylor1(eltype(t),k)
    taylor_expansion = ϕ(C,t+X)
    coef = TaylorSeries.getcoeff(taylor_expansion,k) 
    der = coef * factorial(d)
    return der
end
williamson_dist(G::Generator, d) = WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)


# Maybe the three following generators do not need to be that much special case ? 
# Maybe they do. 


struct WGenerator <: Generator end
# max_monotony(G::Wgenerator) = 2
# ϕ(G::WGenerator, t) = 

struct IndependentGenerator <: Generator end
# max_monotony(G::IndependentGenerator) = Inf
# ϕ(   G::IndependentGenerator, t) = exp(-t)

struct MGenerator <: Generator end
# max_monotony(G::MGenerator) = Inf
# ϕ(   G::MGenerator, t) = throw(ArgumentError("The MGenerator should never be called."))

