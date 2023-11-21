abstract type Generator end
max_monotony(G::Generator) = throw("This generator does not have a defined max monotony. You need to implement `max_monotony(G)`.")
ϕ(   G::Generator, t) = throw("This generator has not been defined correctly, the function `ϕ(G,t)` is not defined.")
ϕ⁻¹( G::Generator, x) = Roots.find_zero(t -> ϕ(G,t) - x, (0.0, Inf))
ϕ⁽¹⁾(G::Generator, t) = ForwardDiff.derivative(x -> ϕ(G,x), t)
function ϕ⁽ᵏ⁾(G::Generator, k, t)
    X = TaylorSeries.Taylor1(eltype(t),k)
    taylor_expansion = ϕ(G,t+X)
    coef = TaylorSeries.getcoeff(taylor_expansion,k) 
    der = coef * factorial(k)
    return der
end
williamson_dist(G::Generator, d) = WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)



abstract type UnivariateGenerator <: Generator end
abstract type ZeroVariateGenerator <: Generator end