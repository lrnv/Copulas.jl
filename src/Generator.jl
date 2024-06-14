"""
    Generator

Abstract type. Implements the API for archimedean generators. 

An Archimedean generator is simply a function 
``\\phi :\\mathbb R_+ \\to [0,1]`` such that ``\\phi(0) = 1`` and ``\\phi(+\\infty) = 0``. 

To generate an archimedean copula in dimension ``d``, the function also needs to be ``d``-monotone, that is : 

- ``\\phi`` is ``d-2`` times derivable.
- ``(-1)^k \\phi^{(k)} \\ge 0 \\;\\forall k \\in \\{1,..,d-2\\},`` and if ``(-1)^{d-2}\\phi^{(d-2)}`` is a non-increasing and convex function. 

The access to the function ``\\phi`` itself is done through the interface: 

    ϕ(G::Generator, t)

We do not check algorithmically that the proposed generators are d-monotonous. Instead, it is up to the person implementing the generator to tell the interface how big can ``d`` be through the function 

    max_monotony(G::MyGenerator) = # some integer, the maximum d so that the generator is d-monotonous.


More methods can be implemented for performance, althouhg there are implement defaults in the package : 

* `ϕ⁻¹( G::Generator, x)` gives the inverse function of the generator.
* `ϕ⁽¹⁾(G::Generator, t)` gives the first derivative. 
* `ϕ⁽ᵏ⁾(G::Generator, k, t)` gives the kth derivative. 
* `williamson_dist(G::Generator, d)` gives the Wiliamson d-transform of the generator, see [WilliamsonTransforms.jl](https://github.com/lrnv/WilliamsonTransforms.jl).

References: 
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.
"""
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

# τ(G::Generator) = @error("This generator has no kendall tau implemented.")
# ρ(G::Generator) = @error ("This generator has no Spearman rho implemented.")
# τ⁻¹(G::Generator, τ_val) = @error("This generator has no inverse kendall tau implemented.")
# ρ⁻¹(G::Generator, ρ_val) = @error ("This generator has no inverse Spearman rho implemented.")


abstract type UnivariateGenerator <: Generator end
abstract type ZeroVariateGenerator <: Generator end
abstract type DistordedGenerator <: Generator end