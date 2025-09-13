"""
    PowerGenerator{TG,T}

Fields:
* `G::Generator` - another generator
* `α::Real` - parameter, the inner power, positive
* `β::Real` - parameter, the outer power, positive

Constructor

    PowerGenerator(G, α, β)

    The inner/outer power generator based on the generator ϕ given by 
    
```math
\\phi_{\\alpha,\\beta}(t) = \\phi(t^\\alpha)^\\beta
```

It keeps the monotony of ϕ. 

It has a few special cases: 
    - When α = 1 and β = 1, it returns G.

References : 
* [nelsen2006](@cite) Nelsen, R. B. (2006). An introduction to copulas. Springer, theorem 4.5.1 p141
"""
struct PowerGenerator{TG,T} <: Generator
    G::TG
    α::T
    β::T
    function PowerGenerator(G, α, β)
        @assert α > 0
        @assert β > 0
        α == 1 && β == 1 && return G
        α,β = promote(α,β)
        return new{typeof(G),typeof(β)}(G, α, β)
    end
end
ϕ(G::PowerGenerator, t) = exp(G.β * log(ϕ(G.G, exp(G.α * log(t)))))
ϕ⁻¹(G::PowerGenerator, y) = exp((1/G.α) * log(ϕ⁻¹(G.G, exp((1/G.β) * log(y)))))
max_monotony(G::PowerGenerator) = max_monotony(G.G)
Distributions.params(G::PowerGenerator) = (G.α, G.β, Distributions.params(G.G)...)

