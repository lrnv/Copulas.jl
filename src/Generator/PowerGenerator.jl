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
        if α == 1 && β == 1
            return G
        end
        α,β = promote(α,β)
        return new{typeof(G),typeof(β)}(G, α, β)
    end
end

# Parameter extraction for consistency with other generators
Distributions.params(G::PowerGenerator) = (G.α, G.β)

# Maximum monotony is preserved from the underlying generator
max_monotony(G::PowerGenerator) = max_monotony(G.G)

# Core generator function: ϕ(t) = ϕ_G(t^α)^β
ϕ(G::PowerGenerator, t) = ϕ(G.G, t^G.α)^G.β

# Inverse function: if y = ϕ_G(t^α)^β, then t = (ϕ_G⁻¹(y^(1/β)))^(1/α)
ϕ⁻¹(G::PowerGenerator, y) = (ϕ⁻¹(G.G, y^(1/G.β)))^(1/G.α)

# Kendall's tau - preserved from the underlying generator according to the theory
τ(G::PowerGenerator) = τ(G.G)

# Williamson distribution - use the underlying generator's distribution  
williamson_dist(G::PowerGenerator, ::Val{d}) where d = williamson_dist(G.G, Val{d}())
williamson_dist(G::PowerGenerator, d::Int) = williamson_dist(G.G, d)