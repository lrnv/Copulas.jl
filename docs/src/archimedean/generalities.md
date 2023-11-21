```@meta
CurrentModule = Copulas
```

# General Discussion

Archimedean copulas are a large class of copulas that are defined as : ... [ details needed...]

In our implementation, we separate the generator, in the `Generator` type, from the copula itself. The easyest way to understand this orthogonal design is to look at the Clayton implementation. 

We have the `ClaytonGenerator` that is defined as follows: 

```julia
struct ClaytonGenerator{T} <: UnivariateGenerator
    θ::T
    function ClaytonGenerator(θ)
        if θ < -1
            throw(ArgumentError("Theta must be greater than -1"))
        elseif θ == -1
            return WGenerator()
        elseif θ == 0
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::ClaytonGenerator) = G.θ >= 0 ? Inf : Int(floor(1 - 1/G.θ))
ϕ(  G::ClaytonGenerator, t) = max(1+G.θ*t,zero(t))^(-1/G.θ)
ϕ⁻¹(G::ClaytonGenerator, t) = (t^(-G.θ)-1)/G.θ
# ϕ⁽¹⁾(G::ClaytonGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::ClaytonGenerator, k, t) = kth derivative of ϕ
τ(G::ClaytonGenerator) = ifelse(isfinite(G.θ), G.θ/(G.θ+2), 1)
τ⁻¹(::Type{T},τ) where T<:ClaytonGenerator = ifelse(τ == 1,Inf,2τ/(1-τ))
williamson_dist(G::ClaytonGenerator, d) = G.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/G.θ,G.θ),d) : ClaytonWilliamsonDistribution(G.θ,d)
```


And then the `ClaytonCopula{d,T}` is simply an alias for `ArchimedeanCopula{d,ClaytonGenerator{T}}`: 

```julia
const ClaytonCopula{d,T} = ArchimedeanCopula{d,ClaytonGenerator{T}}
ClaytonCopula(d,θ) = ArchimedeanCopula(d,ClaytonGenerator(θ))
generatorof(::Type{ClaytonCopula}) = ClaytonGenerator
```

Not everything is necessary however, depending on the features you need. Indeed, the Archimedean API is modular: 

- Only `ϕ` and `max_monotony` are really required. Indeed, all other functions have generic fallbacks. In particular, the `williamson_dist` has a generic fallback that uses [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl) for any generator. Note however that providing the `williamson_dist` yourself if you know it will allow sampling to be an order of magnitude faster: see how in the ClaytonCopula case we provided a sampler that is different for positive parameters (the generator is completely monotonous, and thus the frailty distribution is known, so we used `WilliamsonFromFrailty`), and the negative dependent cases, where only the CDF is known and implemented in `ClaytonWilliamsonDistribution`.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.


```@docs
ArchimedeanCopula
```



# Implement your own 

If you think that the WilliamsonCopula interface is too barebone and does not provide you with enough flexibility in your modeling of an archimedean copula, you might be intersted in the possiiblity to directly subtype `ArchimedeanCopula` and implement your own. This is actually a fairly easy process and you only need to implement a few functions. Let's here together try to reimplement come archimedean copula with the follçowing generator: 

```math
my_generator
```

(describe the process...)

```julia
struct MyAC{d,T} <: ArchimedeanCopula{d}
    par::T
end
ϕ(C::MyAC{d},x)
ϕ⁻¹(C::MyAC{d},x)
τ(C::MyAC{d})
τ⁻¹(::MyAC{d},τ)
williamson_dist(C::MyAC{d})
```