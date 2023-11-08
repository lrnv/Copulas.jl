```@meta
CurrentModule = Copulas
```

# General discussion

Archimedean copulas are a large class of copulas that are defined as : ... [ details needed...]

Adding a new `ArchimedeanCopula` is very easy. The `Clayton` implementation is as short as: 

```julia
struct ClaytonCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function ClaytonCopula(d,θ)
        if θ < -1/(d-1)
            throw(ArgumentError("Theta must be greater than -1/(d-1)"))
        elseif θ == -1/(d-1)
            return WCopula(d)
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::ClaytonCopula,      t) = max(1+C.θ*t,zero(t))^(-1/C.θ)
ϕ⁻¹(C::ClaytonCopula,      t) = (t^(-C.θ)-1)/C.θ
τ(C::ClaytonCopula) = ifelse(isfinite(C.θ), C.θ/(C.θ+2), 1)
τ⁻¹(::Type{ClaytonCopula},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))
williamson_dist(C::ClaytonCopula{d,T}) where {d,T} = C.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,1),d) : ClaytonWilliamsonDistribution(C.θ,d)
```

Not everything is necessary however, depending on the features you need. Indeed, the Archimedean API is modular: 

- To sample an archimedean, only `ϕ` is required. Indeed, the `williamson_dist` has a generic fallback that uses [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl) for any generator. Note however that providing the `williamson_dist` yourself if you know it will allow sampling to be an order of magnitude faster: see how in the ClaytonCopula case we provided a sampler that is different for positive parameters (the generator is completely monotonous, and thus the frailty distribution is known, so we used `WiliamsonFromFrailty`), and the negative dependent cases, where only the CDF is known and implemented in `ClaytonWilliamsonDistribution`.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.


```@docs
ArchimedeanCopula
```

## The WilliamsonCopula concept

In this package, there is the possibility to directly implement, sample, and evaluate the pdf and cdf of an archimedean copula by only providing its generator, in an efficient way.

```@docs
WilliamsonCopula
```

