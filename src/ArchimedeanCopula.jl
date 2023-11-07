#Using the convention that generator satisfies ϕ(0) = 1 (this is opposite to e.g. https://en.wikipedia.org/wiki/Copula_(probability_theory))


# We should follow closely
# https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/preprintmariushofert.pdf
# for the implementation of sampling methods. 


"""
    ArchimedeanCopula{d}

Abstract type.

[ The description of the APi done here could be more detailled]

The archimedean copulas are all ``<:ArchimedeanCopula{d}` for some dimension d. This type serves as an internal interface to handle all archimdean copulas at once. 

Adding a new `ArchimedeanCopula` is very easy. The `Clayton` implementation is as short as: 

```julia
struct ClaytonCopula{d,T} <: Copulas.ArchimedeanCopula{d}
    θ::T
end
ClaytonCopula(d,θ)            = ClaytonCopula{d,typeof(θ)}(θ)     # Constructor
ϕ(C::ClaytonCopula, t)        = (1+sign(C.θ)*t)^(-1/C.θ)          # Generator
ϕ⁻¹(C::ClaytonCopula,t)       = sign(C.θ)*(t^(-C.θ)-1)            # Inverse Generator
τ(C::ClaytonCopula)           = C.θ/(C.θ+2)                       # θ -> τ
τ⁻¹(::Type{ClaytonCopula},τ)  = 2τ/(1-τ)                          # τ -> θ
frailty_dist(C::ClaytonCopula) = Distributions.Gamma(1/C.θ,1)      # Radial distribution
```
The Archimedean API is modular: 

- To sample an archimedean, only `frailty_dist` and `ϕ` are needed.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.
- We plan on implementing the Williamson transformations so that `radial-dist` can be automaticlaly deduced from `ϕ` and vice versa, if you dont know much about your archimedean family
"""
abstract type ArchimedeanCopula{d} <: Copula{d} end
function Distributions.cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    @assert length(C) == length(u) 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C,us)
    end
    return ϕ(C,sum_ϕ⁻¹u)
end
ϕ⁽¹⁾(C::CT, t) where {CT<:ArchimedeanCopula} = ForwardDiff.derivative(x -> ϕ(C,x), t)
function ϕ⁽ᵈ⁾(C::ArchimedeanCopula{d},t) where d
    X = TaylorSeries.Taylor1(eltype(t),d)
    taylor_expansion = ϕ(C,t+X)
    coef = TaylorSeries.getcoeff(taylor_expansion,d) # gets the dth coef. 
    der = coef * factorial(d) # gets the dth derivative of $\phi$ taken in t. 
    return der
end
function Distributions._logpdf(C::CT, u) where {CT<:ArchimedeanCopula}
    d = length(C)
    # @assert d == length(u) "Dimension mismatch"
    if !all(0 .<= u .<= 1)
        return eltype(u)(-Inf)
    end
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C,ϕ⁻¹u)) #log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    
    numer = ϕ⁽ᵈ⁾(C, sum_ϕ⁻¹u)
    dimension_sign = iseven(d) ? 1.0 : -1.0 #need this for log since (-1.0)ᵈ ϕ⁽ᵈ⁾ ≥ 0.0
    return log(dimension_sign*numer) - sum_logϕ⁽¹⁾ϕ⁻¹u
end

ϕ(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
ϕ⁻¹(C::ArchimedeanCopula{d},x) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
frailty_dist(C::ArchimedeanCopula{d}) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
τ(C::ArchimedeanCopula{d}) where d  = @error "Archimedean interface not implemented for $(typeof(C)) yet."
τ⁻¹(::ArchimedeanCopula{d},τ) where d = @error "Archimedean interface not implemented for $(typeof(C)) yet."
# frailty_dist(C::ArchimedeanCopula) = laplace_transform(t -> ϕ(C,t))

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    r = rand(rng,frailty_dist(C))
    Random.rand!(rng,x)
    for i in 1:length(C)
        x[i] = ϕ(C,-log(x[i])/r)
    end
    return x
end
function Distributions.fit(::Type{CT},u) where {CT <: ArchimedeanCopula}
    # @info "Archimedean fits are by default through inverse kendall tau."
    d = size(u,1)
    τ = StatsBase.corkendall(u')
    # Then the off-diagonal elements of the matrix should be averaged: 
    avgτ = (sum(τ) .- d) / (d^2-d)
    θ = τ⁻¹(CT,avgτ)
    return CT(d,θ)
end
