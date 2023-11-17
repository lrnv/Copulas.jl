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
williamson_dist(C::ClaytonCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,1),d) # Radial distribution
```
The Archimedean API is modular: 

- To sample an archimedean, only `williamson_dist` and `ϕ` are needed.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.
- We plan on implementing the Williamson transformations so that `radial-dist` can be automaticlaly deduced from `ϕ` and vice versa, if you dont know much about your archimedean family

If you only know the generator of your copula, take a look at WilliamsonCopula that allows to generate automatically the associated williamson distribution. 
If on the other hand you have a univaraite positive random variable with no atom at zero, then the williamson transform can produce an archimdean copula out of it, with the same constructor. 
"""
struct ArchimedeanCopula{d,TG} <: Copula{d}
    G::TG
    function ArchimedeanCopula(d::Int,G::Generator)
        @assert d <= max_monotony(G) "The generator you provided is not d-monotonous according to its max_monotonicity property, and thus this copula does not exists."
        return ArchimedeanCopula{d,typeof(G)}(G)
    end
    # three special cases: 
    ArchimedeanCopula(d,::WGenerator) = WCopula(d)
    ArchimedeanCopula(d,::IndependentGenerator) = IndependentCopula(d)
    ArchimedeanCopula(d,::MGenerator) = MCopula(d)
end
function _cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C.G,us)
    end
    return ϕ(C.G,sum_ϕ⁻¹u)
end
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .<= u .<= 1)
        return eltype(u)(-Inf)
    end
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C.G,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C.G,ϕ⁻¹u)) # log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    numer = ϕ⁽ᵏ⁾(C.G, d, sum_ϕ⁻¹u)
    dimension_sign = iseven(d) ? 1.0 : -1.0 #need this for log since (-1.0)ᵈ ϕ⁽ᵈ⁾ ≥ 0.0


    # I am not sure this is the right reasoning :
    if numer == 0
        if sum_logϕ⁽¹⁾ϕ⁻¹u == -Inf
            return Inf
        else
            return -Inf
        end
    else
        return log(dimension_sign*numer) - sum_logϕ⁽¹⁾ϕ⁻¹u
    end
end
_W(C::ArchimedeanCopula{d,TG}) where {d,TG} = williamson_dist(C.G,d)
function τ(C::ArchimedeanCopula)  
    @show C
    return 4*Distributions.expectation(r -> ϕ(C,r), _W(C)) - 1
end

function _archi_rand!(rng,C::ArchimedeanCopula{d},R,x) where d
    # x is assumed to already be random exponentials produced by Random.randexp
    r = rand(rng,R)
    sx = sum(x)
    for i in 1:d
        x[i] = ϕ(C.G,r * x[i]/sx)
    end
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # By default, we use the williamson sampling. 
    Random.randexp!(rng,x)
    _archi_rand!(rng,C.G,_W(C),x)
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # More efficient version that precomputes the williamson transform on each call to sample in batches: 
    Random.randexp!(rng,A)
    R = _W(C)
    for i in 1:size(A,2)
        _archi_rand!(rng,C,R,view(A,:,i))
    end
    return A
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