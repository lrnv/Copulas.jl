#Using the convention that generator satisfies ϕ(0) = 1 (this is opposite to e.g. https://en.wikipedia.org/wiki/Copula_(probability_theory))


# We should follow closely
# https://www.uni-ulm.de/fileadmin/website_uni_ulm/mawi.inst.zawa/forschung/preprintmariushofert.pdf
# for the implementation of sampling methods. 

"""
    WilliamsonGenerator{d,Tϕ,TX}

Fields:
    - ϕ::Tϕ -- a function representing the archimedean generator.
    - X::TX -- a random variable that represents its williamson d-transform

Constructors

    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)

The WilliamsonCopula is the barebone Archimedean Copula that directly leverages the Williamson transform and inverse transform (in their greatest generalities), that are implemented in [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl). You can construct it by providing the Williamson-d-tranform as a (non-negative) random variable `X::Distributions.UnivariateDistribution`, or by providing the ``d``-monotone generator `ϕ::Function` as a function from ``\\mathbb R_+`` to ``[0,1]``, decreasing and d-monotone. The other component will be computed automatically. You also have the option to provide both, which will probably be faster if you have an analytical expression for both. 

About `WilliamsonCopula(X::Distributions.UnivariateDistribution, d)`: For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer ``d\\ge 2``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

This function has several properties: 
- We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

These properties makes this function what is called an *archimedean generator*, able to generate *archimedean copulas* in dimensions up to ``d``. 

About `WilliamsonCopula(ϕ::Function, d)`: On the other hand, `WilliamsonCopula(ϕ::Function, d)` Computes the inverse Williamson d-transform of the d-monotone archimedean generator ϕ. 

A ``d``-monotone archimedean generator is a function ``\\phi`` on ``\\mathbb R_+`` that has these three properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

For such a function ``\\phi``, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``X``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\\phi)(x) = 1 - \\frac{(-x)^{d-1} \\phi_+^{(d-1)}(x)}{k!} - \\sum_{k=0}^{d-2} \\frac{(-x)^k \\phi^{(k)}(x)}{k!}
```

We return this cumulative distribution function in the form of the corresponding random variable `<:Distributions.ContinuousUnivariateDistribution` from `Distributions.jl`. You may then compute : 
    - The cdf via `Distributions.cdf`
    - The pdf via `Distributions.pdf` and the logpdf via `Distributions.logpdf`
    - Samples from the distribution via `rand(X,n)`


References: 
    Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""




"""
    ArchimedeanCopula{d, TG}

Fields: 
    - G::TG : the generator <: Generator. 

Constructor: 

    ArchimedeanCopula(d::Int,G::Generator)

For some archimedean generator `G::Generator` and some dimenson `d`, this class models the archimedean copula wich has this generator. More formally, a ``d``-monotone archimedean generator is a function ``\\phi`` on ``\\mathbb R_+`` that has these three properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

In the code, this function is implemented as `ϕ(G::Generator,t)`, and its max monotony `d` is given by `max_monotony(G)`. The corresponding d-variate archimedean copula writes: 

```math
C(\\mathbf u) = \\phi^{-1}\\left(\\sum_{i=1}^d \\phi(u_i)\\right)
```

It can be sampled through A Radial simplex decomposition as follows: If ``\\mathbf U \\sim C``, then ``U \\equal R \\mathbf S`` for a random vector ``\\mathbf S \\sim`` `Dirichlet(ones(d))`, that is uniformity on the d-variate simplex, and a non-negative random variable ``R`` that is the Williamson d-transform of `\\phi`. The density, cdf, kendall tau, and may other properties of the Archimedean copula are then directly deduced from this relationship. 

There exists several known parametric generators that are implement in the package. For every `NamedGenerator <: Generator` implemented in the package, we provide a type alias ``NamedCopula{d,...} = ArchimedeanCopula{d,NamedGenerator{...}}` to be able to manipulate the archimedean copulas without too much hassle for known and usefull special cases. 

A generic archimdeanc copula can be constructed as follows: 

```julia
struct MyGenerator <: Generator end
ϕ(G::MyGenerator,t) = exp(-t) # your archimedean generator, can be any d-monotonous function.
max_monotony(G::MyGenerator) = Inf # could depend on generators parameters. 
C = ArchimedeanCopula(d,MyGenerator())
```

The obtained model can be used as follows: 
```julia
spl = rand(C,1000)   # sampling
cdf(C,spl)           # cdf
pdf(C,spl)           # pdf
loglikelihood(C,spl) # llh
```

The following functions have defaults but can be overridden for performance: 

```julia
ϕ⁻¹(C::MyArchimedean, t)        # Inverse of ϕ
ϕ⁽¹⁾(C::MyArchimedean, t)       # first defrivative of ϕ
ϕ⁽ᵈ⁾(C::MyArchimedean,t)        # dth defrivative of ϕ
τ(C::MyArchimedean)             # Kendall tau
τ⁻¹(::Type{MyArchimedean},τ) =  # Inverse kendall tau
fit(::Type{MyArchimedean},data) # fitting.
```

As a bonus, If you know the williamson d-transform of your generator and not your generator itself, you may take a look at [`WilliamsonGenerator`](@ref) that implements them. If you rather know the frailty distribution, take a look at `WilliamsonFromFrailty`.
"""
struct ArchimedeanCopula{d,TG} <: Copula{d}
    G::TG
    function ArchimedeanCopula(d::Int,G::Generator)
        @assert d <= max_monotony(G) "The generator you provided is not d-monotonous according to its max_monotonicity property, and thus this copula does not exists."
        return new{d,typeof(G)}(G)
    end
end
ϕ(C::ArchimedeanCopula{d,TG},t)    where {d,TG} = ϕ(C.G,t)
ϕ⁻¹(C::ArchimedeanCopula{d,TG},t)  where {d,TG} = ϕ⁻¹(C.G,t)
ϕ⁽¹⁾(C::ArchimedeanCopula{d,TG},t) where {d,TG} = ϕ⁽¹⁾(C.G,t)
ϕ⁽ᵏ⁾(C::ArchimedeanCopula{d,TG},k,t) where {d,TG} = ϕ⁽ᵏ⁾(C.G,k,t)
williamson_dist(C::ArchimedeanCopula{d,TG}) where {d,TG} = williamson_dist(C.G,d)


function _cdf(C::CT,u) where {CT<:ArchimedeanCopula} 
    sum_ϕ⁻¹u = 0.0
    for us in u
        sum_ϕ⁻¹u += ϕ⁻¹(C,us)
    end
    return ϕ(C,sum_ϕ⁻¹u)
end
function Distributions._logpdf(C::ArchimedeanCopula{d,TG}, u) where {d,TG}
    if !all(0 .<= u .<= 1)
        return eltype(u)(-Inf)
    end
    sum_ϕ⁻¹u = 0.0
    sum_logϕ⁽¹⁾ϕ⁻¹u = 0.0
    for us in u
        ϕ⁻¹u = ϕ⁻¹(C,us)
        sum_ϕ⁻¹u += ϕ⁻¹u
        sum_logϕ⁽¹⁾ϕ⁻¹u += log(-ϕ⁽¹⁾(C,ϕ⁻¹u)) # log of negative here because ϕ⁽¹⁾ is necessarily negative
    end
    numer = ϕ⁽ᵏ⁾(C, d, sum_ϕ⁻¹u)
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

# function τ(C::ArchimedeanCopula)  
#     return 4*Distributions.expectation(r -> ϕ(C,r), williamson_dist(C)) - 1
# end

function _archi_rand!(rng,C::ArchimedeanCopula{d},R,x) where d
    # x is assumed to already be random exponentials produced by Random.randexp
    r = rand(rng,R)
    sx = sum(x)
    for i in 1:d
        x[i] = ϕ(C,r * x[i]/sx)
    end
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # By default, we use the williamson sampling. 
    Random.randexp!(rng,x)
    _archi_rand!(rng,C,williamson_dist(C),x)
    return x
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, A::DenseMatrix{T}) where {T<:Real, CT<:ArchimedeanCopula}
    # More efficient version that precomputes the williamson transform on each call to sample in batches: 
    Random.randexp!(rng,A)
    R = williamson_dist(C)
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

τ(C::ArchimedeanCopula{d,TG}) where {d,TG} = τ(C.G)
function τ⁻¹(::Type{T},τ_val) where {T<:ArchimedeanCopula}
    return τ⁻¹(generatorof(T),τ_val)
end


################################################################################################
################                                                                ################
################ Define all "named" archimedean copulas at once like that :     ################
################                                                                ################
################################################################################################

################################################################################################
# Deal with easy cases with meta-programming: 
const AMHCopula{d,T} = ArchimedeanCopula{d,AMHGenerator{T}}
AMHCopula(d,θ) = ArchimedeanCopula(d,AMHGenerator(θ))
generatorof(::Type{AMHCopula}) = AMHGenerator

const ClaytonCopula{d,T} = ArchimedeanCopula{d,ClaytonGenerator{T}}
ClaytonCopula(d,θ) = ArchimedeanCopula(d,ClaytonGenerator(θ))
generatorof(::Type{ClaytonCopula}) = ClaytonGenerator

const FrankCopula{d,T} = ArchimedeanCopula{d,FrankGenerator{T}}
FrankCopula(d,θ) = ArchimedeanCopula(d,FrankGenerator(θ))
generatorof(::Type{FrankCopula}) = FrankGenerator

const GumbelBarnettCopula{d,T} = ArchimedeanCopula{d,GumbelBarnettGenerator{T}}
GumbelBarnettCopula(d,θ) = ArchimedeanCopula(d,GumbelBarnettGenerator(θ))
generatorof(::Type{GumbelBarnettCopula}) = GumbelBarnettGenerator

const GumbelCopula{d,T} = ArchimedeanCopula{d,GumbelGenerator{T}}
GumbelCopula(d,θ) = ArchimedeanCopula(d,GumbelGenerator(θ))
generatorof(::Type{GumbelCopula}) = GumbelGenerator

const InvGaussianCopula{d,T} = ArchimedeanCopula{d,InvGaussianGenerator{T}}
InvGaussianCopula(d,θ) = ArchimedeanCopula(d,InvGaussianGenerator(θ))
generatorof(::Type{InvGaussianCopula}) = InvGaussianGenerator

const JoeCopula{d,T} = ArchimedeanCopula{d,JoeGenerator{T}}
JoeCopula(d,θ) = ArchimedeanCopula(d,JoeGenerator(θ))
generatorof(::Type{JoeCopula}) = JoeGenerator

const IndependentCopula{d} = ArchimedeanCopula{d,IndependentGenerator}
IndependentCopula(d) = ArchimedeanCopula(d,IndependentGenerator())
generatorof(::Type{IndependentCopula}) = IndependentGenerator

const MCopula{d} = ArchimedeanCopula{d,MGenerator}
MCopula(d) = ArchimedeanCopula(d,MGenerator())
generatorof(::Type{MCopula}) = MGenerator

const WCopula{d} = ArchimedeanCopula{d,WGenerator}
WCopula(d) = ArchimedeanCopula(d,WGenerator())
generatorof(::Type{WCopula}) = WGenerator

# The zero-variate ones just need a few more methods: 
Distributions._logpdf(::ArchimedeanCopula{d,IndependentGenerator}, u) where {d} = all(0 .<= u .<= 1) ? zero(eltype(u)) : eltype(u)(-Inf)
Distributions._logpdf(::ArchimedeanCopula{d,MGenerator},           u) where {d} = all(u == u[1]) ?     zero(eltype(u)) : eltype(u)(-Inf)
Distributions._logpdf(::ArchimedeanCopula{d,WGenerator},           u) where {d} = sum(u) == 1 ?   zero(eltype(u)) : eltype(u)(-Inf)

_cdf(::ArchimedeanCopula{d,IndependentGenerator}, u) where d = prod(u)
_cdf(::ArchimedeanCopula{d,MGenerator},           u) where {d} = minimum(u)
_cdf(::ArchimedeanCopula{d,WGenerator},           u) where {d} = max(1 + sum(u)-d,0)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,IndependentGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    Random.rand!(rng,x)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,MGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::ArchimedeanCopula{d,WGenerator}, x::AbstractVector{T}) where {d,T<:Real}
    @assert d==2
    x[1] = rand(rng)
    x[2] = 1-x[1] 
end


function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,IndependentGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    Random.rand!(rng,A)
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,MGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    A[1,:] .= rand(rng,size(A,2))
    for i in 2:size(A,1)
        A[i,:] .= A[1,:]
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ArchimedeanCopula{d,WGenerator}, A::DenseMatrix{T}) where {T<:Real, d}
    @assert size(A,1) == 2
    A[1,:] .= rand(rng,size(A,2))
    A[2,:] .= 1 .- A[1,:]
    return A
end

