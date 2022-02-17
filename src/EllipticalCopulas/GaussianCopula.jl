"""
    GaussianCopula{d,MT}

Fields:
  - Σ::MT - covariance matrix

Constructor

    GaussianCopula(Σ)

The [Gaussian Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Gaussian_copula) is the 
copula of a [Multivariate normal distribution](http://en.wikipedia.org/wiki/Multivariate_normal_distribution). It is constructed as : 

```math
C(\\mathbf{x}; \\boldsymbol{\\Sigma}) = F_{\\Sigma}(F_{\\Sigma,i}^{-1}(x_i),i\\in 1,...d)
```
where ``F_{\\Sigma}`` is a cdf of a gaussina random vector and `F_{\\Sigma,i}` is the ith marignal cdf, while ```\\Sigma`` is the covariance matrix. 

It can be constructed in Julia via:  
```julia
C = GaussianCopula(Σ)
```

The random number generation works as expected:
```julia
rand(C,1000)
# or
Random.rand!(C,u)
```

And yo can fit the distribution via : 
```julia
fit(GaussianCopula,data)
```
"""
struct GaussianCopula{d,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function GaussianCopula(Σ) 
        make_cor!(Σ)
        return new{size(Σ,1),typeof(Σ)}(Σ)
    end
end
U(::Type{T}) where T<: GaussianCopula = Distributions.Normal
N(::Type{T}) where T<: GaussianCopula = Distributions.MvNormal
function Distributions.fit(::Type{CT},u) where {CT<:GaussianCopula}
    dd = Distributions.fit(N(CT), quantile.(U(CT)(),u))
    Σ = Matrix(dd.Σ)
    return GaussianCopula(Σ)
end

