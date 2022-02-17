"""
    TCopula{d,MT}

Fields:
  - df::Int - number of degree of freedom
  - Σ::MT - covariance matrix

Constructor

    TCopula(df,Σ)

The Student's [T Copula](https://en.wikipedia.org/wiki/Multivariate_t-distribution#Copulas_based_on_the_multivariate_t) is the 
copula of a [Multivariate Student distribution](https://en.wikipedia.org/wiki/Multivariate_t-distribution). It is constructed as : 

```math
C(\\mathbf{x}; \\boldsymbol{n,\\Sigma}) = F_{n,\\Sigma}(F_{n,\\Sigma,i}^{-1}(x_i),i\\in 1,...d)
```
where ``F_{n,\\Sigma}`` is a cdf of a multivariate student random vector with covariance matrix ``\\Sigma`` and ``n`` degrees of freedom. and `F_{n,\\Sigma,i}` is the ith marignal cdf. 

It can be constructed in Julia via:  
```julia
C = TCopula(n,Σ)
```

The random number generation works as expected:
```julia
rand(C,1000)
# or
Random.rand!(C,u)
```

And yo can fit the distribution via : 
```julia
fit(TCopula,data)
```

Except that currently it does not work since `fit(Distributions.MvTDist,data)` does not dispatch. 
"""
struct TCopula{d,MT} <: EllipticalCopula{d,MT}
    df::Int
    Σ::MT
    function TCopula(df,Σ)
        make_cor!(Σ)
        return new{size(Σ,1),typeof(Σ)}(df,Σ)
    end
end
U(::Type{T}) where T<: TCopula = Distributions.TDist
N(::Type{T}) where T<: TCopula = Distributions.MvTDist
function Distributions.fit(::Type{CT},u) where {CT<:TCopula}
    N = Distributions.fit(N(CT), quantile.(U(CT)(),u))
    Σ = N.Σ
    df = N.df
    return TCopula(df,Σ)
end