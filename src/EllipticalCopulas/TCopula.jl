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
C = TCopula(2,Σ)
```

You can sample it, compute pdf and cdf, or even fit the distribution via: 
```julia
u = rand(C,1000)
Random.rand!(C,u) # other calling syntax for rng.
pdf(C,u) # to get the density
cdf(C,u) # to get the distribution function 
Ĉ = fit(TCopula,u) # to fit on the sampled data. 
```


Except that currently it does not work since `fit(Distributions.MvTDist,data)` does not dispatch. 

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct TCopula{d,df,MT} <: EllipticalCopula{d,MT}
    Σ::MT
    function TCopula(df,Σ)
        make_cor!(Σ)
        N(TCopula{size(Σ,1),df,typeof(Σ)})(Σ)
        return new{size(Σ,1),df,typeof(Σ)}(Σ)
    end
end
U(::Type{TCopula{d,df,MT}}) where {d,df,MT} = Distributions.TDist(df)
N(::Type{TCopula{d,df,MT}}) where {d,df,MT} = function(Σ)
    Distributions.MvTDist(df,Σ)
end

function Distributions.params(C::TCopula{d,df,MT}) where {d,df,MT}
    Σ = C.Σ
    n = size(Σ, 1)
    # Extraemos los elementos únicos de la triangular superior (sin diagonal)
    rhos = Tuple(Σ[i, j] for i in 1:n for j in (i+1):n)
    return (df, rhos...)  # Devuelve (df, ρ₁₂, ρ₁₃, ..., ρₙ₋₁ₙ)
end

# Kendall tau of bivariate student: 
# Lindskog, F., McNeil, A., & Schmock, U. (2003). Kendall’s tau for elliptical distributions. In Credit risk: Measurement, evaluation and management (pp. 149-156). Heidelberg: Physica-Verlag HD.
τ(C::TCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π 