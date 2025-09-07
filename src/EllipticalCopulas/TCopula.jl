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
function Distributions.fit(::Type{CT},u) where {CT<:TCopula}
    N = Distributions.fit(N(CT), quantile.(U(CT),u))
    Σ = N.Σ
    df = N.df
    return TCopula(df,Σ)
end

# Kendall tau of bivariate student: 
# Lindskog, F., McNeil, A., & Schmock, U. (2003). Kendall’s tau for elliptical distributions. In Credit risk: Measurement, evaluation and management (pp. 149-156). Heidelberg: Physica-Verlag HD.
τ(C::TCopula{2,MT}) where MT = 2*asin(C.Σ[1,2])/π 

# Conditioning colocated
@inline function DistortionFromCop(C::TCopula{D,ν,MT}, js::NTuple{p,Int64}, uⱼₛ::NTuple{p,Float64}, i::Int64) where {p,D,ν,MT}
    Σ = C.Σ; jst = js; ist = Tuple(setdiff(1:D, jst)); @assert i in ist
    Jv = collect(jst); zJ = Distributions.quantile.(Distributions.TDist(ν), collect(uⱼₛ))
    ΣJJ = Σ[Jv, Jv]; RiJ = Σ[i, Jv]; RJi = Σ[Jv, i]
    if length(Jv) == 1
        r = RiJ[1]; μz = r * zJ[1]; σ0² = 1 - r^2; δ = zJ[1]^2
    else
        L = LinearAlgebra.cholesky(Symmetric(ΣJJ))
        μz = dot(RiJ, (L' \ (L \ zJ)))
        σ0² = 1 - dot(RiJ, (L' \ (L \ RJi)))
        y = L \ zJ; δ = dot(y, y)
    end
    νp = ν + length(Jv); σz = sqrt(max(σ0², zero(σ0²))) * sqrt((ν + δ) / νp)
    return StudentDistortion(float(μz), float(σz), Int(ν), Int(νp))
end
@inline function ConditionalCopula(C::TCopula{D,df,MT}, js, uⱼₛ) where {D,df,MT}
    p = length(js); J = collect(Int64, js); I = collect(setdiff(1:D, J)); Σ = C.Σ
    if p == 1
        Σcond = Σ[I, I] - Σ[I, J] * (Σ[J, I] / Σ[J, J])
    else
        L = LinearAlgebra.cholesky(Symmetric(Σ[J, J]))
        Σcond = Σ[I, I] - Σ[I, J] * (L' \ (L \ Σ[J, I]))
    end

    # Subsetting colocated
    SubsetCopula(C::TCopula{d,df,MT}, dims::NTuple{p, Int64}) where {d,df,MT,p} = TCopula(df, C.Σ[collect(dims),collect(dims)])
    σ = sqrt.(LinearAlgebra.diag(Σcond)); R_cond = Matrix(Σcond ./ (σ * σ'))
    return TCopula(df + p, R_cond)
end