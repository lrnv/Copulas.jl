"""
    TCopula{d, df, MT}

Fields:
- `df::Int` — degrees of freedom
- `Σ::MT` — correlation matrix

Constructor

    TCopula(df, Σ)

The Student t copula is the copula of a multivariate Student t distribution. It is defined by

```math
C(\\mathbf{x}; \\nu, \\boldsymbol{\\Sigma}) = F_{\\nu,\\Sigma}(F_{\\nu,\\Sigma,1}^{-1}(x_1), \\ldots, F_{\\nu,\\Sigma,d}^{-1}(x_d)),
```

where ``F_{\\nu,\\Sigma}`` is the cdf of a centered multivariate t with correlation ``\\Sigma`` and ``\\nu`` degrees of freedom.

Example usage:
```julia
C = TCopula(2, Σ)
u = rand(C, 1000)
pdf(C, u); cdf(C, u)
Ĉ = fit(TCopula, u)
```

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
@inline function DistortionFromCop(C::TCopula{D,ν,MT}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {p,D,ν,MT}
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
    p = length(js); J = collect(Int, js); I = collect(setdiff(1:D, J)); Σ = C.Σ
    if p == 1
        Σcond = Σ[I, I] - Σ[I, J] * (Σ[J, I] / Σ[J, J])
    else
        L = LinearAlgebra.cholesky(Symmetric(Σ[J, J]))
        Σcond = Σ[I, I] - Σ[I, J] * (L' \ (L \ Σ[J, I]))
    end
    σ = sqrt.(LinearAlgebra.diag(Σcond))
    R_cond = Matrix(Σcond ./ (σ * σ'))
    return TCopula(df + p, R_cond)
end
# Subsetting colocated
SubsetCopula(C::TCopula{d,df,MT}, dims::NTuple{p, Int}) where {d,df,MT,p} = TCopula(df, C.Σ[collect(dims),collect(dims)])