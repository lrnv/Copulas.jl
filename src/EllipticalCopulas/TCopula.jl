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