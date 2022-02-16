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

