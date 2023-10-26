struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C,m)
        d = length(C)
        @assert length(m) == d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
        return new{typeof(C),typeof(m)}(C,m)
    end    
end
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
  
function Distributions.cdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins}
    return Distributions.cdf(S.C,Distributions.cdf.(S.m,x))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, x::AbstractVector{T}) where {CT,TplMargins,T}
    Random.rand!(rng,S.C,x)
     x .= Distributions.quantile.(S.m,x)
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins},u) where {CT,TplMargins}
    sum(Distributions.logpdf(S.m[i],u[i]) for i in 1:length(u)) + Distributions.logpdf(S.C,Distributions.cdf.(S.m,u))
end


function Distributions.fit(::Type{SklarDist{CT,TplMargins}},x) where {CT,TplMargins}
    # The first thing to do is to fit the marginals : 
    @assert length(TplMargins.parameters) == size(x,1)
    m = Tuple(Distributions.fit(TplMargins.parameters[i],x[i,:]) for i in 1:size(x,1))
    u = pseudos(x)
    C = Distributions.fit(CT,u)
    return SklarDist(C,m)
end