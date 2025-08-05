struct ConditionalDistribution{d,p,D,T,TD} <: Distributions.ContinuousMultivariateDistribution
    X::TD
    dims::Ntuple{p,Int64}
    xs::NTuple{p, T}
    function ConditionalDistribution(X,dims,xs)
        
        # Get dimensions: 
        D = length(rand(X))
        p = length(xs)
        d = D - p
        @assert length(dims) == p
        @assert all(dims .<= D)
        xs = Tuple(xs...)
        T = eltype(xs)
        TD = typeof(X)
        return new{d,p,D,T,TD}(X,dims,xs)
    end
end

function ConditionalDistribution(X::SklarDist{TC,Tm},dims,xs) where {TC<:GaussianCopula, Tm}
    # If the copula is gaussian, then the conditional distribution has a closed form formula. 
    # It is also a SklarDist with a Gaussian copula, but not exactly the same one. 
    # invert the xs to z-scale
    Z = Normal()
    zs = similar(xs)
    for (i,di) in enumerate(dims)
        zs[i] = quantile(Z,cdf(X.m[di],xs[i]))
    end

    # Now we simply need to condition a gaussian random vector and apply back 
end


function ConditionalDistribution(X::SklarDist{IndependentCopula{d},Tm},dims,xs) where {d,Tm}
    # If the copula is the independence, conditionning is just subsetting. 
    otherdims = (i for i in d if !(i in dims))
    return subsetdims(X,otherdims)
end

function _v(u,j,uj)
    return [(i == j ? uj : u[i]) for i in eachindex(u)]
end
function _der(X,dims,u)
    if length(dims)==1
        j = dims[1]
        return ForwardDiff.derivative(uj -> cdf(X,_v(u,j,uj)), u[j])
    else
        j = pop!(dims)
        return ForwardDiff.derivative(uj -> _der(X,dims,_v(u,j,uj)), u[j])
    end
end
function Distributions.cdf(X::ConditionalDistribution{d,p,D,T,TD},u) where {d,p,D,T,TD}
    # So we need the derivative of the original cdf. 
    
    # Make the full vector x :
    x = zeros(D)    
    j = 1
    for i in 1:D
        if !(i in X.dims)
            x[i] = u[j]
            j += 1
        end
    end
    x[X.dims...] .= X.xs

    # Now derivate the cdf: 
    return _der(C.C, C.dims, x)
end 