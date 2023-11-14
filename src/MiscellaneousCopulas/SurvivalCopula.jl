"""
SurvivalCopula{d,CT,VI}

Constructor

    SurvivalCopula(C,indices)

Computes the survival version of any copula on given indices. From a copula ``C`` in dimension ``d``, and some indices ``i_1,...i_k`` in ``{1,...,d}``, the survival copula associated simply reverses its arguments on chosen indices. For exemple, for ``d=4`` and indices ``(2,3)``, we have: 

```math
S(u_1,...u_4) = C(u_1,1-u_2,1-u3,u_4)
```

This constructor allows to derive new "survival" families. For exemple, in bivariate cases, this allows to do "rotations". The obtained models can be treated as the starting one, i.e. as a random vector in [0,1]^d with uniforms marginals. 
"""
struct SurvivalCopula{d,CT,VI} <: Copula{d}
    C::CT
    indices::VI
    function SurvivalCopula(C,indices)
        if typeof(C) == IndependentCopula
            return C
        end
        if length(indices) == 0
            return C
        end
        d = length(C)
        @assert all(indices .<= d)
        return new{d,typeof(C),typeof(indices)}(C,indices)
    end
end
function reverse!(u,idx)
    if ndims(u) == 1
        for i in idx
            u[i] = 1 - u[i]
        end
    else
        for i in idx
            u[i,:] .= 1 .- u[i,:]
        end
    end
    return u
end
function reverse(u,idx)
    v = deepcopy(u)
    reverse!(v,idx)
    return v
end
_cdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = _cdf(C.C,reverse(u,C.indices))
Distributions._logpdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions._logpdf(C.C,reverse(u,C.indices))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d,CT,VI}, x::AbstractVector{T}) where {d,CT,VI,T}
    Distributions._rand!(rng,C.C,x)
    reverse!(x,C.indices)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d,CT,VI}, A::DenseMatrix{T}) where {d,CT,VI,T}
    Distributions._rand!(rng,C.C,A)
    reverse!(A,C.indices)
end
function Distributions.fit(T::Type{CT},u) where {CT <: SurvivalCopula}
    # d = size(u,1)
    d,subCT,indices = T.parameters
    subfit = Distributions.fit(subCT,reverse(u,indices))
    return SurvivalCopula(subfit,indices)
end