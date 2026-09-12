
###############################################################################
#####  Subsetting framework.
#####  User-facing function: `subsetdims()`
#####
#####  When implementing a new copula, you can overwrite: 
#####   - `SubsetCopula(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}`
###############################################################################
"""
    SubsetCopula(C::Copula, dims)

Internal fallback representing the marginal copula of `C` on the coordinates
listed by `dims`. Coordinate order is significant, indices must be distinct
and valid, and selecting one coordinate returns a uniform distribution rather
than a `SubsetCopula`. Sampling projects samples from `C`; CDF evaluation fixes
discarded coordinates at one.

Families may specialize this constructor to return a closed-form copula of the
same marginal law. Such specializations must preserve the requested coordinate
order and all distribution semantics. Public code should call `subsetdims`;
the wrapper type and its fields are internal and unstable.

See also: [`subsetdims`](@ref), [`conditional_copula`](@ref), [`_cdf`](@ref).
"""
struct SubsetCopula{d,CT} <: Copula{d}
    C::CT
    dims::NTuple{d,Int}
    function SubsetCopula{p}(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}

        # p == d is allowed: a `dims` that is a (non-identity) permutation reorders the
        # coordinates. The identity `dims == 1:d` is already returned above.
        @assert 1 <= p <= d "You cannot construct a subsetcopula with dimension p < 1 or p > d (d = $d, p = $p provided)"
        dims == Tuple(1:d) && return C
        @assert all(i -> 1 <= i <= d, dims)
        @assert p <= d
        @assert length(unique(dims))==length(dims)
        p==1 && return Distributions.Uniform()
        return new{p, typeof(C)}(C,Tuple(Int.(dims)))
    end
end
copula_measure_style(::Type{<:SubsetCopula{d,CT}}) where {d,CT} =
    copula_measure_style(CT)
copula_measure_style(C::SubsetCopula) = copula_measure_style(C.C)
SubsetCopula(C::Copula, dims::NTuple{p,Int}) where {p} = SubsetCopula{p}(C, dims)
function SubsetCopula(CS::SubsetCopula{d,CT}, dims2::NTuple{p, Int}) where {d,CT,p}
    return SubsetCopula{p}(CS.C, ntuple(i -> CS.dims[dims2[i]], p))
end
_available_fitting_methods(::Type{<:SubsetCopula}, d) = Tuple{}() # cannot be fitted. 
Base.eltype(C::SubsetCopula{d,CT}) where {d,CT} = Base.eltype(C.C)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SubsetCopula{d,CT}, A::AbstractMatrix{T}) where {T<:Real, d,CT}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    U = rand(rng, C.C, size(A, 2))
    @inbounds for (row, source) in enumerate(C.dims), col in axes(A, 2)
        A[row, col] = U[source, col]
    end
    return A
end
function _cdf(C::SubsetCopula{d,CT},u) where {d,CT}
    # Simplyu saturate dimensions that are not choosen.
    v = ones(eltype(u), length(C.C))
    for (i,j) in enumerate(C.dims)
        v[j] = u[i]
    end 
    return Distributions.cdf(C.C,v)
end
function Distributions._logpdf(S::SubsetCopula{d,<:Copula{D}}, u) where {d,D}
    return log(_partial_cdf(S.C, Tuple(setdiff(1:D, S.dims)), S.dims, ones(D-d), u))
end

# Dependence metrics are symetric in bivariate cases: 
τ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = τ(C.C)
ρ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ρ(C.C)
β(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = β(C.C)
γ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = γ(C.C)
ι(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ι(C.C)
λₗ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λₗ(C.C)
λᵤ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λᵤ(C.C)

"""
    subsetdims(C::Copula, dims::NTuple{p, Int})
    subsetdims(D::SklarDist, dims)

Return the marginal distribution on the coordinates selected by `dims`.

Indices are one-based, distinct, and order-sensitive: `(3, 1)` both selects and
reorders coordinates. For a copula, selecting one coordinate returns the
uniform marginal; for a `SklarDist`, it returns the corresponding original
marginal. Selecting several coordinates preserves their joint marginal law and
returns a copula or Sklar distribution of that dimension. Selecting every
coordinate in natural order may return the original object.

The result can be a family-specific closed form or an internal generic wrapper;
callers should rely on its distribution behavior rather than its concrete type.
Invalid, repeated, or empty index collections are rejected.

# Example
```julia
C = GaussianCopula(3, [1.0 0.2 0.6; 0.2 1.0 0.4; 0.6 0.4 1.0])
C31 = subsetdims(C, (3, 1))
length(C31) == 2
```

See also: [`condition`](@ref), [`SklarDist`](@ref), [`measure`](@ref).
"""
function subsetdims(C::Copula{d}, dims::NTuple{p,Int}) where {d,p}
    # Validate the public operation before dispatching to a native submodel:
    # specialized `SubsetCopula(C, dims)` methods may assume valid indices.
    @assert 1 <= p <= d "You cannot construct a subsetcopula with dimension p < 1 or p > d (d = $d, p = $p provided)"
    @assert all(i -> 1 <= i <= d, dims)
    @assert length(unique(dims)) == p
    dims == Tuple(1:d) && return C
    p == 1 && return Distributions.Uniform()
    return SubsetCopula(C, dims)
end
function subsetdims(D::SklarDist, dims::NTuple{p, Int}) where p
    p==1 && return D.m[dims[1]] # if dims[1] is not a valid index, this will throw.
    return SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))
end
subsetdims(C::Union{Copula, SklarDist}, dims) = subsetdims(C, Tuple(collect(Int, dims)))

# Pairwise dependence metrics, leveraging subsetting: 
function _as_biv(f::F, C::Copula{d}) where {F, d}
    first_val = f(SubsetCopula(C, (1,2)))
    K = ones(eltype(first_val),d,d)
    K[1,2] = first_val
    K[2,1] = first_val
    for i in 1:d
        for j in i+1:d
            if (i,j) != (1,2)
                K[i,j] = f(SubsetCopula(C, (i,j)))
                K[j,i] = K[i,j]
            end
        end
    end
    return K
end
StatsBase.corkendall(C::Copula)  = _as_biv(τ, C)
StatsBase.corspearman(C::Copula) = _as_biv(ρ, C)
corblomqvist(C::Copula)          = _as_biv(β, C)
corgini(C::Copula)               = _as_biv(γ, C)
corentropy(C::Copula)            = _as_biv(ι, C) - LinearAlgebra.I
coruppertail(C::Copula)          = _as_biv(λᵤ, C)
corlowertail(C::Copula)          = _as_biv(λₗ, C)
