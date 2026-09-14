"""
    BernsteinCopula(C; m=10)
    BernsteinCopula(data; m=10)
    BernsteinCopula{d}(C_or_data; m=10)
    BernsteinCopula(d, C_or_data; m=10)

The Bernstein copula in dimension ``d`` is defined as

``
B_m(C)(u) = \\sum_{s_1=0}^{m_1} \\cdots \\sum_{s_d=0}^{m_d}C\\left(\\tfrac{s_1}{m_1}, \\ldots, \\tfrac{s_d}{m_d}\\right)\\prod_{j=1}^d \\binom{m_j}{s_j} u_j^{s_j}(1-u_j)^{m_j-s_j}.
``

It is a polynomial approximation of the base copula ``C`` using the multivariate Bernstein operator.

Behavior and cost:
- The choice of `m` controls smoothness: larger values give a finer polynomial
  approximation but require work and memory proportional to ``\\prod_j m_j``.
  Large `d` or `m` can therefore be prohibitive.
- If ``C`` is an `EmpiricalCopula`, the constructor produces the *empirical Bernstein copula*, a smoothed version of the empirical copula.
- For an empirical sample of size ``n`` without ties, this construction is a
  genuine copula if and only if every degree ``m_j`` divides ``n``. Invalid
  degree choices are rejected. With `m=nothing`, the largest divisor of ``n``
  not exceeding ``\lfloor n^{1/d}\rfloor`` is selected in every dimension.
- Raw data supplied with `pseudo_values=false` must have tie-free margins.
  Resolve ties explicitly with `pseudos(data; ties=:first)`, `:last`, or
  `:random` before construction when deliberate tie breaking is scientifically
  justified.
- Supports `cdf`, `logpdf`, and random generation via mixtures of beta distributions.

See also: [`BetaCopula`](@ref), [`EmpiricalCopula`](@ref), [`Copula`](@ref),
[`Distributions.fit`](@ref).

References:
* [sancetta2004bernstein](@cite) Sancetta, A., & Satchell, S. (2004). The Bernstein copula and its applications to modeling and approximations of multivariate distributions. Econometric Theory, 20(3), 535-562.
* [segers2017](@cite) Segers, J., Sibuya, M., & Tsukahara, H. (2017). The empirical beta copula. Journal of Multivariate Analysis, 155, 35-51.
"""
struct BernsteinCopula{d} <: Copula{d}
    m::NTuple{d,Int}
    weights::Array{Float64, d}
    function BernsteinCopula{d}(base::Copula{d}; m::Union{Int,Tuple,Nothing}=10) where {d}
        if m === nothing && base isa EmpiricalCopula
            n = size(base.u, 2)
            target = max(1, floor(Int, n^(1 / d)))
            m_est = findlast(k -> iszero(n % k), 1:target)
            mtuple = ntuple(_ -> m_est, d)
        elseif base isa EmpiricalCopula
            n = size(base.u, 2)
            mtuple = _bernstein_degrees(m, d)
            all(iszero(n % mj) for mj in mtuple) || throw(ArgumentError(
                "each Bernstein degree must divide the empirical sample size n=$n; got m=$mtuple",
            ))
        else
            mtuple = _bernstein_degrees(something(m, 10), d)
        end
        # Compute measures values using multidimensional finite differences on the grid of cdf values. 
        weights = Array{Float64}(undef, (mi+1 for mi in mtuple)...)
        for idx in CartesianIndices(weights)
            u = ntuple(j -> (idx[j]-1) / mtuple[j], d)
            weights[idx] = Distributions.cdf(base, collect(u))
        end
        for axis in 1:d
            weights = Base.diff(weights, dims=axis)
        end
        _validate_bernstein_weights(weights, mtuple)
        return new{d}(mtuple, weights)
    end
    function BernsteinCopula{d}(m::NTuple{d,Int}, weights::Array{Float64,d}) where {d}
        size(weights) == m || throw(DimensionMismatch(
            "weights must have size m=$m; got $(size(weights))",
        ))
        _validate_bernstein_weights(weights, m)
        return new{d}(m, weights)
    end
end

function _bernstein_degrees(m::Union{Int,Tuple}, d::Int)
    mtuple = m isa Int ? ntuple(_ -> m, d) : m
    length(mtuple) == d || throw(DimensionMismatch("m must have length $d"))
    all(mj -> mj isa Int && mj > 0, mtuple) ||
        throw(ArgumentError("Bernstein degrees must be positive integers; got m=$mtuple"))
    return ntuple(j -> Int(mtuple[j]), d)
end

function _validate_bernstein_weights(weights::AbstractArray, m::Tuple)
    scale = max(1.0, maximum(abs, weights))
    atol = 100 * eps(Float64) * length(weights) * scale
    minimum(weights) >= -atol || throw(ArgumentError(
        "the Bernstein coefficients do not define a probability distribution",
    ))
    isapprox(sum(weights), 1.0; atol, rtol=0) || throw(ArgumentError(
        "the Bernstein coefficients do not sum to one",
    ))

    d = length(m)
    for j in 1:d
        other_dims = Tuple(setdiff(1:d, (j,)))
        marginal = isempty(other_dims) ? weights :
            dropdims(sum(weights; dims=other_dims); dims=other_dims)
        all(x -> isapprox(x, inv(m[j]); atol, rtol=0), marginal) ||
            throw(ArgumentError(
                "the Bernstein construction does not have a uniform margin in dimension $j",
            ))
    end
    return nothing
end
Distributions.params(C::BernsteinCopula) = (m=C.m, weights=C.weights)
BernsteinCopula(base::Copula{d}; kwargs...) where {d} = BernsteinCopula{d}(base; kwargs...)
BernsteinCopula(d::Integer, base::Copula; kwargs...) = BernsteinCopula{d}(base; kwargs...)
function BernsteinCopula{d}(data::AbstractMatrix; kwargs...) where {d}
    size(data, 1) == d || throw(DimensionMismatch("data must have $d rows"))
    pseudo_values = get(kwargs, :pseudo_values, true)
    pseudo_values || _require_tie_free_rows(data, "BernsteinCopula")
    return BernsteinCopula{d}(EmpiricalCopula{d}(data; pseudo_values=pseudo_values);
                              m=get(kwargs, :m, nothing))
end
BernsteinCopula(data::AbstractMatrix; kwargs...) = BernsteinCopula{size(data, 1)}(data; kwargs...)
BernsteinCopula(d::Integer, data::AbstractMatrix; kwargs...) = BernsteinCopula{d}(data; kwargs...)

@inline function _bernvec_all(u::T, m::Int) where {T<:Real}
    v = zeros(T, m+1)
    if iszero(u)
        v[1] = 1; return v
    elseif isone(u)
        v[end] = 1; return v
    end
    inv1mu = 1 - u
    r = u / inv1mu
    p = inv1mu^m
    v[1] = p
    @inbounds for s in 1:m
        p *= ((m - s + 1) / s) * r
        v[s+1] = p
    end
    return v
end
@inline function _betavec_pdf_all(u::T, m::Int) where {T<:Real}
    v = zeros(T, m)
    if iszero(u)
        v[1] = m; return v
    elseif isone(u)
        v[m] = m; return v
    end
    inv1mu = 1 - u
    r = u / inv1mu
    q = inv1mu^(m-1)
    v[1] = q
    @inbounds for s in 1:m-1   # s = k+1, k=0..m-2
        q *= ((m - s) / s) * r
        v[s+1] = q
    end
    return v .* m
end
function _cdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    m = B.m
    P = ntuple(j -> _bernvec_all(u[j], m[j]), d)
    total = zero(eltype(first(P)))
    @inbounds for s in Iterators.product((0:mi for mi in m)...)
        w = 0.0
        for t in Iterators.product((1:s[j] for j in 1:d)...)
            w += B.weights[t...]
        end
        iszero(w) && continue
        total += w * prod(P[j][s[j]+1] for j in 1:d)
    end
    return total
end

function Distributions._logpdf(B::BernsteinCopula{d}, u::AbstractVector) where {d}
    m = B.m
    BetaV = ntuple(j -> _betavec_pdf_all(u[j], m[j]), d)
    dens = zero(eltype(first(BetaV)))
    weights = B.weights
    @inbounds for s in Iterators.product((0:(mi-1) for mi in m)...)
        w = weights[(s[j]+1 for j in 1:d)...]
        iszero(w) && continue
        dens += w * prod(BetaV[j][s[j]+1] for j in 1:d)
    end
    return dens > zero(dens) ? log(dens) : oftype(dens, -Inf)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, B::BernsteinCopula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    m = B.m
    weights = max.(vec(B.weights), 0.0)
    components = rand(rng, Distributions.Categorical(weights ./ sum(weights)), size(A, 2))
    indices = CartesianIndices(B.weights)
    @inbounds for (j, col) in enumerate(axes(A, 2))
        s = Tuple(indices[components[j]])
        for row in axes(A, 1)
            A[row, col] = rand(rng, Distributions.Beta(s[row], m[row] + 1 - s[row]))
        end
    end
    return A
end


function distortion(B::BernsteinCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,p}
    # Build mixture weights over s_i given fixed u_J for J = js.
    Iset = Tuple(setdiff(1:D, js))
    @assert i in Iset "i must refer to a non-conditioned coordinate"
    m = B.m
    mi = m[i]
    α = zeros(Float64, mi)
    # Iterate over s on the grid
    for s in Iterators.product((0:(mj-1) for mj in m)...)
        wJ = 1.0
        @inbounds for (t, j) in pairs(js)
            wJ *= Distributions.pdf(Distributions.Beta(s[j] + 1, m[j] - s[j]), uⱼₛ[t])
            wJ == 0.0 && break
        end
        wJ == 0.0 && continue
        Δ = B.weights[(s[j]+1 for j in 1:D)...]
        (Δ <= 0) && continue
        α[s[i] + 1] += Δ * wJ
    end
    sα = sum(α)
    if sα <= 0
        return NoDistortion()
    end
    α ./= sα
    comps = [Distributions.Beta(k, mi - (k - 1)) for k in 1:mi]
    return BernsteinDistortion(Distributions.MixtureModel(comps, α))
end

# Fitting colocated. 
StatsBase.dof(::BernsteinCopula) = 0
_available_fitting_methods(::Type{<:BernsteinCopula}, d) = (:bernstein,)
"""
    _fit(::Type{<:BernsteinCopula}, U, ::Val{:bernstein};
         m::Union{Int,Tuple,Nothing}=nothing, pseudo_values::Bool=true, kwargs...) -> (C, meta)

Empirical plug-in fitting of `BernsteinCopula` based on `U`, using the empirical copula and (optionally) a degree `m` per dimension.

# Arguments
- `U::AbstractMatrix`: `d×n` pseudo-observations (if `pseudo_values=true`) or raw data.
- `m`: integer (same degree in all coordinates), tuple of degrees per dimension,
or `nothing` for automatic selection.
- `pseudo_values`: if `false`, pseudo-observations are constructed with `pseudos(U)`.
- `kwargs...`: forwarded to the `BernsteinCopula` constructor.

# Returns
- `(C, meta)` where `C::BernsteinCopula` and
`meta = (; emp_kind = :bernstein, pseudo_values, m = C.m)`.

**Note**: Method with no free parameters (`dof=0`).
"""
function _fit(::Type{<:BernsteinCopula}, U, ::Val{:bernstein};
              m::Union{Int,Tuple,Nothing}=nothing,
              pseudo_values::Bool=true, kwargs...)
    C = BernsteinCopula(U; m=m, pseudo_values=pseudo_values, kwargs...)
    return C, (; emp_kind=:bernstein, pseudo_values, m=C.m)
end

function SubsetCopula(C::BernsteinCopula{d}, dims::NTuple{p, Int}) where {d,p}
    # dims: indices to keep, e.g. (1,3) for a 3D copula
    # Step 1: Permute axes so that kept dims are first
    permuted_weights = PermutedDimsArray(C.weights, vcat(collect(dims), setdiff(1:d, dims)))
    # Step 2: Sum over trailing axes (those not in dims)
    new_m = ntuple(i -> C.m[dims[i]], p)
    rmdims = tuple(((p+1):d)...)
    return BernsteinCopula{p}(new_m, dropdims(sum(permuted_weights, dims=rmdims), dims=rmdims))
end
