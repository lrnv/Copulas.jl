
###############################################################################
#####  SklarDist framework.
#####  User-facing function: `SklarDist(C::Copula{d}, m::NTuple{d, <:UnivariateDistribution}) where d`
#####
#####  Nothing here should be overwritten when defining new copulas. 
###############################################################################

"""
    SklarDist{CT,TplMargins} 

Fields:
  - `C::CT` - The copula
  - `m::TplMargins` - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

Construct a joint distribution via Sklar's theorem from marginals and a copula. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem):

!!! theorem "Sklar 1959"
    For every random vector ``\\boldsymbol X``, there exists a copula ``C`` such that 

    ``\\forall \\boldsymbol x\\in \\mathbb R^d, F(\\boldsymbol x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).``
    The copula ``C`` is uniquely determined on ``\\mathrm{Ran}(F_{1}) \\times ... \\times \\mathrm{Ran}(F_{d})``, where ``\\mathrm{Ran}(F_i)`` denotes the range of the function ``F_i``. In particular, if all marginals are absolutely continuous, ``C`` is unique.


The resulting random vector follows the `Distributions.jl` API (rand/cdf/pdf/logpdf). A `fit` method is also provided. Example:

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset

# You may estimate a copula using the `fit` function:
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
```

References: 
* [sklar1959](@cite) Sklar, M. (1959). Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP (Vol. 8, No. 3, pp. 229-231).
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
        return new{typeof(C),typeof(m)}(C,m)
    end    
end
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
Distributions.cdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins} = Distributions.cdf(S.C,Distributions.cdf.(S.m,x))
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, x::AbstractVector{T}) where {CT,TplMargins,T}
    Random.rand!(rng,S.C,x)
     x .= Distributions.quantile.(S.m,x)
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins},u) where {CT,TplMargins}
    sum(Distributions.logpdf(S.m[i],u[i]) for i in eachindex(u)) + Distributions.logpdf(S.C,clamp.(Distributions.cdf.(S.m,u),0,1))
end
function StatsBase.dof(S::SklarDist)
    a = StatsBase.dof(S.C)
    b = sum(hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d)) for d in S.m)
    return a+b
end

function _local_blockdiag(Vs::AbstractMatrix...)
    nb = length(Vs)
    nb == 0 && return Matrix{Float64}(undef, 0, 0)
    nb == 1 && return Matrix{Float64}(Vs[1])
    Bs = map(V -> Matrix{Float64}(V), Vs)
    rs = cumsum(vcat(0, map(B -> size(B,1), Bs)))
    N = rs[end]
    M = zeros(Float64, N, N)
    @inbounds for (k, B) in enumerate(Bs)
        r = (rs[k] + 1) : rs[k+1]
        M[r, r] .= B
    end
    return M
end

function _assemble_vcov_sklar(cmeta, margins, sklar_method::Symbol; Vm_hint=nothing)
    # Copula
    Vcop = get(cmeta, :vcov, nothing)
    Vcop = (Vcop === nothing || isempty(Vcop)) ? nothing : Matrix{Float64}(Vcop)

    # Márgenes
    d  = length(margins)
    Vm = Vector{Union{Nothing, Matrix{Float64}}}(undef, d)

    _is_valid_cov(V) =
        V !== nothing &&
        V isa AbstractMatrix &&
        ndims(V) == 2 &&
        size(V,1) == size(V,2) &&
        all(isfinite, Matrix(V)) &&
        all(diag(Matrix(V)) .>= 0)

    @inbounds for i in 1:d
        Vi = nothing

        # 1) generic data from fit
        if Vm_hint !== nothing && Vm_hint isa AbstractVector && i <= length(Vm_hint)
            Vh = Vm_hint[i]
            if _is_valid_cov(Vh)
                Vi = Matrix{Float64}(Vh)
            end
        end

        # 2) test vcov from maginal fit
        if Vi === nothing
            try
                V0 = StatsBase.vcov(margins[i])
                if _is_valid_cov(V0)
                    Vi = Matrix{Float64}(V0)
                end
            catch
                # no-op
            end
        end

        Vm[i] = Vi
    end

    if sklar_method == :ifm
        blocks = Matrix{Float64}[]
        if Vcop !== nothing; push!(blocks, Vcop) end
        for Vi in Vm
            if Vi !== nothing; push!(blocks, Vi) end
        end
        Vfull = isempty(blocks) ? nothing : _local_blockdiag(blocks...)
        return Vcop, Vm, Vfull
    else
        return Vcop, Vm, Vcop
    end
end
# objetive this functions: try get the vcov from marginals...
function _vcov_margin_generic(d::Distributions.UnivariateDistribution, x::AbstractVector; ridge::Real=1e-8)
    p_nt = Distributions.params(d)
    if p_nt isa NamedTuple
        names = collect(keys(p_nt))
        θ0    = Float64.(collect(values(p_nt)))
    else
        names = [Symbol(:θ, i) for i in 1:length(p_nt)]  # pseudo-names
        θ0    = Float64.(collect(p_nt))
    end
    p = length(θ0)

    POS = Set([:σ, :theta, :θ, :α, :alpha, :β, :beta, :k, :λ, :nu, :ν, :η, :ω, :rate, :scale])

    to_uncon(v, name)   = (name in POS) ? log(v) : v
    from_uncon(a, name) = (name in POS) ? exp(a) : a
    jac_diag(a, name)   = (name in POS) ? exp(a) : 1.0

    α0 = [to_uncon(θ0[i], names[i]) for i in 1:p]
    # reconstruct distributions with params in the same order
    function dist_from_α(α)
        pars = ntuple(i -> from_uncon(α[i], names[i]), p)
        return (typeof(d))(pars...)
    end

    function ℓ(α)
        di = dist_from_α(α)
        s = zero(eltype(α))
        @inbounds @simd for xi in x
            s += logpdf(di, xi)
        end
        return s
    end

    Hα = try
        ForwardDiff.hessian(ℓ, α0)
    catch
        return LinearAlgebra.Symmetric(fill(NaN, p, p))
    end

    infoα = -Array(Hα)
    if any(!isfinite, infoα)
        return LinearAlgebra.Symmetric(fill(NaN, p, p))
    end
    infoα .+= ridge .* I
    Vα = try
        inv(infoα)
    catch
        return LinearAlgebra.Symmetric(fill(NaN, p, p))
    end

    Jdiag = [jac_diag(α0[i], names[i]) for i in 1:p]
    J = LinearAlgebra.Diagonal(Jdiag)
    Vθ = (J * Vα * J')
    Vθ = (Vθ + Vθ')/2
    return LinearAlgebra.Symmetric(Matrix{Float64}(Vθ))
end
