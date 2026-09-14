"""
    CopulaInference

Result of applying one uncertainty-quantification procedure to a fitted
[`CopulaModel`](@ref). It stores the model, the inference method, its covariance
matrix, method-specific diagnostics, and the parameter-block decomposition
needed by composite models such as [`SklarDist`](@ref). Construct one with
[`infer`](@ref); its concrete fields remain implementation details.

Inference objects are immutable and independent: several procedures can be
applied to the same fitted model without mutating it.
"""
struct CopulaInference{M<:CopulaModel,V<:AbstractMatrix,D<:NamedTuple,B<:NamedTuple}
    model::M
    method::Symbol
    covariance::V
    diagnostics::D
    blocks::B
end

####### Analytical inference kernels.

@inline function _vcov_copula(CT, ::Val{d}, α, example) where {d}
    return _construct_fitted_copula(CT, Val(d), _rebound_params(CT, d, α), example)
end

function _vcov_upper_triangle(A)
    return [A[idx] for idx in CartesianIndices(A) if idx[1] < idx[2]]
end

_vcov_dependence_measure(::Val{:itau}) = τ
_vcov_dependence_measure(::Val{:irho}) = ρ
_vcov_dependence_measure(::Val{:ibeta}) = β
_vcov_dependence_measure(::Val) = λᵤ

_vcov_pairwise_measure(::Val{:itau}) = StatsBase.corkendall
_vcov_pairwise_measure(::Val{:irho}) = StatsBase.corspearman
_vcov_pairwise_measure(::Val{:ibeta}) = corblomqvist
_vcov_pairwise_measure(::Val) = coruppertail

function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
               vcovv::Val{:hessian}, methodv::Val{method}) where {method}
    return _vcov_hessian(CT, U, θ, Val(size(U, 1)), vcovv, methodv)
end

function _vcov_hessian(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
                       ::Val{d}, ::Val{:hessian},
                       methodv::Val{method}) where {d,method}
    α = _unbound_params(CT, d, θ)
    example = _example(CT, d)
    vd = Val(d)
    ℓ(αv) = Distributions.loglikelihood(_vcov_copula(CT, vd, αv, example), U)
    H = ForwardDiff.hessian(ℓ, α)
    Iα = .-H
    any(!isfinite, Iα) && throw(ArgumentError(
        "Hessian inference produced non-finite observed information"))
    Iα = (Iα + Iα') / 2
    p = size(Iα, 1)
    I_p = Matrix{Float64}(LinearAlgebra.I, p, p)
    λ = 1e-8
    Vα = nothing
    @inbounds for _ in 1:8
        ch = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Iα + λ * I_p);
                                    check=false)
        if ch.info == 0
            Vα = ch \ I_p
            break
        end
        λ *= 10
    end
    (Vα === nothing || any(!isfinite, Vα)) && throw(ArgumentError(
        "Hessian inference could not stabilize the observed information"))
    return _vcov_finalize(CT, U, θ, d, α, Vα, Val(:hessian), methodv;
                          observed_information_ridge=λ)
end

function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
               ::Val{:godambe}, methodv::Val{method}) where {method}
    return _vcov_godambe(CT, U, θ, Val(false), Val(:godambe), methodv)
end

function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
               ::Val{:godambe_pairwise}, methodv::Val{method}) where {method}
    return _vcov_godambe(CT, U, θ, Val(true), Val(:godambe_pairwise), methodv)
end

function _vcov_godambe(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
                       pairwisev::Val{pairwise}, vcovv::Val{vcovm},
                       methodv::Val{method}) where {pairwise,vcovm,method}
    return _vcov_godambe(CT, U, θ, Val(size(U, 1)), pairwisev, vcovv, methodv)
end

function _vcov_godambe(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
                       ::Val{d}, ::Val{pairwise}, vcovv::Val{vcovm},
                       methodv::Val{method}) where {d,pairwise,vcovm,method}
    n = size(U, 2)
    α = _unbound_params(CT, d, θ)
    example = _example(CT, d)
    vd = Val(d)
    φ = _vcov_dependence_measure(methodv)
    if pairwise
        pairwise_φ = _vcov_pairwise_measure(methodv)
        q = d * (d - 1) ÷ 2
        Dα = ForwardDiff.jacobian(
            αv -> _vcov_upper_triangle(pairwise_φ(_vcov_copula(CT, vd, αv, example))),
            α)
        Dα = reshape(Dα, q, length(α))
        B = clamp(Int(floor(sqrt(n))), 10, 200)
        M = Matrix{Float64}(undef, B, q)
        idx = Vector{Int}(undef, n)
        rng = Random.default_rng()
        @inbounds for b in 1:B
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end
            M[b, :] .= _vcov_upper_triangle(pairwise_φ((@view U[:, idx])'))
        end
    else
        q = 1
        Dα = ForwardDiff.jacobian(
            αv -> [φ(_vcov_copula(CT, vd, αv, example))], α)
        Dα = reshape(Dα, q, length(α))
        B = clamp(Int(floor(sqrt(n))), 10, 200)
        M = Matrix{Float64}(undef, B, q)
        idx = Vector{Int}(undef, n)
        rng = Random.default_rng()
        @inbounds for b in 1:B
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end
            M[b, 1] = φ(@view U[:, idx])
        end
    end
    Ω = n * Statistics.cov(M; corrected=true)
    DtD = Dα' * Dα
    stabilized_inv = inv(DtD + 1e-10LinearAlgebra.I)
    Vα = stabilized_inv * (Dα' * Ω * Dα) * stabilized_inv / n
    return _vcov_finalize(CT, U, θ, d, α, Vα, vcovv, methodv)
end

function _vcov_finalize(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple,
                        d::Int, α, Vα, ::Val{vcovm},
                        methodv::Val{method};
                        observed_information_ridge=nothing) where {vcovm,method}
    J = ForwardDiff.jacobian(
        αv -> _flatten_params(_rebound_params(CT, d, αv))[2], α)
    Vθ = J * Vα * J'
    all(isfinite, Vθ) || throw(ArgumentError(
        "inference produced a non-finite covariance matrix"))
    Vθ = (Vθ + Vθ') / 2
    λ, Q = LinearAlgebra.eigen(Matrix(Vθ))
    λ_reg = map(x -> max(x, 1e-12), λ)
    Vθ = LinearAlgebra.Symmetric(Q * LinearAlgebra.Diagonal(λ_reg) * Q')
    all(isfinite, Matrix(Vθ)) || throw(ArgumentError(
        "inference produced a non-finite regularized covariance matrix"))
    return Vθ, (; eigenvalue_floor=1e-12, observed_information_ridge)
end

function _default_inference_method(M::CopulaModel)
    M.result isa SklarDist && return :bootstrap
    spec = M.recipe
    parameters = Distributions.params(fitteddistribution(M))
    d = length(fitteddistribution(M))
    analytical_coordinates = spec isa _CopulaFitSpec && spec.target isa Type &&
        parameters isa NamedTuple &&
        applicable(_unbound_params, spec.target, d, parameters)
    if fitting_method(M) === :mle && analytical_coordinates &&
            !(M.result isa Union{TCopula,tEVCopula,FGMCopula})
        return :hessian
    end
    if fitting_method(M) in (:itau, :irho, :ibeta, :iupper) && analytical_coordinates
        return :godambe
    end
    throw(ArgumentError(
        "no default covariance estimator is defined for fits using method=$(fitting_method(M)); " *
        "choose an explicit supported inference method"))
end

function _inference_inputs(M::CopulaModel)
    spec = M.recipe
    spec isa _CopulaFitSpec || throw(ArgumentError(
        "this model does not store a reproducible fitting specification"))
    data = M.data
    data isa AbstractMatrix || throw(ArgumentError(
        "the fitting observations required for inference are unavailable"))
    parameters = Distributions.params(_copula_of(M))
    parameters isa NamedTuple && !isempty(parameters) || throw(ArgumentError(
        "no finite-dimensional free parameter vector is available for inference"))
    return spec.target, data, parameters
end

function _resampling_covariance(M::CopulaModel, indices; rng, nresamples)
    p = StatsBase.dof(M)
    estimates = Matrix{Float64}(undef, nresamples, p)
    for b in 1:nresamples
        sample = indices(rng)
        estimates[b, :] .= StatsBase.coef(_refit(M, sample; replay_input=true))
    end
    return Statistics.cov(estimates; corrected=true)
end

function _infer(M::CopulaModel, ::Val{:bootstrap};
                rng=Random.default_rng(), nresamples::Integer=200)
    nresamples > 1 || throw(ArgumentError("nresamples must be greater than one"))
    _, data, _ = _inference_inputs(M)
    n = size(data, 2)
    rng_state = copy(rng)
    sample(rng) = @view data[:, rand(rng, 1:n, n)]
    V = _resampling_covariance(M, sample; rng, nresamples)
    return V, (; nresamples, rng_state)
end

function _infer(M::CopulaModel, ::Val{:jackknife})
    _, data, _ = _inference_inputs(M)
    n = size(data, 2)
    n > 1 || throw(ArgumentError("jackknife inference requires at least two observations"))
    p = StatsBase.dof(M)
    estimates = Matrix{Float64}(undef, n, p)
    keep = Vector{Int}(undef, n - 1)
    for omitted in 1:n
        k = 1
        for j in 1:n
            j == omitted && continue
            keep[k] = j
            k += 1
        end
        subset = @view data[:, keep]
        estimates[omitted, :] .= StatsBase.coef(
            _refit(M, subset; replay_input=true))
    end
    center = vec(Statistics.mean(estimates; dims=1))
    deviations = estimates .- center'
    V = (n - 1) / n .* (deviations' * deviations)
    return V, (; nreplicates=n)
end

function _infer(M::CopulaModel, ::Val{method}) where {method}
    method in (:hessian, :godambe, :godambe_pairwise) || throw(ArgumentError(
        "unknown inference method `$method`; expected :hessian, :godambe, " *
        ":godambe_pairwise, :jackknife, or :bootstrap"))
    M.result isa SklarDist && throw(ArgumentError(
        "analytical `$method` inference is not defined for Sklar estimators; " *
        "use :bootstrap or :jackknife to refit the complete margins-and-copula procedure"))
    spec = M.recipe
    (spec isa _CopulaFitSpec && spec.target isa Type) || throw(ArgumentError(
        "analytical `$method` inference is unavailable for runtime-structured fitting targets; " *
        "use :bootstrap or :jackknife"))
    method === :hessian && fitting_method(M) !== :mle && throw(ArgumentError(
        "Hessian inference is defined only for maximum-likelihood fits"))
    method in (:godambe, :godambe_pairwise) &&
        !(fitting_method(M) in (:itau, :irho, :ibeta, :iupper)) &&
        throw(ArgumentError(
            "Godambe inference is currently defined only for supported rank-matching fits; " *
            "analytical maximum pseudo-likelihood inference is unavailable, so use " *
            "method=:bootstrap or method=:jackknife when appropriate"))
    C = M.result
    method === :hessian && C isa Union{TCopula,tEVCopula} && throw(ArgumentError(
        "Hessian inference is unavailable because incomplete-beta derivatives are not implemented"))
    method === :hessian && C isa FGMCopula && throw(ArgumentError(
        "Hessian inference is not implemented for maximum-likelihood FGM fits"))
    target, _, parameters = _inference_inputs(M)
    U = _copula_data(M)
    d = size(U, 1)
    applicable(_unbound_params, target, d, parameters) || throw(ArgumentError(
        "analytical `$method` inference is not implemented for fitting target $target"))
    engine_method = fitting_method(M) === :mpl ? :mle : fitting_method(M)
    V, diagnostics = _vcov(target, U, parameters, Val(method), Val(engine_method))
    return V, diagnostics
end

"""
    infer(M::CopulaModel; method=:default, kwargs...) -> CopulaInference

Apply an uncertainty-quantification procedure after estimation. `fit` is never
rerun except by resampling procedures, and `M` is not mutated.

The principled default is `:hessian` for supported maximum-likelihood fits and
`:godambe` for supported rank-matching estimators. A fitting extension does not
acquire analytical inference merely by implementing a fitting route. Fits
without a justified default raise an `ArgumentError`. Explicit methods are
`:hessian`, `:godambe`,
`:godambe_pairwise`, `:jackknife`, and `:bootstrap`. Bootstrap inference accepts
`nresamples` and `rng` and records their provenance in the result diagnostics.

For a fitted `SklarDist`, the default is `:bootstrap`. Every resample repeats
the complete estimator: all margins are fitted again, pseudo-observations are
recomputed, and the copula is refitted. The resulting covariance therefore
contains marginal, copula, and cross-component uncertainty. Analytical methods
remain unavailable because the estimators selected by `Distributions.fit` for
arbitrary marginal families do not share a common derivative contract.

See also: [`CopulaInference`](@ref), [`StatsBase.vcov`](@ref),
[`StatsBase.stderror`](@ref), [`StatsBase.confint`](@ref).
"""
function infer(M::CopulaModel; method::Symbol=:default, kwargs...)
    selected = method === :default ? _default_inference_method(M) : method
    V, diagnostics = _infer(M, Val(selected); kwargs...)
    covariance = LinearAlgebra.Symmetric(Matrix{Float64}(V))
    all_parameters = axes(covariance, 1)
    blocks = _parameter_blocks(M)
    return CopulaInference(M, selected, covariance,
                           (; method=selected, diagnostics...), blocks)
end

infer(::CopulaSelection; kwargs...) = throw(ArgumentError(
    "inference after model selection is not automatic; call " *
    "infer(selected_model(selection); ...) only when ignoring selection uncertainty is appropriate"))

"""
    vcov(I::CopulaInference; component=:all)

Return the covariance matrix computed by `infer`. For a fitted `SklarDist`,
`component=:copula` selects the copula block and `component=:margins` selects
all marginal blocks. The default `:all` preserves cross-component covariance.
"""
function StatsBase.vcov(I::CopulaInference; component::Symbol=:all)
    component === :all && return I.covariance
    component === :copula && return I.covariance[I.blocks.copula, I.blocks.copula]
    if component === :margins
        isempty(I.blocks.margins) && throw(ArgumentError(
            "this inference result has no marginal-parameter block"))
        indices = reduce(vcat, collect.(I.blocks.margins))
        return I.covariance[indices, indices]
    end
    throw(ArgumentError("unknown covariance component `$component`; expected :all, :copula, or :margins"))
end
"""Return standard errors derived from a `CopulaInference` covariance matrix."""
StatsBase.stderror(I::CopulaInference) =
    sqrt.(LinearAlgebra.diag(StatsBase.vcov(I)))

"""Return pointwise Wald intervals from a `CopulaInference` result."""
function StatsBase.confint(I::CopulaInference; level::Real=0.95)
    0 < level < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level) / 2)
    parameters = StatsBase.coef(I.model)
    standard_errors = StatsBase.stderror(I)
    return parameters .- z .* standard_errors,
           parameters .+ z .* standard_errors
end

function Base.show(io::IO, I::CopulaInference)
    println(io, "CopulaInference")
    println(io, "  method:     ", I.method)
    println(io, "  model:      ", nameof(typeof(fitteddistribution(I.model))))
    println(io, "  parameters: ", StatsBase.coefnames(I.model))
    print(io, "  covariance: ", size(I.covariance, 1), " x ", size(I.covariance, 2))
end
