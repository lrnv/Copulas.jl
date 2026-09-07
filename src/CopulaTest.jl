"""
    CopulaHypothesis

Abstract supertype for hypotheses about copulas.

Internal description of a test procedure. This is not a public extension API.
"""
abstract type CopulaHypothesis end

struct IndependenceHypothesis <: CopulaHypothesis end

struct ExchangeabilityHypothesis{P} <: CopulaHypothesis
    permutations::P
    weight::Symbol
end

struct RadialSymmetryHypothesis <: CopulaHypothesis end

struct ExtremeValueHypothesis{P} <: CopulaHypothesis
    powers::P
end

struct GoodnessOfFitHypothesis{M} <: CopulaHypothesis
    model::M
end

"""
    CopulaTest{H<:CopulaHypothesis} <: HypothesisTest

Result of a copula hypothesis test.

The hypothesis stores the mathematical null being tested; `CopulaTest` stores
the common result fields: sample size, dimension, observed statistic, p-value,
resampling method, and details useful for display or reproducibility.
"""
struct CopulaTest{H<:CopulaHypothesis,S<:Real,P<:Real,D<:NamedTuple} <: HypothesisTest
    hypothesis::H
    n::Int
    dimension::Int
    statistic_value::S
    p::P
    n_resamples::Int
    statistic::Symbol
    calibration::Symbol
    details::D
end

"""
    teststatistic(test::CopulaTest)

Return the observed value of the test statistic.
"""
teststatistic(test::CopulaTest) = test.statistic_value

"""
    pvalue(test::CopulaTest)

Return the p-value of `test`.
"""
pvalue(test::CopulaTest) = test.p

StatsBase.nobs(test::CopulaTest) = test.n

"""
    testname(x)

Return the display name for a copula hypothesis or test.

This is an extension hook for new copula tests. It is intentionally not exported to avoid clashes with `HypothesisTests.testname`.
"""
testname(test::CopulaTest) = testname(test.hypothesis)

"""
    nullhypothesis(x)

Return the textual null hypothesis for a copula hypothesis or test. This is an extension hook used by the generic display machinery.
"""
nullhypothesis(test::CopulaTest) = nullhypothesis(test.hypothesis)

# Shared input validation and result assembly; each hypothesis has one procedure.
function _run_copula_test(h::CopulaHypothesis, U::AbstractMatrix{<:Real};
        N::Integer=1000, pseudo_values::Bool=false,
        rng::Distributions.AbstractRNG=Random.default_rng())
    N = _check_resamples(N)
    V, d, n = _test_pseudos(U, pseudo_values)
    observed = _teststatistic(h, V)
    p, n_resamples, details = _calibrate(h, V, observed; N, rng)
    statistic, calibration = _test_method(h)
    return CopulaTest(h, n, d, observed, p, n_resamples, statistic, calibration, details)
end

function _test_pseudos(U::AbstractMatrix{<:Real}, pseudo_values::Bool)
    all(isfinite, U) || throw(ArgumentError("input data must be finite"))
    d, n = size(U)
    d >= 2 || throw(ArgumentError("at least two components are required"))
    n >= 2 || throw(ArgumentError("at least two observations are required"))

    for j in 1:d
        allunique(@view U[j, :]) || throw(ArgumentError(
            "copula hypothesis tests currently require continuous, tie-free margins; " *
            "ties were detected in margin $j. Tie-aware procedures are not yet implemented."))
    end

    V = pseudo_values ? Matrix{Float64}(U) : pseudos(U)
    all(x -> 0 <= x <= 1, V) || throw(ArgumentError("pseudo-observations must lie in [0, 1]"))
    return V, d, n
end

function _empirical_copula_partial(Cn::EmpiricalCopula, u::AbstractVector, l::Integer, h::Real)
    lo = Vector{Float64}(u)
    hi = Vector{Float64}(u)
    lo[l] = max(lo[l] - h, 0.0)
    hi[l] = min(hi[l] + h, 1.0)
    width = hi[l] - lo[l]
    return width > 0 ? (Distributions.cdf(Cn, hi) - Distributions.cdf(Cn, lo)) / width : 0.0
end

function _exceedance_pvalue(exceedances::Integer, N::Integer; correction=0.5)
    correction === nothing && return exceedances / N
    return (correction + exceedances) / (N + 1)
end

function _check_resamples(N::Integer)
    N >= 1 || throw(ArgumentError("`N` must be positive."))
    return Int(N)
end

function _calibrate(h::IndependenceHypothesis, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG)
    N = _check_resamples(N)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(_simulation_sample(h, U, rng))
        exceedances += _teststatistic(h, sample) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, (;)
end

_randomization_details(::CopulaHypothesis) = (;)
_randomization_pseudos(::CopulaHypothesis, sample::AbstractMatrix) = pseudos(sample)

function _calibrate(h::RadialSymmetryHypothesis, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG)
    N = _check_resamples(N)
    exceedances = 0
    for _ in 1:N
        sample = _randomization_pseudos(h, _randomization_sample(h, U, rng),)
        exceedances += _teststatistic(h, sample) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, _randomization_details(h)
end

function _calibrate(h::Union{ExchangeabilityHypothesis,ExtremeValueHypothesis}, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG)
    N = _check_resamples(N)
    rep = _multiplier_representation(h, U)
    p = _multiplier_pvalue(rep.matrices, observed, N, rng;
        weights=get(rep, :weights, nothing),
        scale=rep.scale,
        strict=get(rep, :strict, false),
        correction=get(rep, :correction, 0.5))
    return p, N, get(rep, :details, (;))
end

function _multiplier_pvalue(matrices, observed::Real, N::Integer, rng::Distributions.AbstractRNG; weights=nothing, scale::Real, strict::Bool=false, correction=0.5)
    n = size(first(matrices), 2)
    xi = Vector{Float64}(undef, n)
    work = Vector{Float64}(undef, n)
    inv_sqrt_n = inv(sqrt(n))
    exceedances = 0

    for _ in 1:N
        Random.randexp!(rng, xi)
        xi .-= Statistics.mean(xi)
        bootstrap_stat = 0.0

        if weights === nothing
            for Q in matrices
                LinearAlgebra.mul!(work, Q, xi)
                @inbounds for i in 1:n
                    bootstrap_stat += abs2(inv_sqrt_n * work[i])
                end
            end
        else
            for (Q, w) in zip(matrices, weights)
                LinearAlgebra.mul!(work, Q, xi)
                @inbounds for i in 1:n
                    bootstrap_stat += abs2(inv_sqrt_n * work[i]) * w[i]
                end
            end
        end

        value = scale * bootstrap_stat
        exceedances += strict ? value > observed : value >= observed
    end

    return _exceedance_pvalue(exceedances, N; correction)
end

_bootstrap_hypothesis(h::CopulaHypothesis, ::AbstractMatrix) = h

function _calibrate(h::GoodnessOfFitHypothesis, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG)
    N = _check_resamples(N)
    _, n = size(U)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(rand(rng, _bootstrap_copula(h), n))
        bootstrap_hypothesis = _bootstrap_hypothesis(h, sample)
        exceedances += _teststatistic(bootstrap_hypothesis, sample) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, (;)
end

################################################################################
##### Independence
################################################################################


"""
    IndependenceCopulaTest(U; N=1000, pseudo_values=false, rng=Random.default_rng())

Test mutual independence between the components of a random vector.
"""
IndependenceCopulaTest(U::AbstractMatrix{<:Real}; kwargs...) = _run_copula_test(IndependenceHypothesis(), U; kwargs...)

testname(::IndependenceHypothesis) = "Copula independence test"
nullhypothesis(::IndependenceHypothesis) = "The components are mutually independent."
_test_method(::IndependenceHypothesis) = (:cvm, :simulation)

function _teststatistic(::IndependenceHypothesis, U::AbstractMatrix)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - prod(u))
    end
    return s
end

function _simulation_sample(::IndependenceHypothesis, U::AbstractMatrix, rng::Distributions.AbstractRNG)
    sample = similar(U)
    Random.rand!(rng, sample)
    return sample
end

################################################################################
##### Exchangeability
################################################################################


ExchangeabilityHypothesis(; permutations=:G2, weight::Symbol=:wm2) = ExchangeabilityHypothesis(permutations, weight)

"""
    ExchangeabilityCopulaTest(U; permutations=:G2, weight=:wm2, N=1000, pseudo_values=false, rng=Random.default_rng())

Test exchangeability of a copula in arbitrary dimension.
"""
function ExchangeabilityCopulaTest(U::AbstractMatrix{<:Real}; permutations=:G2, weight::Symbol=:wm2, kwargs...)
    d, n = size(U)
    d >= 2 || throw(ArgumentError("at least two components are required"))
    weight in (:none, :wm2) || throw(ArgumentError("expected weight=:none or :wm2"))
    selected = _exchangeability_permutations(permutations, d)
    _check_multiplier_matrix_cost(length(selected), n)
    return _run_copula_test(ExchangeabilityHypothesis(selected, weight), U; kwargs...)
end

testname(::ExchangeabilityHypothesis) = "Copula exchangeability test"
nullhypothesis(::ExchangeabilityHypothesis) = "The copula is exchangeable."
_test_method(::ExchangeabilityHypothesis) = (:Sn, :multiplier)

function _teststatistic(h::ExchangeabilityHypothesis, U::AbstractMatrix)
    return _exchangeability_sn_statistic(U, h.permutations, h.weight)
end

const _MAX_MULTIPLIER_MATRIX_BYTES = 512 * 1024^2

function _check_multiplier_matrix_cost(nperms::Integer, n::Integer;
        label::AbstractString="the requested collection")
    matrix_bytes = big(nperms) * big(n)^2 * sizeof(Float64)

    matrix_bytes <= _MAX_MULTIPLIER_MATRIX_BYTES && return nothing

    estimated_mib = Float64(matrix_bytes) / 1024^2
    limit_mib = _MAX_MULTIPLIER_MATRIX_BYTES / 1024^2

    throw(ArgumentError(
        "$(label) would materialize $(nperms) dense $(n)×$(n) " *
        "multiplier matrices (approximately $(round(estimated_mib; digits=1)) MiB), " *
        "exceeding the current $(round(limit_mib; digits=0)) MiB safety limit. " *
        "Use a smaller sample or fewer permutations/powers."
    ))
end

function _exchangeability_permutations(permutations, d::Integer)
    identity_perm = ntuple(i -> i, d)
    raw = if permutations === :G2
        d == 2 ? ((2, 1),) :
        ((2, 1, ntuple(i -> i + 2, d - 2)...), ntuple(i -> i == d ? 1 : i + 1, d))
    elseif permutations === :G1
        ntuple(i -> Tuple(j == 1 ? i + 1 : j == i + 1 ? 1 : j for j in 1:d), d - 1)
    elseif permutations isa Symbol
        throw(ArgumentError("expected permutations=:G1, :G2, or an explicit collection; :all is unsupported"))
    else
        is_single = (permutations isa Tuple || permutations isa AbstractVector) && length(permutations) == d && all(x -> x isa Integer, permutations)
        is_single ? (permutations,) : permutations
    end

    result = NTuple{d,Int}[]
    for perm in raw
        p = Tuple(Int.(perm))
        length(p) == d || throw(ArgumentError("permutations must have length $d"))
        sort(collect(p)) == collect(1:d) || throw(ArgumentError("invalid permutation `$perm`"))
        p == identity_perm || push!(result, p)
    end
    isempty(result) && throw(ArgumentError("at least one non-identity permutation is required"))
    return unique!(result)
end

function _exchangeability_weight(u::AbstractVector, perm::Tuple, weight::Symbol)
    weight === :none && return 1.0
    weight === :wm2 || throw(ArgumentError("Only `weight=:wm2` and `weight=:none` are implemented."))

    m = minimum(u)
    omega = if count(i -> perm[i] != i, eachindex(perm)) == 2 && all(perm[perm[i]] == i for i in eachindex(perm))
        i = findfirst(k -> perm[k] != k, eachindex(perm))
        j = perm[i]
        abs(u[i] - u[j])
    else
        v = sort(collect(u))
        sum(v[i] - m for i in cld(length(v), 2) + 1:length(v))
    end
    wm = min(m, omega, length(u) - 1 + m - sum(u))
    return abs2(max(wm, 0.0))
end

function _exchangeability_sn_statistic(U::AbstractMatrix, permutations, weight::Symbol)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    uperm = Vector{Float64}(undef, d)

    @inbounds for perm in permutations
        for i in 1:n
            u = @view U[:, i]
            for k in 1:d
                uperm[k] = u[perm[k]]
            end
            diff = Distributions.cdf(Cn, u) - Distributions.cdf(Cn, uperm)
            s += abs2(diff) * _exchangeability_weight(u, perm, weight)
        end
    end
    return s
end

function _multiplier_representation(h::ExchangeabilityHypothesis, U::AbstractMatrix)
    _, n = size(U)
    permutations = h.permutations
    matrices, weights, bandwidth = _exchangeability_multiplier_matrices(U, permutations, h.weight)
    return (;matrices, weights, scale=inv(n), strict=true, correction=nothing,
            details=(; permutations=h.permutations, generator=permutations, weight=h.weight, multiplier=:exponential, derivative_bandwidth=bandwidth),)
end

function _exchangeability_multiplier_matrices(U::AbstractMatrix, permutations, weight::Symbol)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    h = inv(sqrt(n))
    partials = Matrix{Float64}(undef, d, n)
    q_matrices = Matrix{Float64}[]
    weights = Vector{Float64}[]

    @inbounds for i in 1:n
        u = @view U[:, i]
        for l in 1:d
            partials[l, i] = _empirical_copula_partial(Cn, u, l, h)
        end
    end

    @inbounds for perm in permutations
        invperm = Vector{Int}(undef, d)
        for k in 1:d
            invperm[perm[k]] = k
        end

        Q = Matrix{Float64}(undef, n, n)
        w = Vector{Float64}(undef, n)
        for i in 1:n
            u = @view U[:, i]
            w[i] = _exchangeability_weight(u, perm, weight)
            for j in 1:n
                le_u = true
                le_up = true
                for k in 1:d
                    U[k, j] <= u[k] || (le_u = false)
                    U[k, j] <= u[perm[k]] || (le_up = false)
                end

                q = (le_u ? 1.0 : 0.0) - (le_up ? 1.0 : 0.0)
                for l in 1:d
                    le_margin = U[l, j] <= u[l]
                    le_permuted_margin = U[invperm[l], j] <= u[l]
                    q -= partials[l, i] *
                        ((le_margin ? 1.0 : 0.0) - (le_permuted_margin ? 1.0 : 0.0))
                end
                Q[i, j] = q
            end
        end
        push!(q_matrices, Q)
        push!(weights, w)
    end
    return q_matrices, weights, h
end

################################################################################
##### Radial Symmetry
################################################################################


"""
    RadialSymmetryCopulaTest(U; N=1000, pseudo_values=false, rng=Random.default_rng())

Test radial symmetry of a copula.
"""
RadialSymmetryCopulaTest(U::AbstractMatrix{<:Real}; kwargs...) = _run_copula_test(RadialSymmetryHypothesis(), U; kwargs...)

testname(::RadialSymmetryHypothesis) = "Copula radial symmetry test"
nullhypothesis(::RadialSymmetryHypothesis) = "The copula is radially symmetric."
_test_method(::RadialSymmetryHypothesis) = (:Sn, :randomization)

function _teststatistic(::RadialSymmetryHypothesis, U::AbstractMatrix)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    Cbar = EmpiricalCopula(1 .- U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - Distributions.cdf(Cbar, u))
    end
    return s
end

function _randomization_sample(::RadialSymmetryHypothesis, U::AbstractMatrix, rng::Distributions.AbstractRNG)
    d, n = size(U)
    sample = similar(U)
    @inbounds for i in 1:n
        reflected = rand(rng) < 0.5
        for j in 1:d
            sample[j, i] = reflected ? 1 - U[j, i] : U[j, i]
        end
    end
    return sample
end

function _average_pseudos(sample::AbstractMatrix)
    d, n = size(sample)
    U = Matrix{Float64}(undef, d, n)
    denom = n + 1

    @inbounds for j in 1:d
        U[j, :] .= StatsBase.tiedrank(@view sample[j, :]) ./ denom
    end

    return U
end

_randomization_pseudos(::RadialSymmetryHypothesis, sample::AbstractMatrix,) = _average_pseudos(sample)

_randomization_details(::RadialSymmetryHypothesis) = (; reflection_probability=0.5,)

################################################################################
##### Extreme Value
################################################################################


ExtremeValueHypothesis(; powers=3:5) = ExtremeValueHypothesis(powers)

"""
    ExtremeValueCopulaTest(U; powers=3:5, N=1000, pseudo_values=false, rng=Random.default_rng())

Test whether a copula belongs to the extreme-value class.
"""
function ExtremeValueCopulaTest(U::AbstractMatrix{<:Real}; powers=3:5, kwargs...)
    selected = _max_stability_powers(powers)
    _check_multiplier_matrix_cost(length(selected), size(U, 2))
    return _run_copula_test(ExtremeValueHypothesis(selected), U; kwargs...)
end

testname(::ExtremeValueHypothesis) = "Extreme-value copula test"
nullhypothesis(::ExtremeValueHypothesis) = "The copula belongs to the extreme-value class."
_test_method(::ExtremeValueHypothesis) = (:Sn, :multiplier)

function _teststatistic(h::ExtremeValueHypothesis, U::AbstractMatrix)
    return _extreme_value_sn_statistic(U, h.powers)
end

function _max_stability_powers(powers)
    raw = powers isa Real ? (powers,) : powers
    result = Float64[]
    for r in raw
        isfinite(r) && r > 1 ||
            throw(ArgumentError("max-stability powers must be finite and greater than one"))
        push!(result, Float64(r))
    end
    isempty(result) && throw(ArgumentError("at least one max-stability power is required"))
    return Tuple(result)
end

function _extreme_value_sn_statistic(U::AbstractMatrix, powers)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    uroot = Vector{Float64}(undef, d)
    s = 0.0

    @inbounds for r in powers
        invr = inv(r)
        for u in eachcol(U)
            for k in 1:d
                uroot[k] = u[k]^invr
            end
            diff = Distributions.cdf(Cn, uroot)^r - Distributions.cdf(Cn, u)
            s += abs2(diff)
        end
    end
    return s
end

function _multiplier_representation(h::ExtremeValueHypothesis, U::AbstractMatrix)
    powers = h.powers
    matrices, bandwidth = _extreme_value_multiplier_matrices(U, powers)
    _, n = size(U)
    return (;matrices, scale=inv(n), strict=false, correction=0.5,
            details=(; powers, multiplier=:exponential, derivative_bandwidth=bandwidth),)
end

function _extreme_value_multiplier_matrices(U::AbstractMatrix, powers)
    d, n = size(U)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    h = inv(sqrt(n))
    uroot = Vector{Float64}(undef, d)
    partials_u = Vector{Float64}(undef, d)
    partials_root = Vector{Float64}(undef, d)
    matrices = Matrix{Float64}[]

    @inbounds for r in powers
        Q = Matrix{Float64}(undef, n, n)
        invr = inv(r)
        for i in 1:n
            u = @view U[:, i]
            for k in 1:d
                uroot[k] = u[k]^invr
                partials_u[k] = _empirical_copula_partial(Cn, u, k, h)
            end
            croot = Distributions.cdf(Cn, uroot)
            factor = r * croot^(r - 1)
            for k in 1:d
                partials_root[k] = _empirical_copula_partial(Cn, uroot, k, h)
            end

            for j in 1:n
                le_u = true
                le_root = true
                for k in 1:d
                    U[k, j] <= u[k] || (le_u = false)
                    U[k, j] <= uroot[k] || (le_root = false)
                end

                q_u = le_u ? 1.0 : 0.0
                q_root = le_root ? 1.0 : 0.0
                for k in 1:d
                    q_u -= partials_u[k] * (U[k, j] <= u[k] ? 1.0 : 0.0)
                    q_root -= partials_root[k] * (U[k, j] <= uroot[k] ? 1.0 : 0.0)
                end
                Q[i, j] = factor * q_root - q_u
            end
        end
        push!(matrices, Q)
    end
    return matrices, h
end

################################################################################
##### Goodness of Fit
################################################################################




"""
    GOFCopulaTest(C, U; N=1000, pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model, U; N=1000, pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model)

Test goodness of fit for a copula or fitted copula model.

`GOFCopulaTest(C, U)` treats `C` as a fixed specified copula and tests a simple
null hypothesis.

`GOFCopulaTest(M)` tests the fitting sample stored by `M` under a composite null
hypothesis. The estimator specification that produced `M` is replayed in every
parametric-bootstrap replicate.

`GOFCopulaTest(M, U)` first refits that estimator specification on `U`; the
resulting fitted model is used for the observed statistic, and the same fitting
procedure is repeated in every bootstrap replicate. If the fitting procedure is
not reproducibly specified, composite GOF throws an `ArgumentError`.
"""
function GOFCopulaTest(C::Copula, U::AbstractMatrix{<:Real}; kwargs...)
    return _run_copula_test(GoodnessOfFitHypothesis(C), U; kwargs...)
end

function GOFCopulaTest(M::CopulaModel, U::AbstractMatrix{<:Real};
        N::Integer=1000, pseudo_values::Bool=false,
        rng::Distributions.AbstractRNG=Random.default_rng())
    # For a composite null, M describes the estimator specification.
    # The observed statistic must use parameters estimated from the sample being
    # tested, just as every bootstrap replicate is refitted.
    N = _check_resamples(N)
    V, _, _ = _test_pseudos(U, pseudo_values)
    Mrefit = _refit(M, V)
    return _run_copula_test(GoodnessOfFitHypothesis(Mrefit), V; pseudo_values=true, N, rng)
end

function GOFCopulaTest(M::CopulaModel; kwargs...)
    haskey(M.method_details, :U) || throw(ArgumentError("the fitted model does not store its fitting sample"))

    # Most copula fits receive pseudo-observations directly. Empirical fitting
    # routines may instead have received raw data with `pseudo_values=false`;
    # respect that metadata rather than silently treating the stored input as
    # already ranked.
    stored_pseudo_values = get(M.method_details, :pseudo_values, true)

    return _run_copula_test(GoodnessOfFitHypothesis(M), M.method_details.U; pseudo_values=stored_pseudo_values, kwargs...,)
end

testname(::GoodnessOfFitHypothesis) = "Copula goodness-of-fit test"
nullhypothesis(::GoodnessOfFitHypothesis{<:Copula}) = "The data follow the specified copula."
nullhypothesis(::GoodnessOfFitHypothesis{<:CopulaModel}) = "The data belong to the specified copula family."

_test_method(::GoodnessOfFitHypothesis) = (:Sn, :parametric_bootstrap)

function _teststatistic(h::GoodnessOfFitHypothesis, U::AbstractMatrix)
    C = _gof_copula(h)
    length(C) == size(U, 1) || throw(DimensionMismatch("model dimension does not match input data"))
    return _gof_sn_statistic(U, C)
end

_gof_copula(h::GoodnessOfFitHypothesis) = h.model isa CopulaModel ? _copula_of(h.model) : h.model

function _gof_sn_statistic(U::AbstractMatrix, C::Copula)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - Distributions.cdf(C, u))
    end
    return s
end

_bootstrap_copula(h::GoodnessOfFitHypothesis) = _gof_copula(h)

function _bootstrap_hypothesis(h::GoodnessOfFitHypothesis{<:CopulaModel}, U::AbstractMatrix)
    return GoodnessOfFitHypothesis(_refit(h.model, U))
end
