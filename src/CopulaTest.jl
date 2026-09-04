"""
    CopulaHypothesis

Abstract supertype for hypotheses about copulas.

Subtypes describe what is being tested. The generic [`CopulaTest`](@ref)
constructor combines a hypothesis, a statistic, and a calibration method to
produce a standard `StatsAPI.HypothesisTest` result.
"""
abstract type CopulaHypothesis end

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

"""
    _available_statistics(h::CopulaHypothesis)

Return the statistic symbols available for `h`. The first entry is the default, mirroring `_available_fitting_methods`.
"""
function _available_statistics(h::CopulaHypothesis)
    throw(ArgumentError("No statistics are implemented for $(nameof(typeof(h)))."))
end

function default_statistic(h::CopulaHypothesis)
    return _find_statistic(h, :default)
end

"""
    _available_calibrations(h::CopulaHypothesis, ::Val{statistic})

Return the calibration symbols available for a hypothesis/statistic pair. The first entry is the default calibration.
"""
function _available_calibrations(h::CopulaHypothesis, ::Val{statistic}) where {statistic}
    throw(ArgumentError("Statistic :$statistic is not implemented for $(nameof(typeof(h)))."))
end

function default_calibration(h::CopulaHypothesis, stat::Val)
    return _find_calibration(h, stat, :default)
end

_symbol_list(symbols) = join((":" * String(symbol) for symbol in symbols), ", ")

function _find_statistic(h::CopulaHypothesis, statistic::Symbol)
    statistics = _available_statistics(h)
    isempty(statistics) && throw(ArgumentError("No statistics are available for $(nameof(typeof(h)))."))
    statistic === :default && return first(statistics)
    statistic in statistics || throw(ArgumentError("Statistic :$statistic is not available for $(nameof(typeof(h))). Available statistics: $(_symbol_list(statistics))."))
    return statistic
end

function _find_calibration(h::CopulaHypothesis, stat::Val{statistic}, calibration::Symbol) where {statistic}
    calibrations = _available_calibrations(h, stat)
    isempty(calibrations) && throw(ArgumentError("No calibrations are available for statistic :$statistic under $(nameof(typeof(h)))."))
    calibration === :default && return first(calibrations)
    calibration in calibrations || throw(ArgumentError("Calibration :$calibration is not available for statistic :$statistic under $(nameof(typeof(h))). Available calibrations: $(_symbol_list(calibrations))."))
    return calibration
end

function _teststatistic(h::CopulaHypothesis, ::Val{statistic}, U::AbstractMatrix; kwargs...) where {statistic}
    throw(ArgumentError("Statistic :$statistic is not implemented for $(nameof(typeof(h)))."))
end

function _calibrate(h::CopulaHypothesis, ::Val{calibration}, ::Val{statistic}, U::AbstractMatrix, observed::Real; kwargs...) where {calibration,statistic}
    throw(ArgumentError("Calibration :$calibration is not implemented for statistic :$statistic under $(nameof(typeof(h)))."))
end

"""
    CopulaTest(hypothesis, U; statistic, calibration, N, pseudo_values, rng)

Run a copula hypothesis test.

`statistic` and `calibration` are public symbols and are converted internally to `Val` dispatch. New hypotheses, statistics, and calibrations extend the framework by adding methods, not by modifying this constructor.
"""
function CopulaTest(h::CopulaHypothesis, U::AbstractMatrix{<:Real}; statistic::Symbol=:default, calibration::Symbol=:default,
                    N::Integer=1000, pseudo_values::Bool=false, rng::Distributions.AbstractRNG=Random.default_rng(), kwargs...)
    V, d, n = _test_pseudos(U, pseudo_values)
    statistic = _find_statistic(h, statistic)
    stat = Val(statistic)
    calibration = _find_calibration(h, stat, calibration)
    observed = _teststatistic(h, stat, V; kwargs...)
    p, n_resamples, details = _calibrate(h, Val(calibration), stat, V, observed; N=Int(N), rng=rng, kwargs...)
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

function _simulation_sample(h::CopulaHypothesis, U::AbstractMatrix, rng::Distributions.AbstractRNG)
    throw(ArgumentError("Simulation under the null is not implemented for $(nameof(typeof(h)))."))
end

function _calibrate(h::CopulaHypothesis, ::Val{:simulation}, stat::Val, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG, kwargs...)
    N = _check_resamples(N)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(_simulation_sample(h, U, rng))
        exceedances += _teststatistic(h, stat, sample; kwargs...) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, (;)
end

function _randomization_sample(h::CopulaHypothesis, U::AbstractMatrix, rng::Distributions.AbstractRNG)
    throw(ArgumentError("Randomization under the null is not implemented for $(nameof(typeof(h)))."))
end

_randomization_details(::CopulaHypothesis) = (;)

function _calibrate(h::CopulaHypothesis, ::Val{:randomization}, stat::Val, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG, kwargs...)
    N = _check_resamples(N)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(_randomization_sample(h, U, rng))
        exceedances += _teststatistic(h, stat, sample; kwargs...) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, _randomization_details(h)
end

function _multiplier_representation(h::CopulaHypothesis, ::Val{statistic}, U::AbstractMatrix) where {statistic}
    throw(ArgumentError("Calibration :multiplier is not implemented for statistic :$statistic under $(nameof(typeof(h)))."))
end

function _calibrate(h::CopulaHypothesis, ::Val{:multiplier}, stat::Val, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG, kwargs...)
    N = _check_resamples(N)
    rep = _multiplier_representation(h, stat, U)
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

function _bootstrap_copula(h::CopulaHypothesis)
    throw(ArgumentError("Parametric bootstrap is not implemented for $(nameof(typeof(h)))."))
end

_bootstrap_hypothesis(h::CopulaHypothesis, ::AbstractMatrix) = h

function _calibrate(h::CopulaHypothesis, ::Val{:parametric_bootstrap}, stat::Val, U::AbstractMatrix, observed::Real; N::Integer, rng::Distributions.AbstractRNG, kwargs...)
    N = _check_resamples(N)
    _, n = size(U)
    exceedances = 0
    for _ in 1:N
        sample = pseudos(rand(rng, _bootstrap_copula(h), n))
        bootstrap_hypothesis = _bootstrap_hypothesis(h, sample)
        exceedances += _teststatistic(bootstrap_hypothesis, stat, sample; kwargs...) >= observed
    end
    return _exceedance_pvalue(exceedances, N), N, (;)
end

################################################################################
##### Independence
################################################################################

"""
    IndependenceHypothesis()

Hypothesis that the components of a copula are mutually independent.
"""
struct IndependenceHypothesis <: CopulaHypothesis end

"""
    IndependenceCopulaTest(U; statistic=:cvm, N=1000, calibration=:simulation, pseudo_values=false, rng=Random.default_rng())

Test mutual independence between the components of a random vector.
"""
const IndependenceCopulaTest = CopulaTest{IndependenceHypothesis}

(::Type{CopulaTest{IndependenceHypothesis}})(U::AbstractMatrix{<:Real}; kwargs...) = CopulaTest(IndependenceHypothesis(), U; kwargs...)

testname(::IndependenceHypothesis) = "Copula independence test"
nullhypothesis(::IndependenceHypothesis) = "The components are mutually independent."
_available_statistics(::IndependenceHypothesis) = (:cvm,)
_available_calibrations(::IndependenceHypothesis, ::Val{:cvm}) = (:simulation,)

function _teststatistic(::IndependenceHypothesis, ::Val{:cvm}, U::AbstractMatrix; kwargs...)
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

"""
    ExchangeabilityHypothesis(; permutations=:G2, weight=:wm2)

Hypothesis that a copula is invariant under coordinate permutations.
"""
struct ExchangeabilityHypothesis{P} <: CopulaHypothesis
    permutations::P
    weight::Symbol
end

ExchangeabilityHypothesis(; permutations=:G2, weight::Symbol=:wm2) = ExchangeabilityHypothesis(permutations, weight)

"""
    ExchangeabilityCopulaTest(U; statistic=:Sn, permutations=:G2, weight=:wm2, N=1000, calibration=:multiplier, pseudo_values=false, rng=Random.default_rng())

Test exchangeability of a copula in arbitrary dimension.
"""
const ExchangeabilityCopulaTest = CopulaTest{<:ExchangeabilityHypothesis}

function (::Type{<:CopulaTest{<:ExchangeabilityHypothesis}})(U::AbstractMatrix{<:Real}; permutations=:G2, weight::Symbol=:wm2, kwargs...)
    return CopulaTest(ExchangeabilityHypothesis(; permutations, weight), U; kwargs...)
end

testname(::ExchangeabilityHypothesis) = "Copula exchangeability test"
nullhypothesis(::ExchangeabilityHypothesis) = "The copula is exchangeable."
_available_statistics(::ExchangeabilityHypothesis) = (:Sn,)
_available_calibrations(::ExchangeabilityHypothesis, ::Val{:Sn}) = (:multiplier,)

function _teststatistic(h::ExchangeabilityHypothesis, ::Val{:Sn}, U::AbstractMatrix; kwargs...)
    return _exchangeability_sn_statistic(U, _exchangeability_permutations(h.permutations, size(U, 1)), h.weight)
end

function _exchangeability_permutations(permutations, d::Integer)
    identity_perm = ntuple(i -> i, d)
    raw = if permutations === :G2
        d == 2 ? ((2, 1),) :
        ((2, 1, ntuple(i -> i + 2, d - 2)...), ntuple(i -> i == d ? 1 : i + 1, d))
    elseif permutations === :G1
        ntuple(i -> Tuple(j == 1 ? i + 1 : j == i + 1 ? 1 : j for j in 1:d), d - 1)
    elseif permutations === :all
        Combinatorics.permutations(1:d)
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
    return Tuple(result)
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
    return s / n
end

function _multiplier_representation(h::ExchangeabilityHypothesis, ::Val{:Sn}, U::AbstractMatrix)
    permutations = _exchangeability_permutations(h.permutations, size(U, 1))
    matrices, weights, bandwidth = _exchangeability_multiplier_matrices(U, permutations, h.weight)
    _, n = size(U)
    return (;matrices, weights, scale=inv(n^2), strict=true, correction=nothing,
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
    RadialSymmetryHypothesis()

Hypothesis that a copula is radially symmetric.
"""
struct RadialSymmetryHypothesis <: CopulaHypothesis end

"""
    RadialSymmetryCopulaTest(U; statistic=:Sn, N=1000, calibration=:randomization, pseudo_values=false, rng=Random.default_rng())

Test radial symmetry of a copula.
"""
const RadialSymmetryCopulaTest = CopulaTest{RadialSymmetryHypothesis}

(::Type{CopulaTest{RadialSymmetryHypothesis}})(U::AbstractMatrix{<:Real}; kwargs...) = CopulaTest(RadialSymmetryHypothesis(), U; kwargs...)

testname(::RadialSymmetryHypothesis) = "Copula radial symmetry test"
nullhypothesis(::RadialSymmetryHypothesis) = "The copula is radially symmetric."
_available_statistics(::RadialSymmetryHypothesis) = (:Sn,)
_available_calibrations(::RadialSymmetryHypothesis, ::Val{:Sn}) = (:randomization,)

function _teststatistic(::RadialSymmetryHypothesis, ::Val{:Sn}, U::AbstractMatrix; kwargs...)
    Cn = EmpiricalCopula(U; pseudo_values=true)
    Cbar = EmpiricalCopula(1 .- U; pseudo_values=true)
    s = 0.0
    @inbounds for u in eachcol(U)
        s += abs2(Distributions.cdf(Cn, u) - Distributions.cdf(Cbar, u))
    end
    return s / size(U, 2)
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

_randomization_details(::RadialSymmetryHypothesis) = (; reflection_probability=0.5,)

################################################################################
##### Extreme Value
################################################################################

"""
    ExtremeValueHypothesis(; powers=3:5)

Hypothesis that a copula belongs to the extreme-value class.
"""
struct ExtremeValueHypothesis{P} <: CopulaHypothesis
    powers::P
end

ExtremeValueHypothesis(; powers=3:5) = ExtremeValueHypothesis(powers)

"""
    ExtremeValueCopulaTest(U; statistic=:Sn, powers=3:5, N=1000, calibration=:multiplier, pseudo_values=false, rng=Random.default_rng())

Test whether a copula belongs to the extreme-value class.
"""
const ExtremeValueCopulaTest = CopulaTest{<:ExtremeValueHypothesis}

function (::Type{<:CopulaTest{<:ExtremeValueHypothesis}})(U::AbstractMatrix{<:Real}; powers=3:5, kwargs...)
    return CopulaTest(ExtremeValueHypothesis(; powers), U; kwargs...)
end

testname(::ExtremeValueHypothesis) = "Extreme-value copula test"
nullhypothesis(::ExtremeValueHypothesis) = "The copula belongs to the extreme-value class."
_available_statistics(::ExtremeValueHypothesis) = (:Sn,)
_available_calibrations(::ExtremeValueHypothesis, ::Val{:Sn}) = (:multiplier,)

function _teststatistic(h::ExtremeValueHypothesis, ::Val{:Sn}, U::AbstractMatrix; kwargs...)
    return _extreme_value_sn_statistic(U, _max_stability_powers(h.powers))
end

function _max_stability_powers(powers)
    raw = powers isa Real ? (powers,) : Tuple(powers)
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

function _multiplier_representation(h::ExtremeValueHypothesis, ::Val{:Sn}, U::AbstractMatrix)
    powers = _max_stability_powers(h.powers)
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
    GoodnessOfFitHypothesis(model)

Hypothesis that data follow a specified copula or fitted copula model. The field `kind` distinguishes `:simple` and `:composite`.
"""
struct GoodnessOfFitHypothesis{M} <: CopulaHypothesis
    model::M
    kind::Symbol
end

GoodnessOfFitHypothesis(C::Copula) = GoodnessOfFitHypothesis(C, :simple)
GoodnessOfFitHypothesis(M::CopulaModel) = GoodnessOfFitHypothesis(M, :composite)

"""
    GOFCopulaTest(C, U; statistic=:Sn, N=1000, calibration=:parametric_bootstrap, pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model, U; statistic=:Sn, N=1000, calibration=:parametric_bootstrap, pseudo_values=false, rng=Random.default_rng())
    GOFCopulaTest(model; kwargs...)

Test goodness of fit for a copula or fitted copula model.
"""
const GOFCopulaTest = CopulaTest{<:GoodnessOfFitHypothesis}

function (::Type{<:CopulaTest{<:GoodnessOfFitHypothesis}})(C::Copula, U::AbstractMatrix{<:Real}; kwargs...)
    return CopulaTest(GoodnessOfFitHypothesis(C), U; kwargs...)
end

function (::Type{<:CopulaTest{<:GoodnessOfFitHypothesis}})(M::CopulaModel, U::AbstractMatrix{<:Real}; kwargs...)
    return CopulaTest(GoodnessOfFitHypothesis(M), U; kwargs...)
end

function (::Type{<:CopulaTest{<:GoodnessOfFitHypothesis}})(M::CopulaModel; kwargs...)
    haskey(M.method_details, :U) || throw(ArgumentError("the fitted model does not store pseudo-observations"))
    return CopulaTest(GoodnessOfFitHypothesis(M), M.method_details.U; pseudo_values=true, kwargs...)
end

testname(::GoodnessOfFitHypothesis) = "Copula goodness-of-fit test"
function nullhypothesis(h::GoodnessOfFitHypothesis)
    h.kind === :simple && return "The data follow the specified copula."
    return "The data belong to the specified copula family."
end

_available_statistics(::GoodnessOfFitHypothesis) = (:Sn,)
_available_calibrations(::GoodnessOfFitHypothesis, ::Val{:Sn}) = (:parametric_bootstrap,)

function _teststatistic(h::GoodnessOfFitHypothesis, ::Val{:Sn}, U::AbstractMatrix; kwargs...)
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

function _bootstrap_hypothesis(h::GoodnessOfFitHypothesis{<:CopulaModel},
        U::AbstractMatrix)
    return GoodnessOfFitHypothesis(_gof_refit(h.model, U))
end

_gof_refit(M::CopulaModel, U::AbstractMatrix) = _gof_refit(_copula_of(M), M, U)

function _gof_refit(C::Copula, M::CopulaModel, U::AbstractMatrix)
    return Distributions.fit(CopulaModel, typeof(C), U; method=M.method, derived_measures=false, vcov=false,)
end

function _gof_refit(C::NestedArchimedeanCopula, M::CopulaModel, U::AbstractMatrix)
    return Distributions.fit(CopulaModel, C, U; method=M.method, derived_measures=false, vcov=false,)
end

function _gof_refit(C::SurvivalCopula, M::CopulaModel, U::AbstractMatrix)
    return Distributions.fit(CopulaModel, typeof(C), U; method=M.method, flips=C.flipmask, derived_measures=false, vcov=false,)
end
