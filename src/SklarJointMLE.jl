
# Joint parameter geometry and maximum-likelihood optimization for Sklar
# distributions.

# Marginal geometry ----------------------------------------------------------

# Internal marginal parameter geometry for joint Sklar MLE.
#
# Each supported distribution family is represented by a transformation
# from unconstrained coordinates to its natural parameters.

"""
    _sklar_margin_schema(MarginType)

Return the unconstrained parameter transformation for a supported
univariate margin. Parameter order follows `Distributions.params`.
"""
function _sklar_margin_schema(::Type{D},) where {D<:Distributions.UnivariateDistribution}
    throw(ArgumentError("joint Sklar MLE has no parameter geometry for margin family $D",))
end

_sklar_margin_schema(::Type{<:Union{Distributions.Normal,Distributions.LogNormal}}) = as((asℝ, asℝ₊))
_sklar_margin_schema(::Type{<:Distributions.Exponential}) = as((asℝ₊,))
_sklar_margin_schema(::Type{<:Union{Distributions.Gamma,Distributions.Beta,Distributions.Weibull}}) = as((asℝ₊, asℝ₊))

_sklar_margin_has_schema(::Type{<:Distributions.UnivariateDistribution}) = false
_sklar_margin_has_schema(::Type{<:Union{Distributions.Normal,Distributions.LogNormal}}) = true
_sklar_margin_has_schema(::Type{<:Distributions.Exponential}) = true
_sklar_margin_has_schema(::Type{<:Union{Distributions.Gamma,Distributions.Beta,Distributions.Weibull}}) = true

"""
    _sklar_margin_coordinates(margin)

Map natural marginal parameters to unconstrained coordinates.
"""
function _sklar_margin_coordinates(margin::Distributions.UnivariateDistribution)
    Paramorph.has_parameter_geometry(typeof(margin)) && return Paramorph.unconstrain(margin)
    schema = _sklar_margin_schema(typeof(margin))
    return Paramorph.TransformVariables.inverse(schema, Distributions.params(margin))
end

"""
    _sklar_margin_from_coordinates(MarginType, coordinates)

Construct a margin from unconstrained coordinates, preserving
automatic-differentiation numeric types.
"""
function _sklar_margin_from_coordinates(::Type{D}, coordinates::AbstractVector{<:Real},) where {D<:Distributions.UnivariateDistribution}
    schema = _sklar_margin_schema(D)
    parameters = Paramorph.TransformVariables.transform(schema, coordinates,)
    return D(parameters...)
end

function _sklar_margin_from_coordinates(margin::Distributions.UnivariateDistribution, z::AbstractVector{<:Real})
    Paramorph.has_parameter_geometry(typeof(margin)) && return Paramorph.constraint(margin, z)
    family = Base.typename(typeof(margin)).wrapper
    return _sklar_margin_from_coordinates(family, z)
end

# Experimental constructor oracle -------------------------------------------

# This mechanism is intentionally separate from the Paramorph geometry above.
# Its scope is limited to distributions whose
# `params` are flat tuples of scalar real values. Within this experiment,
# AbstractFloat parameters are treated as free and Integer parameters as fixed;
# those are prototype conventions, not a general declaration of estimability
# or structural status. Parameterless distributions are retained as fixed
# components. Families with parameters but no floating-point values are
# rejected, because their estimability cannot be inferred; other parameter
# layouts require an explicit geometry.

"""
    _sklar_constructor_oracle_coordinates(margin)

Extract experimental constructor-oracle coordinates from a marginal.
"""
function _sklar_constructor_oracle_coordinates(margin::Distributions.UnivariateDistribution)
    parameters = Distributions.params(margin)
    parameters isa Tuple || throw(ArgumentError("$(typeof(margin)): params must be a tuple"))
    isempty(parameters) && return Float64[]
    all(x -> x isa AbstractFloat || x isa Integer, parameters) ||
        throw(ArgumentError("$(typeof(margin)): nested, array, and nonnumeric parameters require a declared geometry"))
    all(isfinite, parameters) ||
        throw(ArgumentError("$(typeof(margin)): non-finite starting parameters"))
    any(x -> x isa AbstractFloat, parameters) ||
        throw(ArgumentError("$(typeof(margin)): constructor oracle found no floating-point parameters; declare an explicit geometry"))

    family = Base.typename(typeof(margin)).wrapper
    Distributions.params(family(parameters...)) == parameters ||
        throw(ArgumentError("$(typeof(margin)): constructor does not round-trip through params"))

    return Float64[x for x in parameters if x isa AbstractFloat]
end

"""
    _sklar_constructor_oracle_from_coordinates(margin, coordinates)

Reconstruct a marginal with the experimental constructor oracle. Integer
parameters remain those of `margin`; floating-point parameters are replaced
in parameter order and validated by the original constructor.
"""
function _sklar_constructor_oracle_from_coordinates(margin::Distributions.UnivariateDistribution, coordinates::AbstractVector{<:Real},)
    parameters = Distributions.params(margin)
    nfree = count(x -> x isa AbstractFloat, parameters)
    length(coordinates) == nfree ||
        throw(DimensionMismatch("expected $nfree marginal coordinates"))
    nfree == 0 && return margin

    k = 0
    reconstructed_parameters = ntuple(length(parameters)) do i
        if parameters[i] isa AbstractFloat
            k += 1
            return coordinates[k]
        end
        return parameters[i]
    end

    family = Base.typename(typeof(margin)).wrapper
    return family(reconstructed_parameters...)
end

# Joint geometry -------------------------------------------------------------

# Joint unconstrained coordinates for Sklar distributions.
#
# The copula uses its existing Paramorph geometry.
# Margins use the explicitly supported Sklar marginal schemas.

struct _SklarJointChart{S,R,M}
    prototype::S
    ranges::R
    margin_modes::M
end

_sklar_copula_dimension(C::Copula) = _parameter_dimension(C)
_sklar_copula_dimension(::IndependentCopula) = 0
_sklar_copula_coordinates(C::Copula) = _parameter_coordinates(C)
_sklar_copula_coordinates(::IndependentCopula) = Float64[]
_sklar_copula_reconstruct(C::Copula, z) = _from_parameter_coordinates(C, z)

function _sklar_copula_reconstruct(C::IndependentCopula, z)
    isempty(z) || throw(DimensionMismatch("IndependentCopula has no free parameters",))
    return C
end

# Recover the family constructor without fixing its original numeric type.
# This permits reconstruction with ForwardDiff.Dual parameters.
_sklar_margin_family(m::D) where {D} = Base.typename(D).wrapper

function _sklar_margin_mode(margin::Distributions.UnivariateDistribution)
    geometry = typeof(margin)
    Paramorph.has_parameter_geometry(geometry) && return :paramorph
    _sklar_margin_has_schema(geometry) && return :paramorph
    Distributions.params(margin) === () && return :fixed
    return :constructor_oracle
end

function _sklar_margin_coordinates(margin::Distributions.UnivariateDistribution, mode::Symbol)
    mode === :paramorph && return _sklar_margin_coordinates(margin)
    mode === :constructor_oracle && return _sklar_constructor_oracle_coordinates(margin)
    mode === :fixed && return Float64[]
    throw(ArgumentError("unknown Sklar marginal parameterization $mode"))
end

function _sklar_margin_from_coordinates(margin::Distributions.UnivariateDistribution, coordinates::AbstractVector{<:Real}, mode::Symbol,)
    mode === :paramorph && return _sklar_margin_from_coordinates(margin, coordinates)
    mode === :constructor_oracle && return _sklar_constructor_oracle_from_coordinates(margin, coordinates)
    mode === :fixed && return _sklar_constructor_oracle_from_coordinates(margin, coordinates)
    throw(ArgumentError("unknown Sklar marginal parameterization $mode"))
end

"""
    _sklar_joint_chart(S::SklarDist)

Build an internal parameter chart for a Sklar distribution.

The first coordinate block belongs to the copula, followed by one
block per marginal distribution. Empty blocks are permitted.
"""
function _sklar_joint_chart(S::SklarDist)
    margin_modes = map(_sklar_margin_mode, S.m)
    widths = (_sklar_copula_dimension(S.C), map((m, mode) -> length(_sklar_margin_coordinates(m, mode)), S.m, margin_modes)...,)
    endpoints = cumsum(collect(widths))
    ranges = ntuple(length(widths)) do i
        (endpoints[i] - widths[i] + 1):endpoints[i]
    end

    return _SklarJointChart(S, ranges, margin_modes)
end

"""
    _sklar_joint_coordinates(chart::_SklarJointChart)

Return the concatenated unconstrained copula and marginal coordinates.
"""
function _sklar_joint_coordinates(chart::_SklarJointChart)
    S = chart.prototype
    copula_z = _sklar_copula_coordinates(S.C)
    margin_z = map(_sklar_margin_coordinates, S.m, chart.margin_modes)

    return vcat(copula_z, margin_z...)
end

"""
    _sklar_joint_distribution(chart::_SklarJointChart, z)

Reconstruct a Sklar distribution from a joint coordinate vector.

The reconstruction preserves automatic-differentiation numeric types.
"""
function _sklar_joint_distribution(chart::_SklarJointChart, z::AbstractVector{<:Real},)
    expected = last(chart.ranges[end])
    length(z) == expected || throw(DimensionMismatch("joint Sklar chart expects $expected coordinates; " * "received $(length(z))",))
    S = chart.prototype
    C = _sklar_copula_reconstruct(S.C, view(z, chart.ranges[1]),)

    margins = ntuple(length(S.m)) do i
        _sklar_margin_from_coordinates(S.m[i], view(z, chart.ranges[i + 1]), chart.margin_modes[i])
    end

    return SklarDist(C, margins)
end

# Optimization ---------------------------------------------------------------

_sklar_joint_optimizer_kind(chart::_SklarJointChart) = any(==( :constructor_oracle), chart.margin_modes) ? :nelder_mead : :lbfgs

# Joint maximum-likelihood optimization for Sklar distributions.

function _sklar_joint_central_gradient(objective, z::AbstractVector{<:Real}; step=1e-4)
    gradient = similar(z, Float64)

    for i in eachindex(z)
        h = step * max(1.0, abs(z[i]))
        zplus = copy(z)
        zminus = copy(z)
        zplus[i] += h
        zminus[i] -= h
        gradient[i] = (objective(zplus) - objective(zminus)) / (2h)
    end

    return gradient
end

function _sklar_joint_dual_method_error(err)
    err isa MethodError || return false
    err.f isa Function && parentmodule(err.f) === (@__MODULE__) && return false
    return any(arg -> arg isa ForwardDiff.Dual || (arg isa Type && arg <: ForwardDiff.Dual), err.args)
end

function _sklar_joint_primal_coordinates(z::AbstractVector)
    return Float64[x isa ForwardDiff.Dual ? ForwardDiff.value(x) : x for x in z]
end

function _sklar_joint_primal_objective(chart::_SklarJointChart, X, weights, z)
    primal = _sklar_joint_primal_coordinates(z)
    S = _sklar_joint_distribution(chart, primal)
    return -_weighted_loglikelihood(S, X, weights)
end

function _sklar_joint_dual_numeric_error(err, objective, chart, X, weights, z)
    err isa LinearAlgebra.PosDefException || return false
    primal_value = try
        _sklar_joint_primal_objective(chart, X, weights, z)
    catch primal_err
        primal_err isa LinearAlgebra.PosDefException || rethrow()
        return false
    end
    return isfinite(primal_value)
end

function _sklar_joint_lbfgs_autodiff(objective, z0::AbstractVector{<:Real}; chart=nothing, X=nothing, weights=nothing)
    finite_autodiff = ADTypes.AutoFiniteDiff(; fdtype=Val(:central), relstep=1e-4, absstep=1e-4)
    gradient = try
        ForwardDiff.gradient(objective, z0)
    catch err
        if !isnothing(chart) && _sklar_joint_dual_numeric_error(err, objective, chart, X, weights, z0)
            return finite_autodiff
        end
        _sklar_joint_dual_method_error(err) || rethrow()
        return finite_autodiff
    end

    all(isfinite, gradient) || return finite_autodiff

    numerical_gradient = _sklar_joint_central_gradient(objective, z0)
    all(isfinite, numerical_gradient) || return finite_autodiff

    difference = sqrt(sum(abs2, gradient - numerical_gradient))
    scale = max(1.0, sqrt(sum(abs2, gradient)), sqrt(sum(abs2, numerical_gradient)))

    return difference <= 1e-4 * scale ?
           ADTypes.AutoForwardDiff() :
           finite_autodiff
end

function _sklar_joint_objective(chart::_SklarJointChart, X::AbstractMatrix, weights, z)
    S = try
        _sklar_joint_distribution(chart, z)
    catch err
        err isa DomainError && return Inf
        rethrow()
    end

    value = try
        -_weighted_loglikelihood(S, X, weights)
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
        primal_value = try
            _sklar_joint_primal_objective(chart, X, weights, z)
        catch primal_err
            primal_err isa LinearAlgebra.PosDefException || rethrow()
            return Inf
        end
        isfinite(primal_value) && rethrow()
        return Inf
    end
    return isfinite(value) ? value : Inf
end

"""
    _fit_sklar_joint_mle(S0, X; weights=nothing)

Maximize the complete Sklar log-likelihood, starting from S0.

Uses validated ForwardDiff when available and central finite differences
otherwise.
"""
function _fit_sklar_joint_mle(S0::SklarDist, X::AbstractMatrix; weights=nothing)
    chart = _sklar_joint_chart(S0)
    z0 = _sklar_joint_coordinates(chart)
    Xfit, wfit = _weighted_sample(X, weights)

    last_valid = Ref(copy(z0))
    last_dual_primal = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    objective(z) = begin
        if !(eltype(z) <: AbstractFloat)
            last_dual_primal[] = _sklar_joint_primal_coordinates(z)
        end
        value = _sklar_joint_objective(chart, Xfit, wfit, z)
        if eltype(z) <: AbstractFloat && isfinite(value)
            last_valid[] = copy(z)
        end
        value
    end

    initial = objective(z0)
    isfinite(initial) || throw(ArgumentError("JMLE requires a finite initial log-likelihood"))
    isempty(z0) && return S0

    optimizer_kind = _sklar_joint_optimizer_kind(chart)

    result = if optimizer_kind === :nelder_mead
        Optim.optimize(objective, z0, Optim.NelderMead(), Optim.Options(; iterations=2500))
    else
        autodiff = _sklar_joint_lbfgs_autodiff(objective, z0; chart, X=Xfit, weights=wfit)
        result = try
            Optim.optimize(objective, z0, Optim.LBFGS(); autodiff)
        catch err
            err isa LinearAlgebra.PosDefException || rethrow()
            isnothing(last_dual_primal[]) && rethrow()
            primal_value = try
                _sklar_joint_primal_objective(chart, Xfit, wfit, last_dual_primal[])
            catch primal_err
                primal_err isa LinearAlgebra.PosDefException || rethrow()
                Inf
            end
            isfinite(primal_value) || rethrow()
            nothing
        end
        needs_recovery = isnothing(result) ||
            (!Optim.converged(result) && !isnothing(last_dual_primal[]) &&
             !isfinite(try
                 _sklar_joint_primal_objective(chart, Xfit, wfit, last_dual_primal[])
             catch primal_err
                 primal_err isa LinearAlgebra.PosDefException ? Inf : rethrow()
             end))
        if needs_recovery
            isfinite(objective(last_valid[])) ||
                throw(ErrorException("JMLE could not recover a finite Float64 starting point"))
            Optim.optimize(objective, last_valid[], Optim.LBFGS();
                           autodiff=ADTypes.AutoFiniteDiff(; fdtype=Val(:central), relstep=1e-4, absstep=1e-4))
        else
            result
        end
    end

    zhat = Optim.minimizer(result)
    final = objective(zhat)

    all(isfinite, zhat) && isfinite(final) ||
        throw(ErrorException("JMLE produced non-finite parameters or log-likelihood"))

    Optim.converged(result) ||
        throw(ErrorException("JMLE did not converge after $(Optim.iterations(result)) iterations"))

    tolerance = 1e-8 * max(1.0, abs(initial))
    final <= initial + tolerance ||
        throw(ErrorException("JMLE decreased the log-likelihood relative to its initial fit"))

    return _sklar_joint_distribution(chart, zhat)
end
