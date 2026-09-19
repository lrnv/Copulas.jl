function Base.show(io::IO, C::EmpiricalCopula)
    print(io, "EmpiricalCopula{d}$(size(C.u))")
end
function Base.show(io::IO, C::FGMCopula{d, Tθ, Tf}) where {d, Tθ, Tf}
    print(io, "FGMCopula{$d}(θ = $(C.θ))")
end
function Base.show(io::IO, C::SurvivalCopula)
    print(io, "SurvivalCopula($(basecopula(C)), $(flips(C)))")
end
Base.show(io::IO, C::Rotated90Copula) = print(io, "Rotated90Copula($(basecopula(C)))")
Base.show(io::IO, C::Rotated180Copula) = print(io, "Rotated180Copula($(basecopula(C)))")
Base.show(io::IO, C::Rotated270Copula) = print(io, "Rotated270Copula($(basecopula(C)))")
function Base.show(io::IO, C::ArchimedeanCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ExtremeValueCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ArchimaxCopula)
    print(io, "$(typeof(C))$(Distributions.params(C))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, <:𝒲}) where d
    print(io, "ArchimedeanCopula($d, 𝒲($(C.G.X), $(C.G.order)))")
end
function Base.show(io::IO, C::EllipticalCopula)
    print(io, "$(typeof(C))(Σ = $(C.Σ)))")
end
function Base.show(io::IO, G::𝒲)
    print(io, "𝒲($(G.X), $(G.order))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, <:𝒲{<:Distributions.DiscreteNonParametric}}) where d
    print(io, "ArchimedeanCopula($d, EmpiricalGenerator$((C.G.order, length(Distributions.support(C.G.X)))))")
end
function Base.show(io::IO, G::𝒲{<:Distributions.DiscreteNonParametric})
    print(io, "EmpiricalGenerator$((G.order, length(Distributions.support(G.X))))")
end
function Base.show(io::IO, C::SubsetCopula)
    print(io, "SubsetCopula($(C.C), $(C.dims))")
end
function Base.show(io::IO, tail::EmpiricalEVTail)
    print(io, "EmpiricalEVTail(", length(tail.tgrid), " knots)")
end
function Base.show(io::IO, C::ExtremeValueCopula{2, EmpiricalEVTail})
    print(io, "ExtremeValueCopula{2} ⟨", C.tail, "⟩")
end
function Base.show(io::IO, B::BernsteinCopula{d}) where {d}
    print(io, "BernsteinCopula($d, m=$(B.m))")
end
function Base.show(io::IO, C::BetaCopula)
    print(io, "BetaCopula{d}$(size(C.ranks))")
end
function Base.show(io::IO, C::CheckerboardCopula{d}) where {d}
    print(io, "CheckerboardCopula{", d, "} ⟨m=", C.m, "⟩")
end
function _fmt_copula_family(C)
    fam = String(nameof(typeof(C)))
    fam = endswith(fam, "Copula") ? fam[1:end-6] : fam
    return string(fam, " d=", length(C))
end
"""
Small horizontal rule for section separation.
"""
_hr(io) = println(io, "────────────────────────────────────────────────────────────────────────────────")

"""
Pretty p-value formatting: show very small values as inequalities.
"""
_pstr(p) = p < 1e-16 ? "<1e-16" : Printf.@sprintf("%.4g", p)

"""
Key-value aligned printing for header lines.
"""
function _kv(io, key::AbstractString, val)
    Printf.@printf(io, "%-22s %s\n", key * ":", val)
end

"""
Render a section header with optional suffix, surrounded by horizontal rules.
"""
function _section(io, title::AbstractString; suffix::Union{Nothing,AbstractString}=nothing)
    _hr(io)
    if suffix === nothing
        println(io, "[ ", title, " ]")
    else
        println(io, "[ ", title, " ] ", suffix)
    end
    _hr(io)
end

"""
Print dependence metrics if available/supported by the copula C.
"""
function _has_specialized_copula_method(f, C::Copula{d}) where {d}
    hasmethod(f, Tuple{typeof(C)}) || return false
    return which(f, Tuple{typeof(C)}) !== which(f, Tuple{Copula{d}})
end

function _print_dependence_metrics(io, C)
    _section(io, "Dependence metrics")
    _has(f) = isdefined(Copulas, f) && hasmethod(getfield(Copulas, f), Tuple{typeof(C)})
    _specialized(f) = _has(f) && _has_specialized_copula_method(getfield(Copulas, f), C)
    shown_any = false
    try
        if _specialized(:τ); _kv(io, "Kendall τ", Printf.@sprintf("%.4f", Copulas.τ(C))); shown_any = true; end
        if _specialized(:ρ);  _kv(io, "Spearman ρ", Printf.@sprintf("%.4f", Copulas.ρ(C)));  shown_any = true; end
        if _specialized(:β);  _kv(io, "Blomqvist β",Printf.@sprintf("%.4f", Copulas.β(C)));  shown_any = true; end
        if _specialized(:γ); _kv(io, "Gini γ", Printf.@sprintf("%.4f", Copulas.γ(C))); shown_any = true; end
        if _specialized(:λᵤ); _kv(io, "Upper λᵤ",   Printf.@sprintf("%.4f", Copulas.λᵤ(C))); shown_any = true; end
        if _specialized(:λₗ); _kv(io, "Lower λₗ",   Printf.@sprintf("%.4f", Copulas.λₗ(C))); shown_any = true; end
        if _specialized(:ι); _kv(io, "Entropy ι", Printf.@sprintf("%.4f", Copulas.ι(C))); shown_any = true; end
    catch
        # Display must not fail because an optional dependence metric does.
    end
    shown_any || println(io, "(none available)")
end

function _print_natural_parameter(io, name::AbstractString, value)
    if value isa Number
        _kv(io, name, value)
    elseif value isa AbstractArray
        println(io, name, ":")
        show(IOContext(io, :compact => true, :limit => true), MIME"text/plain"(), value)
        println(io)
    else
        _kv(io, name, sprint(show, value))
    end
end

function _print_distribution_parameters(io, title::AbstractString, D)
    _section(io, title)
    raw = Distributions.params(D)
    if raw isa NamedTuple
        isempty(raw) && return println(io, "(none)")
        for (name, value) in pairs(raw)
            _print_natural_parameter(io, string(name), value)
        end
    elseif raw isa Tuple
        isempty(raw) && return println(io, "(none)")
        for (name, value) in zip(_tuple_parameter_names(D, raw), raw)
            _print_natural_parameter(io, name, value)
        end
    else
        _print_natural_parameter(io, "value", raw)
    end
    return nothing
end

function _print_margins(io, S::SklarDist)
    _section(io, "Marginals")
    for (i, margin) in pairs(S.m)
        print(io, "#", i, " ", nameof(typeof(margin)), "  ")
        show(io, Distributions.params(margin))
        println(io)
    end
end

function Base.show(io::IO, M::CopulaModel)
    R = fitted_distribution(M)
    if R isa SklarDist
        famC = _fmt_copula_family(R.C)
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        _section(io, "CopulaModel: SklarDist";
                 suffix="(Copula=" * famC * ", Margins=" * margins_lbl * ")")
        _kv(io, "Copula", famC)
        _kv(io, "Margins", margins_lbl)
        _kv(io, "Methods", "copula=" * String(fitting_method(M)) *
            ", sklar=" * String(M.recipe.kwargs.sklar_method))
    else
        _section(io, "CopulaModel: " * _fmt_copula_family(R))
        _kv(io, "Method", String(fitting_method(M)))
    end
    _kv(io, "Number of observations", Printf.@sprintf("%d", StatsBase.nobs(M)))
    _kv(io, "Degrees of freedom", StatsBase.dof(M))
    _model_weights(M) === nothing ||
        _kv(io, "Observation weights", "yes, normalized to sum to the number of observations")

    _section(io, "Fit metrics")
    _kv(io, "Loglikelihood", Printf.@sprintf("%12.4f", M.loglikelihood))
    _kv(io, "AIC", Printf.@sprintf("%.3f", StatsBase.aic(M)))
    _kv(io, "BIC", Printf.@sprintf("%.3f", StatsBase.bic(M)))

    C = _copula_of(M)
    _print_dependence_metrics(io, C)
    _print_distribution_parameters(io, "Copula parameters", C)
    R isa SklarDist && _print_margins(io, R)
    return nothing
end

function Base.show(io::IO, S::CopulaSelection)
    show(io, selected_model(S))
    _section(io, "Model selection")
    _kv(io, "Criterion", uppercase(String(S.criterion)))
    _kv(io, "Selected family", _fmt_copula_family(fitted_distribution(selected_model(S))))
end

###############################################################################
##### Copula hypothesis tests
###############################################################################

_show_test_model(::IO, ::CopulaHypothesis) = nothing
function _show_test_model(io::IO, h::GoodnessOfFitHypothesis)
    label =
        h.model isa Copula ? "Specified copula" :
                             "Fitted model"

    model = h.model isa CopulaModel ? _copula_of(h.model) : h.model
    model_label = string(model)
    println(io, "Hypothesis:             ", h.model isa Copula ? :simple : :composite)
    println(io, label, ":           ", model_label)
end

_show_test_details(::IO, ::CopulaHypothesis, ::NamedTuple) = nothing
function _show_test_details(io::IO, ::ExchangeabilityHypothesis, details::NamedTuple)
    hasproperty(details, :permutations) || return nothing
    println(io, "Permutations:           ", details.permutations)
    println(io, "Weight:                 ", details.weight)
    if hasproperty(details, :multiplier)
        println(io, "Multiplier:             ", details.multiplier)
    end
    if hasproperty(details, :derivative_bandwidth)
        println(io, "Derivative bandwidth:   ", details.derivative_bandwidth)
    end
end

function _show_test_details(io::IO, ::RadialSymmetryHypothesis, details::NamedTuple)
    hasproperty(details, :reflection_probability) || return nothing
    println(io, "Reflection probability: ", details.reflection_probability)
end

function _show_test_details(io::IO, ::ExtremeValueHypothesis, details::NamedTuple)
    hasproperty(details, :powers) || return nothing
    println(io, "Powers:                 ", details.powers)
    println(io, "Multiplier:             ", details.multiplier)
    println(io, "Derivative bandwidth:   ", details.derivative_bandwidth)
end

function Base.show(io::IO, ::MIME"text/plain", test::CopulaTest)
    name = testname(test)
    println(io, name)
    println(io, repeat('-', length(name)))
    h = test.hypothesis
    _show_test_model(io, h)
    println(io, "Number of observations: ", StatsBase.nobs(test))
    println(io, "Dimension:              ", test.dimension)
    println(io, "Statistic:              ", replace(string(test.statistic), '_' => ' '))
    println(io, "Observed value:         ", teststatistic(test))
    _show_test_details(io, h, test.details)
    if test.n_resamples > 0
        println(io, "Number of resamples:    ", test.n_resamples)
        println(io, "Calibration:            ", replace(string(test.calibration), '_' => ' '))
    end
    println(io, "p-value:                ", pvalue(test))
    println(io)
    println(io, "Null hypothesis:")
    print(io, nullhypothesis(test))
end
