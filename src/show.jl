function Base.show(io::IO, C::EmpiricalCopula)
    print(io, "EmpiricalCopula{d}$(size(C.u))")
end
function Base.show(io::IO, C::FGMCopula{d, Tθ, Tf}) where {d, Tθ, Tf}
    print(io, "FGMCopula{$d}(θ = $(C.θ))")
end
function Base.show(io::IO, C::SurvivalCopula)
    print(io, "SurvivalCopula($(C.C), $(_survival_flipindices(C.flipmask)))")
end
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
Print a standardized parameter section with optional covariance matrix and vcov method note.
"""
function _print_param_section(io, title::AbstractString, nm::Vector{String}, θ::Vector{Float64};
                              V::Union{Nothing,AbstractMatrix}=nothing,
                              vcov_method::Union{Nothing,AbstractString,Symbol}=nothing)
    suf = vcov_method === nothing ? nothing : string("(vcov=", String(vcov_method), ")")
    _section(io, title; suffix=suf)
    _print_param_table(io, nm, θ; V=V)
end

"""
Print dependence metrics if available/supported by the copula C.
"""
function _print_dependence_metrics(io, C; derived_measures::Bool=true)
    _section(io, "Dependence metrics")
    if !derived_measures
        println(io, "(suppressed)")
        return
    end
    _has(f) = isdefined(Copulas, f) && hasmethod(getfield(Copulas, f), Tuple{typeof(C)})
    shown_any = false
    try
        if _has(:τ);  _kv(io, "Kendall τ",  Printf.@sprintf("%.4f", Copulas.τ(C)));  shown_any = true; end
        if _has(:ρ);  _kv(io, "Spearman ρ", Printf.@sprintf("%.4f", Copulas.ρ(C)));  shown_any = true; end
        if _has(:β);  _kv(io, "Blomqvist β",Printf.@sprintf("%.4f", Copulas.β(C)));  shown_any = true; end
        if _has(:γ);  _kv(io, "Gini γ",     Printf.@sprintf("%.4f", Copulas.γ(C)));  shown_any = true; end
        if _has(:λᵤ); _kv(io, "Upper λᵤ",   Printf.@sprintf("%.4f", Copulas.λᵤ(C))); shown_any = true; end
        if _has(:λₗ); _kv(io, "Lower λₗ",   Printf.@sprintf("%.4f", Copulas.λₗ(C))); shown_any = true; end
        if _has(:ι);  _kv(io, "Entropy ι",  Printf.@sprintf("%.4f", Copulas.ι(C)));  shown_any = true; end
    catch
        # proceed without failing show
    end
    if !shown_any
        println(io, "(none available)")
    end
end
function _print_param_table(io, nm::Vector{String}, θ::Vector{Float64}; V::Union{Nothing, AbstractMatrix}=nothing)
    if V === nothing || isempty(θ)
        Printf.@printf(io, "%-10s %10s\n", "Parameter", "Estimate")
        @inbounds for (j, name) in pairs(nm)
            Printf.@printf(io, "%-10s %10.4f\n", String(name), θ[j])
        end
        return
    end
    se = sqrt.(LinearAlgebra.diag(V))
    z  = θ ./ se
    p  = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    lo, hi = (θ .- 1.959963984540054 .* se, θ .+ 1.959963984540054 .* se)
    Printf.@printf(io, "%-10s %10s %9s %9s %8s %10s %10s\n",
                   "Parameter","Estimate","Std.Err","z-value","p-val","95% Lo","95% Hi")
    @inbounds for j in eachindex(θ)
        Printf.@printf(io, "%-10s %10.4f %9.4f %9.3f %8s %10.4f %10.4f\n",
                        String(nm[j]), θ[j], se[j], z[j], _pstr(p[j]), lo[j], hi[j])
    end
end

function _margin_param_names(mi)
    T = typeof(mi)
    return if     T <: Distributions.Gamma;       ("α","θ")
           elseif T <: Distributions.Beta;        ("α","β")
           elseif T <: Distributions.LogNormal;   ("μ","σ")
           elseif T <: Distributions.Normal;      ("μ","σ")
           elseif T <: Distributions.Exponential; ("θ",)
           elseif T <: Distributions.Weibull;     ("k","λ")
           elseif T <: Distributions.Pareto;      ("α","θ")
           else
               k = length(Distributions.params(mi)); ntuple(j->"θ$(j)", k)
           end
end

function Base.show(io::IO, M::CopulaModel)
    R = M.result
    # Split: [ CopulaModel: ... ] vs [ Fit metrics ]
    if R isa SklarDist
        famC = _fmt_copula_family(R.C)
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        _section(io, "CopulaModel: SklarDist"; suffix="(Copula=" * famC * ", Margins=" * margins_lbl * ")")
    else
        _section(io, "CopulaModel: " * _fmt_copula_family(R))
    end
    if R isa SklarDist
        famC = _fmt_copula_family(R.C)
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        skm = get(M.method_details, :sklar_method, nothing)
        _kv(io, "Copula", famC)
        _kv(io, "Margins", margins_lbl)
        if skm === nothing
            _kv(io, "Methods", "copula=" * String(M.method))
        else
            _kv(io, "Methods", "copula=" * String(M.method) * ", sklar=" * String(skm))
        end
    else
        _kv(io, "Method", String(M.method))
    end
    _kv(io, "Number of observations", Printf.@sprintf("%d", StatsBase.nobs(M)))

    _section(io, "Fit metrics")
    ll  = M.ll
    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0); _kv(io, "Null Loglikelihood", Printf.@sprintf("%12.4f", ll0)); end
    _kv(io, "Loglikelihood", Printf.@sprintf("%12.4f", ll))
    kcop = StatsBase.dof(M)
    if isfinite(ll0) && kcop > 0
        LR = 2*(ll - ll0)
        p  = Distributions.ccdf(Distributions.Chisq(kcop), LR)
        _kv(io, "LR (vs indep.)", Printf.@sprintf("%.2f ~ χ²(%d)  ⇒  p = %s", LR, kcop, _pstr(p)))
    end
    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    _kv(io, "AIC", Printf.@sprintf("%.3f", aic))
    _kv(io, "BIC", Printf.@sprintf("%.3f", bic))
    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        _kv(io, "Converged", conv)
        _kv(io, "Iterations", string(M.iterations))
        tsec = isfinite(M.elapsed_sec) ? Printf.@sprintf("%.3fs", M.elapsed_sec) : "NA"
        _kv(io, "Elapsed", tsec)
    end

    if R isa SklarDist
        # [ Dependence metrics ] section
        C  = M.result isa SklarDist ? M.result.C : M.result
        _print_dependence_metrics(io, C; derived_measures=get(M.method_details, :derived_measures, true))

        # [ Copula parameters ] section
        θ  = StatsBase.coef(M)
        nm = StatsBase.coefnames(M)
        md   = M.method_details
        Vcop = get(md, :vcov_copula, nothing)
        vcovm = get(md, :vcov_method, nothing)
        _print_param_section(io, "Copula parameters", nm, θ; V=Vcop, vcov_method=vcovm)

        # [ Marginals ] section
        _print_marginals_section(io, R::SklarDist, get(M.method_details, :vcov_margins, nothing))
    else
        # Copula-only fits: dependence metrics and parameters
        C0 = M.result isa SklarDist ? M.result.C : M.result
        _print_dependence_metrics(io, C0; derived_measures=get(M.method_details, :derived_measures, true))
        θ  = StatsBase.coef(M)
        nm = StatsBase.coefnames(M)
        vcovm = get(M.method_details, :vcov_method, nothing)
        _print_param_section(io, "Copula parameters", nm, θ; V=StatsBase.vcov(M), vcov_method=vcovm)

    end
end

"""
Print the Marginals section for a SklarDist using precomputed Vm if available.
"""
function _print_marginals_section(io, S::SklarDist, Vm)
    _section(io, "Marginals")
    Printf.@printf(io, "%-6s %-10s %-6s %10s %9s %s\n",
                   "Margin","Dist","Param","Estimate","Std.Err","95% CI")

    crit = 1.959963984540054

    _valid_cov(V, p) = V !== nothing &&
                       ndims(V) == 2 &&
                       size(V) == (p, p) &&
                       all(isfinite, Matrix(V)) &&
                       all(LinearAlgebra.diag(Matrix(V)) .>= 0.0)

    for (i, mi) in enumerate(S.m)
        pname = String(nameof(typeof(mi)))
        θi_nt = Distributions.params(mi)
        names = _margin_param_names(mi)
        vals = Float64.(collect(θi_nt))
        p = length(vals)

        # Use only the precomputed covariance from fitting, if available and valid
        Vi = nothing
        if Vm isa Vector && 1 <= i <= length(Vm)
            Vh = Vm[i]
            if _valid_cov(Vh, p)
                Vi = Vh
            end
        end

        dV = (Vi !== nothing) ? LinearAlgebra.diag(Matrix(Vi)) : fill(NaN, p)
        se = sqrt.(max.(dV, 0.0))
        @inbounds for j in 1:p
            lab = (j == 1) ? "#$(i)" : ""
            distcol = (j == 1) ? pname : ""
            est_str = Printf.@sprintf("%.4f", vals[j])
            se_str  = isfinite(se[j]) ? Printf.@sprintf("%.4f", se[j]) : "—"
            if isfinite(se[j])
                ci_str = Printf.@sprintf("[%.4f, %.4f]", vals[j] - crit*se[j], vals[j] + crit*se[j])
            else
                ci_str = "—"
            end
            Printf.@printf(io, "%-6s %-10s %-6s %10s %9s %s\n",
                           lab, distcol, names[j], est_str, se_str, ci_str)
        end
    end
end

###############################################################################
##### Copula hypothesis tests
###############################################################################

_show_test_model(::IO, ::CopulaHypothesis, ::Val, ::Val, ::NamedTuple) = nothing
function _show_test_model(io::IO, h::GoodnessOfFitHypothesis, ::Val, ::Val, ::NamedTuple)
    label =
        h.kind === :simple ? "Specified copula" :
                             "Fitted model"
                             
    model = h.model isa CopulaModel ? _copula_of(h.model) : h.model
    model_label = replace(string(typeof(model)), "Copulas." => "")
    println(io, "Hypothesis:             ", h.kind)
    println(io, label, ":           ", model_label)
end

_show_test_details(::IO, ::CopulaHypothesis, ::Val, ::Val, ::NamedTuple) = nothing
function _show_test_details(io::IO, ::ExchangeabilityHypothesis, ::Val{:Sn},
        ::Val{:multiplier}, details::NamedTuple)
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

function _show_test_details(io::IO, ::RadialSymmetryHypothesis, ::Val{:Sn},
        ::Val{:randomization}, details::NamedTuple)
    hasproperty(details, :reflection_probability) || return nothing
    println(io, "Reflection probability: ", details.reflection_probability)
end

function _show_test_details(io::IO, ::ExtremeValueHypothesis, ::Val{:Sn},
        ::Val{:multiplier}, details::NamedTuple)
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
    statistic = Val(test.statistic)
    calibration = Val(test.calibration)
    _show_test_model(io, h, statistic, calibration, test.details)
    println(io, "Number of observations: ", StatsBase.nobs(test))
    println(io, "Dimension:              ", test.dimension)
    println(io, "Statistic:              ", replace(string(test.statistic), '_' => ' '))
    println(io, "Observed value:         ", teststatistic(test))
    _show_test_details(io, h, statistic, calibration, test.details)
    if test.n_resamples > 0
        println(io, "Number of resamples:    ", test.n_resamples)
        println(io, "Calibration:            ", replace(string(test.calibration), '_' => ' '))
    end
    println(io, "p-value:                ", pvalue(test))
    println(io)
    println(io, "Null hypothesis:")
    print(io, nullhypothesis(test))
end
