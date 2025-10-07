function Base.show(io::IO, C::EmpiricalCopula)
    print(io, "EmpiricalCopula{d}$(size(C.u))")
end
function Base.show(io::IO, C::FGMCopula{d, Tθ, Tf}) where {d, Tθ, Tf}
    print(io, "FGMCopula{$d}(θ = $(C.θ))")
end
function Base.show(io::IO, C::SurvivalCopula)
    print(io, "SurvivalCopula($(C.C))")
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
function Base.show(io::IO, C::ArchimedeanCopula{d, WilliamsonGenerator{d2, TX}}) where {d, d2, TX}
    print(io, "ArchimedeanCopula($d, i𝒲($(C.G.X), $d2))")
end
function Base.show(io::IO, C::EllipticalCopula)
    print(io, "$(typeof(C))(Σ = $(C.Σ)))")
end
function Base.show(io::IO, G::WilliamsonGenerator{d, TX}) where {d, TX}
    print(io, "i𝒲($(G.X), $(d))")
end
function Base.show(io::IO, C::ArchimedeanCopula{d, <:WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}}) where {d, d2}
    print(io, "ArchimedeanCopula($d, EmpiricalGenerator$((d2, length(Distributions.support(C.G.X)))))")
end
function Base.show(io::IO, G::WilliamsonGenerator{d2, <:Distributions.DiscreteNonParametric}) where {d2}
    print(io, "EmpiricalGenerator$((d2, length(Distributions.support(G.X))))")
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
function _print_param_table(io, nm::Vector{String}, θ::Vector{Float64}; V::Union{Nothing, AbstractMatrix}=nothing)
    if V === nothing || isempty(θ)
        println(io, "────────────────────────────────────────")
        Printf.@printf(io, "%-14s %12s\n", "Parameter", "Estimate")
        println(io, "────────────────────────────────────────")
        @inbounds for (j, name) in pairs(nm)
            Printf.@printf(io, "%-14s %12.4f\n", String(name), θ[j])
        end
        println(io, "────────────────────────────────────────")
        return
    end
    se = sqrt.(LinearAlgebra.diag(V))
    z  = θ ./ se
    p  = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
    lo, hi = (θ .- 1.959963984540054 .* se, θ .+ 1.959963984540054 .* se)
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    Printf.@printf(io, "%-14s %12s %12s %12s %12s %12s %12s\n",
                   "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","95% Lo","95% Hi")
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @inbounds for j in eachindex(θ)
        Printf.@printf(io, "%-14s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n",
                        String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
    end
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
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
        _hr(io); println(io, "[ CopulaModel: SklarDist (Copula=", famC, ", Margins=", margins_lbl, ") ]"); _hr(io)
    else
        _hr(io); println(io, "[ CopulaModel: ", _fmt_copula_family(R), " ]"); _hr(io)
    end
    if R isa SklarDist
        # Build copula family label
        famC = _fmt_copula_family(R.C)
        # Margins label
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        skm = get(M.method_details, :sklar_method, nothing)
        if skm === nothing
            println(io, "SklarDist{Copula=", famC, ", Margins=", margins_lbl, "} fitted via ", M.method)
        else
            println(io, "SklarDist{Copula=", famC, ", Margins=", margins_lbl, "} fitted via ",
                    "copula_method=", M.method, ", sklar_method=", skm)
        end
    else
        println(io, _fmt_copula_family(R), " fitted via ", M.method)
    end
    _kv(io, "Number of observations", Printf.@sprintf("%d", StatsBase.nobs(M)))

    _hr(io); println(io, "[ Fit metrics ]"); _hr(io)
    ll  = M.ll
    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0)
        Printf.@printf(io, "Null Loglikelihood:  %12.4f\n", ll0)
    end
    Printf.@printf(io, "Loglikelihood:       %12.4f\n", ll)

    # For the LR test use d.f. of the COPULA if it is SklarDist
    kcop = (R isa SklarDist) ? StatsBase.dof(_copula_of(M)) : StatsBase.dof(M)
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
        if R isa SklarDist
        # [ Copula ] section
        C  = _copula_of(M)
        θ  = StatsBase.coef(M)
        C  = _copula_of(M)
        θ  = StatsBase.coef(M)
        nm = StatsBase.coefnames(M)
        md   = M.method_details
        Vcop = get(md, :vcov_copula, nothing)  # <- used vcov copula
        vcovm = get(md, :vcov_method, nothing)
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Copula ]")
        println(io, "──────────────────────────────────────────────────────────")

        println(io, "Family: ", _fmt_copula_family(C))
        if vcovm !== nothing
            println(io, "vcov method: ", vcovm)
        end

        _print_param_table(io, nm, θ; V=Vcop)
        # meassures optinals
        if get(M.method_details, :derived_measures, true)
            println(io, "[ Copula Derived measures ]")

            C = _copula_of(M)
            have_any = false

            _has(f) = isdefined(Copulas, f) && hasmethod(getfield(Copulas, f), Tuple{typeof(C)})
            _print(lbl, val) = (Printf.@printf(io, "%-14s = %.4f\n", lbl, val); have_any = true)

            try
                _has(:τ)  && _print("Kendall τ(θ)",  Copulas.τ(C))
                _has(:ρ)  && _print("Spearman ρ(θ)", Copulas.ρ(C))
                _has(:β)  && _print("Blomqvist β(θ)",Copulas.β(C))
                _has(:γ)  && _print("Gini γ(θ)",     Copulas.γ(C))
                _has(:λᵤ) && _print("Upper λᵤ(θ)",   Copulas.λᵤ(C))
                _has(:λₗ)  && _print("Lower λₗ(θ)",   Copulas.λₗ(C))
                _has(:ι)  && _print("Entropy ι(θ)",  Copulas.ι(C))
            catch
                # dont break show
            end

            if !have_any
                println(io, "(none available)")
            end
        end
        # [ Marginals ] section
        S  = R::SklarDist
        md = M.method_details
        Vm = get(md, :vcov_margins, nothing)   # precomputed marginal vcov from fitting

        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Marginals ]")
        println(io, "──────────────────────────────────────────────────────────")
        Printf.@printf(io, "%-6s %-12s %-7s %12s %12s %12s\n",
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

            if Vi === nothing
                @inbounds for j in 1:p
                    lab = (j == 1) ? "#$(i)" : ""
                    Printf.@printf(io, "%-6s %-12s %-7s %12.4f %12s %12s\n",
                                lab, pname, names[j], vals[j], "—", "—")
                end
            else
                dV = LinearAlgebra.diag(Matrix(Vi))
                se = sqrt.(max.(dV, 0.0))
                lo = vals .- crit .* se
                hi = vals .+ crit .* se
                @inbounds for j in 1:p
                    lab = (j == 1) ? "#$(i)" : ""
                    Printf.@printf(io, "%-6s %-12s %-7s %12.4f %12.4f [%12.4f, %12.4f]\n",
                                lab, pname, names[j], vals[j], se[j], lo[j], hi[j])
                end
            end
        end
    else
        # Coefficient table (generic) for copula-only fits
        nm = StatsBase.coefnames(M)
        θ  = StatsBase.coef(M)
        _print_param_table(io, nm, θ; V=StatsBase.vcov(M))

    end
end
