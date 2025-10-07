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
        _hr(io); println(io, "[ CopulaModel: SklarDist (Copula=", famC, ", Margins=", margins_lbl, ") ]"); _hr(io)
    else
        _hr(io); println(io, "[ CopulaModel: ", _fmt_copula_family(R), " ]"); _hr(io)
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
        lvl = 95
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Copula ]")
        println(io, "──────────────────────────────────────────────────────────")

        fam = String(nameof(typeof(C)))
        fam = endswith(fam, "Copula") ? fam[1:end-6] : fam
        fam = string(fam, " d=", length(C))
        println(io, "Family: ", fam)

        if Vcop === nothing || isempty(θ)
            Printf.@printf(io, "%-12s %12s\n", "Param","Estimate")
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-12s %12.4f\n", String(nm[j]), θ[j])
            end
        else
            dV = LinearAlgebra.diag(Matrix(Vcop))
            if length(dV) == length(θ)
                se   = sqrt.(max.(dV, 0.0))
                crit = 1.959963984540054
                z    = θ ./ se
                p    = 2 .* Distributions.ccdf.(Distributions.Normal(), abs.(z))
                lo   = θ .- crit .* se
                hi   = θ .+ crit .* se

                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
                Printf.@printf(io, "%-12s %12s %12s %9s %10s %12s %12s\n",
                            "Param","Estimate","Std.Err","z-value","Pr(>|z|)","95% Lo","95% Hi")
                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
                @inbounds for j in eachindex(θ)
                    Printf.@printf(io, "%-12s %12.4f %12.4f %9.3f %10.3g %12.4f %12.4f\n",
                                String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
                end
                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            end
        end
        # meassures optinals
        if get(M.method_details, :derived_measures, true)
            println(io, "[ Copula Derived measures ]")

            C = _copula_of(M)
            have_any = false

            _has(f) = isdefined(Copulas, f) && hasmethod(getfield(Copulas, f), Tuple{typeof(C)})
            _print(lbl, val) = (Printf.@printf(io, "%-14s = %.4f\n", lbl, val); have_any = true)

            try
                _has(:τ)    && _print("Kendall τ(θ)",  Copulas.τ(C))
                _has(:ρ)    && _print("Spearman ρ(θ)", Copulas.ρ(C))
                _has(:β)    && _print("Blomqvist β(θ)",Copulas.β(C))
                _has(:γ)    && _print("Gini γ(θ)",     Copulas.γ(C))
                _has(:λᵤ)   && _print("Upper λᵤ(θ)",   Copulas.λᵤ(C))
                _has(:λₗ)    && _print("Lower λₗ(θ)",   Copulas.λₗ(C))
                _has(:ι)    && _print("Entropy ι(θ)",  Copulas.ι(C).H)
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
        Vm = get(md, :vcov_margins, nothing)   # Vector{Union{Nothing,Matrix}} o nothing
        Xm = get(md, :X_margins, nothing)      # Vector{Vector} opcional (para fallback genérico)

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
                        all(diag(Matrix(V)) .>= 0.0)

        function _pick_Vi(i, mi, p, Vm, Xm)
            Vi = nothing

            # 1) method_details[:vcov_margins]
            if Vm isa Vector && 1 <= i <= length(Vm)
                Vh = Vm[i]
                if _valid_cov(Vh, p)
                    return Vh
                end
            end

            # 2)marginal vcov
            try
                V0 = StatsBase.vcov(mi)
                if _valid_cov(V0, p)
                    return V0
                end
            catch
                # no-op
            end

            # 3) generic fallback data saved
            if Xm !== nothing
                try
                    Vg = _vcov_margin_generic(mi, Xm[i])
                    if _valid_cov(Vg, p)
                        return Vg
                    end
                catch
                    # no-op
                end
            end

            return nothing
        end

        for (i, mi) in enumerate(S.m)
            pname = String(nameof(typeof(mi)))
            θi_nt = Distributions.params(mi)
            # names..,
            T = typeof(mi)
            names = if     T <: Distributions.Gamma;       ("α","θ")
                    elseif T <: Distributions.Beta;        ("α","β")
                    elseif T <: Distributions.LogNormal;   ("μ","σ")
                    elseif T <: Distributions.Normal;      ("μ","σ")
                    elseif T <: Distributions.Exponential; ("θ",)
                    elseif T <: Distributions.Weibull;     ("k","λ")
                    elseif T <: Distributions.Pareto;      ("α","θ")
                    else
                        k = length(θi_nt); ntuple(j->"θ$(j)", k)
                    end

            vals = Float64.(collect(θi_nt))
            p = length(vals)

            Vi = _pick_Vi(i, mi, p, Vm, Xm)

            if Vi === nothing
                @inbounds for j in 1:p
                    lab = (j == 1) ? "#$(i)" : ""
                    Printf.@printf(io, "%-6s %-12s %-7s %12.4f %12s %12s\n",
                                lab, pname, names[j], vals[j], "—", "—")
                end
            else
                dV = diag(Matrix(Vi))
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

        elseif StatsBase.dof(M) == 0 || M.method == :emp
        # Empirical summary
        md   = M.method_details
        kind = get(md, :emp_kind, :unspecified)
        d    = get(md, :d, missing)
        n    = get(md, :n, missing)
        pv   = get(md, :pseudo_values, missing)

        hdr = "d=$(d), n=$(n)" * (pv === missing ? "" : ", pseudo_values=$(pv)")
        extra = ""
        if kind === :bernstein
            m = get(md, :m, nothing)
            extra = m === nothing ? "" : ", m=$(m)"
        elseif kind === :exact
            m = get(md, :m, nothing)
            extra = m === nothing ? "" : ", m=$(m)"
        elseif kind === :ev_tail
            method = get(md, :method, :unspecified)
            grid   = get(md, :grid, missing)
            eps    = get(md, :eps,  missing)
            extra  = ", method=$(method), grid=$(grid), eps=$(eps)"
        end

        println(io, "Empirical summary ($kind)")
        println(io, hdr * extra)

        # Estadísticos clásicos
        has_tau  = all(haskey.(Ref(md), (:tau_mean, :tau_sd, :tau_min, :tau_max)))
        has_rho  = all(haskey.(Ref(md), (:rho_mean, :rho_sd, :rho_min, :rho_max)))
        has_beta = all(haskey.(Ref(md), (:beta_mean, :beta_sd, :beta_min, :beta_max)))
        has_gamma = all(haskey.(Ref(md), (:gamma_mean, :gamma_sd, :gamma_min, :gamma_max)))

        if d === missing || d == 2
            println(io, "────────────────────────────")
            Printf.@printf(io, "%-10s %18s\n", "Stat", "Value")
            println(io, "────────────────────────────")
                if has_tau; Printf.@printf(io, "%-10s %18.3f\n", "tau", md[:tau_mean]); end
                if has_rho; Printf.@printf(io, "%-10s %18.3f\n", "rho", md[:rho_mean]); end
                if has_beta; Printf.@printf(io, "%-10s %18.3f\n", "beta", md[:beta_mean]); end
                if has_gamma; Printf.@printf(io, "%-10s %18.3f\n", "gamma", md[:gamma_mean]); end
            println(io, "────────────────────────────")
        else
            println(io, "───────────────────────────────────────────────────────")
            Printf.@printf(io, "%-10s %10s %10s %10s %10s\n", "Stat", "Mean", "SD", "Min", "Max")
            println(io, "───────────────────────────────────────────────────────")
            if has_tau
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "tau", md[:tau_mean], md[:tau_sd], md[:tau_min], md[:tau_max])
            end
            if has_rho
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "rho", md[:rho_mean], md[:rho_sd], md[:rho_min], md[:rho_max])
            end
            if has_beta
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "beta", md[:beta_mean], md[:beta_sd], md[:beta_min], md[:beta_max])
            end
            if has_gamma
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "gamma", md[:gamma_mean], md[:gamma_sd], md[:gamma_min], md[:gamma_max])
            end
            println(io, "───────────────────────────────────────────────────────")
        end
    else
        # Coefficient table
        params = Distributions.params(_copula_of(M))
        C = _copula_of(M)
        if C isa GaussianCopula
            Σ = params.Σ
            d = size(Σ, 1)
            θ = Float64[]
            nm = String[]
            @inbounds for j in 2:d, i in 1:j-1
                push!(θ, float(Σ[i, j]))
                push!(nm, "Σ_$(i)_$(j)")
            end

            V = StatsBase.vcov(M) 
            if V === nothing || isempty(θ)
                println(io, "────────────────────────────────────────")
                Printf.@printf(io, "%-14s %12s\n", "Parameter", "Estimate")
                println(io, "────────────────────────────────────────")
                @inbounds for j in eachindex(θ)
                    Printf.@printf(io, "%-14s %12.6g\n", nm[j], θ[j])
                end
                println(io, "────────────────────────────────────────")
            else
                se   = sqrt.(LinearAlgebra.diag(V))
                crit = 1.959963984540054  # z_{0.975}
                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
                Printf.@printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n",
                               "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","95% Lo","95% Hi")
                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
                @inbounds for j in eachindex(θ)
                    s  = se[j]
                    z  = (isfinite(s) && s > 0) ? θ[j]/s : NaN
                    p  = isfinite(z) ? 2*Distributions.ccdf(Distributions.Normal(), abs(z)) : NaN
                    lo = isfinite(s) ? θ[j] - crit*s : NaN
                    hi = isfinite(s) ? θ[j] + crit*s : NaN
                    Printf.@printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n",
                                   nm[j], θ[j], s, z, p, lo, hi)
                end
                println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            end
            return
        end
        # Linearize the parameters: 
        θ = Float64[]
        nm = String[]
        for (k, v) in pairs(params)
            if isa(v, Number)
                push!(θ, float(v))
                push!(nm, String(k))
            elseif isa(v, AbstractMatrix)
                for i in axes(v, 1), j in axes(v, 2)
                    push!(θ, float(v[i, j]))
                    push!(nm, "$(k)_$(i)_$(j)")
                end
            elseif isa(v, AbstractVector)
                for i in eachindex(v)
                    push!(θ, float(v[i]))
                    push!(nm, "$(k)_$(i)")
                end
            else
                try
                    push!(θ, float(v))
                    push!(nm, String(k))
                catch
                end
            end
        end

        V  = StatsBase.vcov(M)
        if V === nothing || isempty(θ)
            println(io, "────────────────────────────────────────")
            Printf.@printf(io, "%-14s %12s\n", "Parameter", "Estimate")
            println(io, "────────────────────────────────────────")
            @inbounds for (j, name) in pairs(nm)
                Printf.@printf(io, "%-14s %12.4f\n", String(name), θ[j])
            end
            println(io, "────────────────────────────────────────")
        else
            se = sqrt.(LinearAlgebra.diag(V))
            z  = θ ./ se
            p  = 2 .* Distributions.ccdf(Distributions.Normal(), abs.(z))
            lo, hi = StatsBase.confint(M; level=0.95)
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

    end
end
