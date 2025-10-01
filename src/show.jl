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
function Base.show(io::IO, M::CopulaModel)
    R = M.result
    # Header: family/margins without helper functions
    if R isa SklarDist
        # Build copula family label
        famC = String(nameof(typeof(R.C)))
        famC = endswith(famC, "Copula") ? famC[1:end-6] : famC
        famC = string(famC, " d=", length(R.C))
        # Margins label
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        println(io, "SklarDist{Copula=", famC, ", Margins=", margins_lbl, "} fitted via ", M.method)
    else
        fam = String(nameof(typeof(R)))
        fam = endswith(fam, "Copula") ? fam[1:end-6] : fam
        fam = string(fam, " d=", length(R))
        println(io, fam, " fitted via ", M.method)
    end

    n  = StatsBase.nobs(M)
    ll = M.ll
    Printf.@printf(io, "Number of observations: %9d\n", n)

    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0)
        Printf.@printf(io, "Null Loglikelihood:  %12.4f\n", ll0)
    end
    Printf.@printf(io, "Loglikelihood:       %12.4f\n", ll)

    # Para el test LR usa g.l. de la CÓPULA si es SklarDist
    kcop = (R isa SklarDist) ? StatsBase.dof(_copula_of(M)) : StatsBase.dof(M)
    if isfinite(ll0) && kcop > 0
        LR = 2*(ll - ll0)
        p  = Distributions.ccdf(Distributions.Chisq(kcop), LR)
        Printf.@printf(io, "LR Test (vs indep. copula): %.2f ~ χ²(%d)  =>  p = %.4g\n", LR, kcop, p)
    end

    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    Printf.@printf(io, "AIC: %.3f       BIC: %.3f\n", aic, bic)
    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        tsec = isfinite(M.elapsed_sec) ? Printf.@sprintf("%.3fs", M.elapsed_sec) : "NA"
        println(io, "Converged: $(conv)   Iterations: $(M.iterations)   Elapsed: $(tsec)")
    end

    # Branches: SklarDist → sections; empirical → summary; else → coefficient table
    if R isa SklarDist
        # [ Copula ] section
        C = _copula_of(M)
        θ = StatsBase.coef(M)
        nm = StatsBase.coefnames(M)
        V  = StatsBase.vcov(M)
        lvl = 95
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Copula ]")
        println(io, "──────────────────────────────────────────────────────────")
        fam = String(nameof(typeof(C))); fam = endswith(fam, "Copula") ? fam[1:end-6] : fam; fam = string(fam, " d=", length(C))
        Printf.@printf(io, "%-16s %-9s %10s %10s %12s\n", "Family","Param","Estimate","Std.Err","$lvl% CI")
        if V === nothing || isempty(θ)
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n", fam, String(nm[j]), θ[j], "—", "—")
            end
        else
            se = sqrt.(LinearAlgebra.diag(V))
            lo, hi = StatsBase.confint(M; level=0.95)
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-16s %-9s %10.3g %10.3g [%0.3g, %0.3g]\n", fam, String(nm[j]), θ[j], se[j], lo[j], hi[j])
            end
        end
        if isdefined(Copulas, :τ) && hasmethod(Copulas.τ, Tuple{typeof(C)})
            τth = Copulas.τ(C)
            Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n", "Kendall", "τ(θ)", τth, "—", "—")
        end

        # [ Marginals ] section
        S = R::SklarDist
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Marginals ]")
        println(io, "──────────────────────────────────────────────────────────")
        Printf.@printf(io, "%-6s %-12s %-7s %10s %10s %12s\n", "Margin","Dist","Param","Estimate","Std.Err","$lvl% CI")
        for (i, mi) in enumerate(S.m)
            pname = String(nameof(typeof(mi)))
            θi    = Distributions.params(mi)
            # Inline param name mapping
            T = typeof(mi)
            names = if     T <: Distributions.Gamma;       ("α","θ")
                    elseif T <: Distributions.Beta;        ("α","β")
                    elseif T <: Distributions.LogNormal;   ("μ","σ")
                    elseif T <: Distributions.Normal;      ("μ","σ")
                    elseif T <: Distributions.Exponential; ("θ",)
                    elseif T <: Distributions.Weibull;     ("k","λ")
                    elseif T <: Distributions.Pareto;      ("α","θ")
                    else
                        k = length(θi); ntuple(j->"θ$(j)", k)
                    end
            @inbounds for j in eachindex(θi)
                lab = (j == 1) ? "#$(i)" : ""
                Printf.@printf(io, "%-6s %-12s %-7s %10.3g %10s %12s\n", lab, pname, names[j], θi[j], "—", "—")
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
                Printf.@printf(io, "%-14s %12.6g\n", String(name), θ[j])
            end
            println(io, "────────────────────────────────────────")
        else
            se = sqrt.(LinearAlgebra.diag(V))
            z  = θ ./ se
            p  = 2 .* Distributions.ccdf(Distributions.Normal(), abs.(z))
            lo, hi = StatsBase.confint(M; level=0.95)
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            Printf.@printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n", "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","95% Lo","95% Hi")
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n", String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
            end
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
        end
    end
end
