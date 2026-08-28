# Equivalence obligation: deterministic optimized implementations must agree
# with a generic fallback or an independent mathematical oracle.
# Multivariate Archimedean, EV, Liouville, nested, and Gaussian formulas are
# covered by their defining identities in correctness/mathematical.jl and the
# focused family regressions. Singular and mixed CDFs have no Lebesgue-density
# fallback; their mass identities and sampler structure are checked there too.

function _unique_bivariate_routes(operation, predicate)
    seen = Set{Method}()
    routes = NamedTuple[]
    for case in ROUTING_COPULA_CASES
        C = case.build()
        length(C) == 2 || continue
        predicate(case, C) || continue
        method = operation(case, C)
        method in seen && continue
        push!(seen, method)
        push!(routes, (; case, C, method))
    end
    return routes
end

@testset "specialized continuous CDFs agree with density integration" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Copulas._cdf, Tuple{typeof(C),Vector{Float64}}),
        (case, _) -> case.kind === :continuous,
    )
    generic_method = which(Copulas._cdf,
        Tuple{Copulas.Copula,Vector{Float64}})
    compared = 0
    u = [0.53, 0.67]
    for (; case, C, method) in routes
        method === generic_method && continue
        expected = invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
        @info "Comparing specialized CDF with generic integration" copula=case.name method
        @test isapprox(cdf(C, u), expected; atol=3e-5, rtol=3e-5)
        compared += 1
    end
    @test compared > 0
end

@testset "specialized bivariate log-densities agree with CDF derivatives" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Distributions._logpdf,
                        Tuple{typeof(C),Vector{Float64}}),
        (case, _) -> case.kind === :continuous,
    )
    u = [0.53, 0.67]
    h = 2e-5
    for (; case, C, method) in routes
        expected = (
            cdf(C, u .+ (h, h)) - cdf(C, u .+ (h, -h)) -
            cdf(C, u .+ (-h, h)) + cdf(C, u .- (h, h))
        ) / (4h^2)
        @info "Comparing log-density route with mixed CDF derivative" copula=case.name method
        @test isapprox(pdf(C, u), expected; atol=8e-4, rtol=8e-4)
        @test logpdf(C, u) ≈ log(pdf(C, u))
    end
    @test !isempty(routes)
end

@testset "specialized dependence measures agree with generic definitions" begin
    # Entropy and Gini's gamma use substantially more expensive multidimensional
    # expectations and are covered by their independent identities in
    # correctness/. Kendall's generic definition is stochastic, so singular
    # Kendall formulas keep their exact family identities instead of a noisy,
    # repeated 10_000-observation comparison here. The CDF-only definitions of
    # rho, beta and tail dependence remain valid for singular and mixed laws.
    for index in (1, 2, 3, 6, 7)
        measure = SCALAR_DEPENDENCE_MEASURES[index]
        routes = _unique_bivariate_routes(
            (_, C) -> which(measure, Tuple{typeof(C)}),
            (case, _) -> measure === Copulas.τ ?
                case.kind === :continuous : true,
        )
        generic_method = which(measure, Tuple{Copulas.Copula{2}})
        for (; case, C, method) in routes
            method === generic_method && continue
            expected = invoke(measure, Tuple{Copulas.Copula}, C)
            @info "Comparing specialized dependence measure with generic definition" measure=nameof(measure) copula=case.name method
            @test isapprox(measure(C), expected; atol=3e-4, rtol=3e-4)
        end
    end
end

@testset "limit and subset dependence routes agree with independent identities" begin
    independence = IndependentCopula{2}()
    @test Copulas.γ(independence) == 0
    @test Copulas.ι(independence) == 0
    @test Copulas.γ(MCopula{2}()) == 1
    @test Copulas.ι(MCopula{2}()) == -Inf

    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    for measure in SCALAR_DEPENDENCE_MEASURES
        @test measure(subset) == measure(parent)
    end
end

@testset "singular Kendall routes agree with sample concordance" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Copulas.τ, Tuple{typeof(C)}),
        (case, _) -> case.kind !== :continuous,
    )
    generic_method = which(Copulas.τ, Tuple{Copulas.Copula{2}})
    compared = 0
    for (index, route) in pairs(routes)
        (; case, C, method) = route
        method === generic_method && continue
        U = rand(StableRNG(8_000 + index), C, 600)
        empirical = StatsBase.corkendall(transpose(U))[1, 2]
        @info "Comparing singular Kendall route with sample concordance" copula=case.name method
        @test Copulas.τ(C) ≈ empirical atol=0.12
        compared += 1
    end
    @test compared > 0
end

@testset "all gamma and entropy dispatches have an independent proof" begin
    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    candidates = Any[]
    for case in ROUTING_COPULA_CASES
        C = case.build()
        length(C) == 2 && push!(candidates, C)
    end
    push!(candidates, subset)

    for (measure, checked) in (
        (Copulas.γ, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
        (Copulas.ι, (PolynomialOracleCopula(0.4), IndependentCopula{2}(),
                     MCopula{2}(), subset)),
    )
        selected_methods = Set(which(measure, Tuple{typeof(C)}) for C in candidates)
        checked_methods = Set(which(measure, Tuple{typeof(C)}) for C in checked)
        @test selected_methods == checked_methods
    end
end

@testset "specialized FGM paths agree with the generic polynomial oracle" begin
    θ = 0.4
    generic = PolynomialOracleCopula(θ)
    specialized = FGMCopula{2}(θ)
    u = [0.37, 0.68]

    generic_integrated_cdf =
        invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, generic, u)
    @test cdf(specialized, u) ≈ generic_integrated_cdf atol=2e-5
    @test pdf(specialized, u) ≈ pdf(generic, u)
    @test Copulas.measure(specialized, [0.15, 0.25], [0.55, 0.65]) ≈
          Copulas.measure(generic, [0.15, 0.25], [0.55, 0.65])

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2])
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6

    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
    @test inverse_rosenblatt(specialized, rosenblatt(specialized, u)) ≈
          inverse_rosenblatt(generic, rosenblatt(generic, u)) atol=2e-6
    @test Copulas.ρ(specialized) ≈ Copulas.ρ(generic) atol=2e-5
    @test Copulas.β(specialized) ≈ Copulas.β(generic)
end

@testset "specialized Gumbel generator agrees with its generic oracle" begin
    θ = 1.5
    generic = PowerExponentialOracleGenerator(θ)
    specialized = Copulas.GumbelGenerator(θ)
    for t in (0.2, 0.7, 1.4)
        p = Copulas.ϕ(generic, t)
        @test Copulas.ϕ(specialized, t) ≈ p
        @test Copulas.ϕ⁻¹(specialized, p) ≈ Copulas.ϕ⁻¹(generic, p)
        @test Copulas.ϕ⁽¹⁾(specialized, t) ≈ Copulas.ϕ⁽¹⁾(generic, t)
        @test Copulas.ϕ⁽ᵏ⁾(specialized, 2, t) ≈
              Copulas.ϕ⁽ᵏ⁾(generic, 2, t)
        @test Copulas.ϕ⁻¹⁽¹⁾(specialized, p) ≈ Copulas.ϕ⁻¹⁽¹⁾(generic, p)
    end
end

@testset "specialized logistic tail agrees with its generic oracle" begin
    θ = 1.5
    generic_tail = LogisticOracleTail(θ)
    specialized_tail = Copulas.LogTail(θ)
    x = [0.4, 0.7]
    weight = Tuple(x ./ sum(x))
    @test Copulas.ℓ(specialized_tail, x) ≈ Copulas.ℓ(generic_tail, x)
    @test Copulas.A(specialized_tail, weight) ≈ Copulas.A(generic_tail, weight)
    for indices in ((), (1,), (2,), (1, 2))
        @test Copulas.ellpartial(specialized_tail, x, indices) ≈
              Copulas.ellpartial(generic_tail, x, indices) atol=2e-6
    end

    generic = ExtremeValueCopula{2}(generic_tail)
    specialized = LogCopula{2}(θ)
    u = [0.37, 0.68]
    @test cdf(specialized, u) ≈ cdf(generic, u)
    @test pdf(specialized, u) ≈ pdf(generic, u) atol=2e-6

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2]) atol=2e-6
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6
    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
end

@testset "all distortion quantile specializations agree with generic inversion" begin
    generic_method = which(quantile, Tuple{Copulas.Distortion,Real})
    seen = Set{Method}()
    for (name, D, kind) in DISTORTION_CASES
        kind === :continuous || continue
        method = which(quantile, Tuple{typeof(D),Float64})
        method === generic_method && continue
        method in seen && continue
        push!(seen, method)
        generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, 0.63)
        @info "Comparing distortion quantile route" distortion=name method
        @test isapprox(quantile(D, 0.63), generic; atol=2e-8, rtol=2e-8)
    end
    @test !isempty(seen)
end

@testset "bivariate conditioning routes agree with CDF derivatives" begin
    seen = Set{Method}()
    for case in ROUTING_COPULA_CASES
        C = case.build()
        length(C) == 2 || continue
        case.kind === :continuous || continue
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),Tuple{Int},Tuple{Float64},Int})
        method in seen && continue
        push!(seen, method)

        conditioned, target = 0.41, 0.63
        h = 2e-5
        D = condition(C, 1, conditioned)
        expected_cdf = (cdf(C, [conditioned + h, target]) -
                        cdf(C, [conditioned - h, target])) / (2h)
        expected_pdf = (
            cdf(C, [conditioned + h, target + h]) -
            cdf(C, [conditioned + h, target - h]) -
            cdf(C, [conditioned - h, target + h]) +
            cdf(C, [conditioned - h, target - h])
        ) / (4h^2)
        @info "Comparing conditioning route with mixed CDF derivatives" copula=case.name method
        @test isapprox(cdf(D, target), expected_cdf;
                       atol=3e-5, rtol=3e-5)
        @test isapprox(pdf(D, target), expected_pdf;
                       atol=3e-4, rtol=3e-4)
    end
    @test !isempty(seen)
end

function _finite_conditional_cdf(C, js, values, target_index, target; h=2e-4)
    d = length(C)
    function mixed_at(target_value)
        total = 0.0
        for corner in Iterators.product(ntuple(_ -> (-1, 1), length(js))...)
            point = ones(d)
            point[target_index] = target_value
            for k in eachindex(js)
                point[js[k]] = values[k] + corner[k] * h
            end
            total += prod(corner) * cdf(C, point)
        end
        return total / (2h)^length(js)
    end
    return mixed_at(target) / mixed_at(1.0)
end

@testset "multivariate conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Method}()
    for case in ROUTING_COPULA_CASES
        C = case.build()
        d = length(C)
        d > 2 || continue
        case.kind === :continuous || continue
        js = Tuple(1:(d - 1))
        values = ntuple(k -> 0.3 + 0.08k, d - 1)
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),typeof(js),typeof(values),Int})
        method in seen && continue
        push!(seen, method)

        target_index = d
        target = 0.63
        D = condition(C, js, values)
        expected = _finite_conditional_cdf(
            C, js, values, target_index, target)
        @info "Comparing multivariate conditioning route with normalized CDF derivatives" copula=case.name method
        @test isapprox(cdf(D, target), expected; atol=2e-3, rtol=2e-3)
    end
    @test !isempty(seen)
end

@testset "specialized Rosenblatt implementations agree with the generic path" begin
    u = [0.2 0.7; 0.4 0.6; 0.8 0.3]
    for C in (
        ClaytonCopula{3}(1.5),
        GaussianCopula{3}([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
        TCopula{3}(5, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    )
        specialized = rosenblatt(C, u)
        generic = invoke(Copulas.rosenblatt,
            Tuple{Copulas.Copula{3},AbstractMatrix{<:Real}}, C, u)
        @test specialized ≈ generic atol=3e-10
        @test inverse_rosenblatt(C, specialized) ≈ u atol=3e-10
    end
end

@testset "all specialized Rosenblatt routes have an equivalence proof" begin
    checked = (
        ClaytonCopula{3}(1.5),
        GaussianCopula{3}([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
        TCopula{3}(5, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    )
    generic_method = which(Copulas.rosenblatt,
        Tuple{Copulas.Copula{3},Matrix{Float64}})
    candidates = Any[checked[3]]
    for case in ROUTING_COPULA_CASES
        C = case.build()
        length(C) == 3 && case.rosenblatt && push!(candidates, C)
    end
    selected_methods = Set(
        which(Copulas.rosenblatt, Tuple{typeof(C),Matrix{Float64}})
        for C in candidates
        if which(Copulas.rosenblatt,
                 Tuple{typeof(C),Matrix{Float64}}) !== generic_method
    )
    checked_methods = Set(
        which(Copulas.rosenblatt, Tuple{typeof(C),Matrix{Float64}})
        for C in checked
    )
    @test selected_methods == checked_methods

    generic_inverse_method = which(Copulas.inverse_rosenblatt,
        Tuple{Copulas.Copula{3},Matrix{Float64}})
    selected_inverse_methods = Set(
        which(Copulas.inverse_rosenblatt,
              Tuple{typeof(C),Matrix{Float64}})
        for C in candidates
        if which(Copulas.inverse_rosenblatt,
                 Tuple{typeof(C),Matrix{Float64}}) !== generic_inverse_method
    )
    checked_inverse_methods = Set(
        which(Copulas.inverse_rosenblatt,
              Tuple{typeof(C),Matrix{Float64}})
        for C in checked
    )
    @test selected_inverse_methods == checked_inverse_methods
end

@testset "EV analytic partials agree with the differentiable CDF path" begin
    f(z) = z[1]^2 * z[2]^3 + z[3]
    mixed_point = [0.4, 0.7, 1.1]
    expected12 = 6 * mixed_point[1] * mixed_point[2]^2
    @test Copulas._mixed_partial(f, mixed_point, (1, 2)) ≈ expected12
    @test Copulas._mixed_partial(f, Tuple(mixed_point), [1, 2]) ≈ expected12

    z = [0.31, 0.57, 0.73]
    for C in (LogCopula{3}(2.0), GalambosCopula{3}(0.7))
        analytic = Copulas._partial_cdf(C, (3,), (1, 2),
                                        (z[3],), (z[1], z[2]))
        differentiated = ForwardDiff.derivative(
            a -> ForwardDiff.derivative(
                b -> cdf(C, [a, b, z[3]]), z[2]), z[1])
        @test analytic ≈ differentiated atol=1e-11 rtol=2e-8
    end

    # Numerical-kernel tails cannot accept dual numbers; their analytic STDF
    # partials must nevertheless power conditioning and Rosenblatt end to end.
    C = tEVCopula{3}(4.0, 0.2)
    D = condition(C, (1, 2), (0.31, 0.58))
    q = quantile(D, 0.6)
    @test cdf(D, q) ≈ 0.6 atol=2e-6 rtol=2e-6
    u = [0.21, 0.53, 0.74]
    @test inverse_rosenblatt(C, rosenblatt(C, u)) ≈ u atol=2e-6 rtol=2e-6
end

@testset "conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula{4}(2.0)
    xf = [0.3, 0.5, 0.4, 0.6]
    xb = big.(xf)

    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db.den isa BigFloat
    @test eltype(db.uⱼₛ) === BigFloat
    @test cdf(db, xb[2]) isa BigFloat
    @test Float64(cdf(db, xb[2])) ≈ cdf(df, xf[2]) atol=1e-9

    mb = condition(C, (1, 3), Tuple(xb[[1, 3]]))
    @test mb.C.den isa BigFloat
    @test cdf(mb, xb[[2, 4]]) isa BigFloat

    C3 = ClaytonCopula{3}(2.0)
    @test condition(C3, 1, big"0.3") isa SklarDist
    X = SklarDist(C3, (Normal(), LogNormal(), Exponential()))
    big_conditioned = condition(X, (1,), (big"0.2",))
    float_conditioned = condition(X, (1,), (0.2,))
    @test big_conditioned isa SklarDist
    @test cdf(big_conditioned, [0.3, 0.5]) ≈
          cdf(float_conditioned, [0.3, 0.5]) atol=1e-6
    @test condition(ClaytonCopula{3}(2.0), (1, 2), (0.3f0, 0.4f0)) isa
          Copulas.Distortion
end


@testset "generic numeric sampler buffers" begin
    C = ClaytonCopula{3}(1.0)
    storage = fill(Float32(NaN), 5, 2)
    buffer = @view storage[2:4, :]
    @test rand!(StableRNG(52), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    @test all(isnan, storage[[1, 5], :])
    @test_throws DimensionMismatch rand!(StableRNG(52), C, zeros(Float32, 2, 1))
end
