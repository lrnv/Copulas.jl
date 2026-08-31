# Equivalence obligation: deterministic optimized implementations must agree
# with a generic fallback or an independent mathematical oracle.
# Multivariate Archimedean, EV, Liouville, nested, and Gaussian formulas are
# covered by their defining identities in the correctness obligations.
# Singular and mixed CDFs have no Lebesgue-density
# fallback; their mass identities and sampler structure are checked there too.

@testset "all documented Nataf dispatches have an oracle" begin
    r, s = 0.2, 0.8
    lognormal_scale = sqrt(expm1(s^2))
    uniform_lognormal = sqrt(2) / s * quantile(
        Normal(), 1 / 2 + r * lognormal_scale / (2sqrt(3)))
    exact_cases = (
        (Normal(), Normal(2, 3), r),
        (LogNormal(0, s), LogNormal(1, s), log1p(r * expm1(s^2)) / s^2),
        (Normal(), LogNormal(0, s), r * lognormal_scale / s),
        (LogNormal(0, s), Normal(), r * lognormal_scale / s),
        (Uniform(), Uniform(-2, 3), 2sinpi(r / 6)),
        (Uniform(), Normal(), r * sqrt(π / 3)),
        (Normal(), Uniform(), r * sqrt(π / 3)),
        (Uniform(), LogNormal(0, s), uniform_lognormal),
        (LogNormal(0, s), Uniform(), uniform_lognormal),
    )
    checked = Set{Method}()
    for (Fᵢ, Fⱼ, expected) in exact_cases
        @test Nataf((Fᵢ, Fⱼ), r) ≈ expected
        push!(checked, which(Copulas._nataf_problem,
            Tuple{typeof(Fᵢ),typeof(Fⱼ),Float64,Int}))
    end

    # The generic quadrature route is independently validated end to end in
    # correctness/nataf.jl; here it is included in the dispatch inventory and its
    # pair symmetry is checked directly.
    Fᵢ, Fⱼ = Gamma(2.0, 1.0), Beta(2.0, 3.0)
    generic = Nataf((Fᵢ, Fⱼ), r; nodes=8)
    @test generic ≈ Nataf((Fⱼ, Fᵢ), r; nodes=8) atol=1e-7
    @test -1 < generic < 1
    push!(checked, which(Copulas._nataf_problem,
        Tuple{typeof(Fᵢ),typeof(Fⱼ),Float64,Int}))

    documented_pairs = (
        (Normal(), Normal()), (Normal(), LogNormal(0, s)),
        (Normal(), Uniform()), (LogNormal(0, s), Normal()),
        (LogNormal(0, s), LogNormal(0, s)), (LogNormal(0, s), Uniform()),
        (Uniform(), Normal()), (Uniform(), LogNormal(0, s)),
        (Uniform(), Uniform()), (Fᵢ, Fⱼ),
    )
    selected = Set(which(Copulas._nataf_problem,
        Tuple{typeof(a),typeof(b),Float64,Int}) for (a, b) in documented_pairs)
    @test selected == checked
end

function _unique_bivariate_routes(operation, predicate)
    Base.@nospecialize operation predicate
    seen = Set{Method}()
    routes = NamedTuple[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        predicate(case, C) || continue
        method = operation(case, C)
        method in seen && continue
        push!(seen, method)
        push!(routes, (; case, C, method))
    end
    return routes
end

@testset verbose=true "specialized continuous CDFs agree with density integration" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Copulas._cdf, Tuple{typeof(C),Vector{Float64}}),
        (case, C) -> is_absolutely_continuous(C) &&
            !(C isa Union{CheckerboardCopula,LiouvilleCopula}),
    )
    generic_method = which(Copulas._cdf,
        Tuple{Copulas.Copula,Vector{Float64}})
    compared = 0
    u = [0.53, 0.67]
    for (; case, C, method) in routes
        if method === generic_method
            # The generic density integral is independently validated by the
            # polynomial oracle in correctness/mathematical.jl.
            prove_dispatch_route!(:cdf, C, :generic_density_integral)
            continue
        end
        @testset "$(case.name)" begin
            test_progress("equivalence", "cdf", case.name)
            expected = if C isa ArchimedeanCopula
                Copulas.ϕ(C.G, sum(Copulas.ϕ⁻¹(C.G, x) for x in u))
            else
                invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, C, u)
            end
            @test isapprox(cdf(C, u), expected;
                           atol=max(3e-5, case.numerical_atol), rtol=3e-5)
        end
        prove_dispatch_route!(:cdf, C,
                              C isa ArchimedeanCopula ?
                              :generator_composition : :density_integration)
        compared += 1
    end
    @test compared > 0
end

@testset "checkerboard CDF equals exact box overlap" begin
    fixture = only(filter(x -> x.copula isa CheckerboardCopula,
                          ROUTING_COPULA_FIXTURES))
    case, C = fixture.case, fixture.copula
    u = [0.53, 0.67]
    expected = zero(eltype(values(C.boxes)))
    for (box, weight) in C.boxes
        overlap = one(expected)
        for i in eachindex(u)
            overlap *= clamp(C.m[i] * u[i] - box[i], 0, 1)
        end
        expected += weight * overlap
    end
    @test cdf(C, u) ≈ expected
    prove_dispatch_route!(:cdf, C, :exact_box_overlap)
end

@testset verbose=true "specialized bivariate log-densities agree with CDF derivatives" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Distributions._logpdf,
                        Tuple{typeof(C),Vector{Float64}}),
        (case, C) -> is_absolutely_continuous(C) && !(C isa LiouvilleCopula),
    )
    u = [0.53, 0.67]
    h = 2e-5
    for (; case, C, method) in routes
        @testset "$(case.name)" begin
            test_progress("equivalence", "logpdf", case.name)
            expected = (
                cdf(C, u .+ (h, h)) - cdf(C, u .+ (h, -h)) -
                cdf(C, u .+ (-h, h)) + cdf(C, u .- (h, h))
            ) / (4h^2)
            @test isapprox(pdf(C, u), expected; atol=8e-4, rtol=8e-4)
            @test logpdf(C, u) ≈ log(pdf(C, u))
        end
        prove_dispatch_route!(:logpdf, C, :cdf_mixed_derivative)
    end
    @test !isempty(routes)
end

@testset "singular and mixed CDF routes satisfy mass identities" begin
    seen = Set{Any}()
    split = 0.46
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        is_absolutely_continuous(C) && continue
        key = dispatch_route_key(:cdf, C)
        key in seen && continue
        push!(seen, key)
        d = length(C)
        for i in 1:d
            margin_point = ones(d)
            margin_point[i] = 0.37
            @test cdf(C, margin_point) ≈ 0.37 atol=case.margin_atol
        end
        lower = collect(range(0.12, 0.18; length=d))
        upper = collect(range(0.78, 0.84; length=d))
        whole = Copulas.measure(C, lower, upper)
        left_upper = copy(upper)
        left_upper[1] = split
        right_lower = copy(lower)
        right_lower[1] = split
        @test whole ≈
              Copulas.measure(C, lower, left_upper) +
              Copulas.measure(C, right_lower, upper)
        prove_dispatch_route!(:cdf, C, :singular_mass_identity)
    end
    @test !isempty(seen)
end

@testset verbose=true "specialized dependence measures agree with generic definitions" begin
    # Entropy and Gini's gamma use substantially more expensive multidimensional
    # expectations and are covered by their independent identities in
    # correctness/. Kendall's generic definition is stochastic, so singular
    # Kendall formulas keep their exact family identities instead of a noisy,
    # repeated 10_000-observation comparison here. The CDF-only definitions of
    # rho, beta and tail dependence remain valid for singular and mixed laws.
    @testset verbose=true "$(nameof(SCALAR_DEPENDENCE_MEASURES[index]))" for index in (1, 2, 3, 6, 7)
        measure = SCALAR_DEPENDENCE_MEASURES[index]
        routes = _unique_bivariate_routes(
            (_, C) -> which(measure, Tuple{typeof(C)}),
            (_, C) -> measure === Copulas.τ ?
                is_absolutely_continuous(C) : true,
        )
        generic_method = which(measure, Tuple{Copulas.Copula{2}})
        for (; case, C, method) in routes
            if method === generic_method
                # The bivariate generic mechanism is proved independently by
                # PolynomialOracleCopula in correctness/mathematical.jl.
                @test dependence_route_key(measure, C) in
                      PROVEN_DEPENDENCE_ROUTES[measure]
                continue
            end
            @testset "$(case.name)" begin
                test_progress("equivalence", nameof(measure), case.name)
                if measure === Copulas.τ &&
                   (C isa GaussianCopula || C isa TCopula)
                    # Kendall's tau is invariant over the radial distribution
                    # of an elliptical copula.  At ρ = 1/2, the exact identity
                    # 2asin(ρ)/π = 1/3 validates both elliptical
                    # specializations without repeatedly evaluating their
                    # expensive numerical CDFs inside a cubature.
                    reference = C isa GaussianCopula ?
                        GaussianCopula{2}(0.5) :
                        TCopula{2}(C.df, [1.0 0.5; 0.5 1.0])
                    @test Copulas.τ(reference) ≈ 1 / 3 atol=2e-15
                elseif measure === Copulas.ρ && C isa GaussianCopula
                    # The bivariate Gaussian identity avoids nesting the
                    # numerical normal CDF inside the generic rho cubature.
                    reference = GaussianCopula{2}(0.5)
                    @test Copulas.ρ(reference) ≈ 6asin(0.25) / π atol=2e-15
                else
                    expected = measure === Copulas.τ ?
                        4 * HCubature.hcubature(u -> cdf(C, u) * pdf(C, u),
                                               zeros(2), ones(2); rtol=1e-5)[1] - 1 :
                        invoke(measure, Tuple{Copulas.Copula}, C)
                    @test isapprox(measure(C), expected; atol=3e-4, rtol=3e-4)
                end
            end
            prove_dependence_route!(measure, C)
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
    # Gamma and entropy use stochastic generic expectations. Their forwarding
    # dispatches are inventoried below; exact value equality is meaningful only
    # for the deterministic measures.
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λₗ, Copulas.λᵤ)
        @test measure(subset) == measure(parent)
    end
end

_spectral_matrix(tail::Copulas.DiscreteSpectralTail) = tail.B
_spectral_matrix(tail::Union{Copulas.BC2Tail,Copulas.MOTail}) = tail.spectral.B

function _spectral_curvature_tau(tail)
    # If A(t) = sum_k max(B[1,k]t, B[2,k](1-t)), its second derivative is
    # the discrete measure placing mass B[1,k] + B[2,k] at the corresponding
    # kink.  This is the distributional version of the defining EV Kendall
    # integral and, unlike a sample-concordance check, is exact and noiseless.
    B = _spectral_matrix(tail)
    total = zero(eltype(B))
    for k in axes(B, 2)
        mass = B[1, k] + B[2, k]
        iszero(mass) && continue
        kink = B[2, k] / mass
        total += mass * kink * (1 - kink) / Copulas.A(tail, kink)
    end
    return total
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.BC2Tail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.MOTail})
    # Classical competing-shocks identity. The public bivariate constructor
    # stores private shocks in subset order ([2], [1], [1,2]).
    λ₁, λ₂, λ₁₂ = C.tail.λ[2], C.tail.λ[1], C.tail.λ[3]
    a = λ₁ / (λ₁ + λ₁₂)
    b = λ₂ / (λ₂ + λ₁₂)
    return a * b / (a + b - a * b)
end

_singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.DiscreteSpectralTail}) =
    _spectral_curvature_tau(C.tail)

function _singular_tau_oracle(C::ExtremeValueCopula{2,<:Copulas.CuadrasAugeTail})
    # Its Pickands function has one kink at 1/2 with slope jump 2θ.
    kink = 0.5
    return 2C.tail.θ * kink * (1 - kink) / Copulas.A(C.tail, kink)
end

# This is the classical bivariate Raftery identity, independently obtained
# from its common-factor mixture representation.
_singular_tau_oracle(C::RafteryCopula{2}) = 2C.θ / (3 - C.θ)
_singular_tau_oracle(::MCopula{2}) = 1
_singular_tau_oracle(::WCopula{2}) = -1

@testset verbose=true "singular Kendall routes agree with deterministic identities" begin
    routes = _unique_bivariate_routes(
        (_, C) -> which(Copulas.τ, Tuple{typeof(C)}),
        (_, C) -> !is_absolutely_continuous(C),
    )
    generic_method = which(Copulas.τ, Tuple{Copulas.Copula{2}})
    compared = 0
    for route in routes
        (; case, C, method) = route
        method === generic_method && continue
        @testset "$(case.name)" begin
            test_progress("equivalence", "singular Kendall", case.name)
            expected = _singular_tau_oracle(C)
            @test Copulas.τ(C) ≈ expected atol=2e-12 rtol=2e-12
        end
        prove_dependence_route!(Copulas.τ, C)
        compared += 1
    end
    @test compared > 0
end

@testset "all gamma and entropy dispatches have an independent proof" begin
    parent = ClaytonCopula{2}(1.5)
    subset = subsetdims(parent, (2, 1))
    candidates = Any[]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
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
        for C in checked
            prove_dependence_route!(measure, C)
        end
    end
end

@testset "every pairwise dependence route reduces to bivariate margins" begin
    scalar = Dict(
        StatsBase.corkendall => Copulas.τ,
        StatsBase.corspearman => Copulas.ρ,
        Copulas.corblomqvist => Copulas.β,
        Copulas.corgini => Copulas.γ,
        Copulas.corentropy => Copulas.ι,
        Copulas.corlowertail => Copulas.λₗ,
        Copulas.coruppertail => Copulas.λᵤ,
    )
    for (pairwise, diagonal) in PAIRWISE_DEPENDENCE_MEASURES
        selected = Set((which(pairwise, Tuple{typeof(fixture.copula)}),
                        length(fixture.copula) == 2 ? :bivariate : :multivariate)
                       for fixture in ROUTING_COPULA_FIXTURES
                       if _dependence_is_defined(pairwise, fixture.copula))
        checked = Set{Any}()
        for fixture in ROUTING_COPULA_FIXTURES
            case, C = fixture.case, fixture.copula
            _dependence_is_defined(pairwise, C) || continue
            key = (which(pairwise, Tuple{typeof(C)}),
                   length(C) == 2 ? :bivariate : :multivariate)
            key in checked && continue
            # Generic gamma and entropy estimators sample internally. Reusing
            # the same RNG state makes this an exact forwarding test instead
            # of comparing two independent Monte Carlo estimates.
            seed = 0x51a7 + hash((pairwise, key))
            Random.seed!(seed)
            observed = pairwise(C)
            Random.seed!(seed)
            if C isa EmpiricalCopula &&
               pairwise in (StatsBase.corkendall, StatsBase.corspearman)
                expected = pairwise(transpose(C.u))
            else
                d = length(C)
                expected = Matrix{Float64}(I, d, d) .* diagonal
                for i in 1:d, j in 1:(i - 1)
                    value = scalar[pairwise](subsetdims(C, (i, j)))
                    expected[i, j] = expected[j, i] = value
                end
            end
            @test observed ≈ expected atol=1e-8
            push!(checked, key)
        end
        @test checked == selected
    end
end

@testset "multivariate Archimedean and Raftery dependence identities" begin
    # These closed forms are dimension-dependent dispatch routes and therefore
    # cannot be represented by the bivariate specialization comparison above.
    clayton = ClaytonCopula{3}(1.5)
    @test Copulas.τ(clayton) ≈ 3 / 7
    # The generator specialization is dimension invariant; its bivariate
    # value is independently checked against the generic integral above.
    @test Copulas.ρ(clayton) == Copulas.ρ(ClaytonCopula{2}(1.5))
    prove_dependence_route!(Copulas.τ, clayton)
    prove_dependence_route!(Copulas.ρ, clayton)

    raftery = RafteryCopula{3}(0.5)
    @test Copulas.τ(raftery) ≈ 0.4
    @test Copulas.ρ(raftery) ≈ 13 / 27
    prove_dependence_route!(Copulas.τ, raftery)
    prove_dependence_route!(Copulas.ρ, raftery)
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

@testset verbose=true "all distortion quantile specializations agree with generic inversion" begin
    generic_method = which(quantile, Tuple{Copulas.Distortion,Real})
    seen = Set{Method}()
    for (name, D) in CONDITIONAL_DISTRIBUTION_CASES
        conditional_measure_style(D) isa Copulas.AbsolutelyContinuousMeasure ||
            continue
        D isa Copulas.Distortion || continue
        method = which(quantile, Tuple{typeof(D),Float64})
        method === generic_method && continue
        method in seen && continue
        push!(seen, method)
        @testset "$name" begin
            test_progress("equivalence", "distortion quantile", name)
            generic = invoke(quantile, Tuple{Copulas.Distortion,Real}, D, 0.63)
            @test isapprox(quantile(D, 0.63), generic; atol=2e-8, rtol=2e-8)
        end
    end
    @test !isempty(seen)
end

@testset verbose=true "bivariate conditioning routes agree with CDF derivatives" begin
    seen = Set{Method}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 2 || continue
        is_absolutely_continuous(C) || continue
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),Tuple{Int},Tuple{Float64},Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            test_progress("equivalence", "bivariate conditioning", case.name)
            conditioned, target = 0.41, 0.63
            D = condition(C, 1, conditioned)
            if D isa Copulas.LiouvilleDistortion
                x = quantile(D.margin, 1 - target)
                expected_cdf = ccdf(D.conditional_margin, x)
                expected_pdf = pdf(D.conditional_margin, x) / pdf(D.margin, x)
            elseif C isa GaussianCopula
                ρ = C.Σ[1, 2]
                zⱼ = quantile(Normal(), conditioned)
                zᵢ = quantile(Normal(), target)
                z = (zᵢ - ρ * zⱼ) / sqrt(1 - ρ^2)
                expected_cdf = cdf(Normal(), z)
                expected_pdf = pdf(Normal(), z) / (sqrt(1 - ρ^2) * pdf(Normal(), zᵢ))
            else
                h = 2e-5
                expected_cdf = (cdf(C, [conditioned + h, target]) -
                                cdf(C, [conditioned - h, target])) / (2h)
                expected_pdf = (
                    cdf(C, [conditioned + h, target + h]) -
                    cdf(C, [conditioned + h, target - h]) -
                    cdf(C, [conditioned - h, target + h]) +
                    cdf(C, [conditioned - h, target - h])
                ) / (4h^2)
            end
            @test isapprox(cdf(D, target), expected_cdf;
                           atol=3e-5, rtol=3e-5)
            @test isapprox(pdf(D, target), expected_pdf;
                           atol=3e-4, rtol=3e-4)
        end
        prove_dispatch_route!(:conditioning, C, :cdf_derivative)
    end
    @test !isempty(seen)
end

function _finite_conditional_cdf(C, js, values, target_index, target; h=2e-4)
    Base.@nospecialize C js values
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

function _elliptical_conditional_cdf(C::GaussianCopula, js, values,
                                     target_index, target)
    J = collect(js)
    zJ = quantile.(Normal(), collect(values))
    β = C.Σ[J, J] \ C.Σ[J, target_index]
    μ = dot(C.Σ[target_index, J], C.Σ[J, J] \ zJ)
    σ² = 1 - dot(C.Σ[target_index, J], β)
    return cdf(Normal(), (quantile(Normal(), target) - μ) / sqrt(σ²))
end

function _elliptical_conditional_cdf(C::TCopula, js, values,
                                     target_index, target)
    J = collect(js)
    ν = C.df
    zJ = quantile.(TDist(ν), collect(values))
    solved = C.Σ[J, J] \ zJ
    β = C.Σ[J, J] \ C.Σ[J, target_index]
    μ = dot(C.Σ[target_index, J], solved)
    σ0² = 1 - dot(C.Σ[target_index, J], β)
    δ = dot(zJ, solved)
    νp = ν + length(J)
    σ = sqrt(σ0² * (ν + δ) / νp)
    return cdf(TDist(νp), (quantile(TDist(ν), target) - μ) / σ)
end

@testset verbose=true "multivariate conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Method}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        is_absolutely_continuous(C) || continue
        js = Tuple(1:(d - 1))
        values = ntuple(k -> 0.3 + 0.08k, d - 1)
        method = which(Copulas.DistortionFromCop,
            Tuple{typeof(C),typeof(js),typeof(values),Int})
        method in seen && continue
        push!(seen, method)

        @testset "$(case.name)" begin
            test_progress("equivalence", "multivariate conditioning", case.name)
            target_index = d
            target = 0.63
            D = condition(C, js, values)
            expected = if C isa Union{GaussianCopula,TCopula}
                _elliptical_conditional_cdf(C, js, values, target_index, target)
            elseif D isa Copulas.LiouvilleDistortion
                x = quantile(D.margin, 1 - target)
                ccdf(D.conditional_margin, x)
            else
                _finite_conditional_cdf(C, js, values, target_index, target)
            end
            @test isapprox(cdf(D, target), expected; atol=2e-3, rtol=2e-3)
        end
        prove_dispatch_route!(:conditioning, C,
                              :normalized_cdf_derivative)
    end
    @test !isempty(seen)
end

@testset "atomic conditioning routes satisfy generalized inversion" begin
    seen = Set{Any}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        is_absolutely_continuous(C) && continue
        # Point conditioning is not canonically defined away from the finite
        # support of an empirical copula. Its generic method is exercised and
        # proved by the Raftery representative below.
        C isa EmpiricalCopula && continue
        key = dispatch_route_key(:conditioning, C)
        key in seen && continue
        push!(seen, key)
        d = length(C)
        D = condition(C, Tuple(1:(d - 1)), ntuple(_ -> 0.4, d - 1))
        @testset "$(case.name)" begin
            for p in (0.2, 0.6, 0.85)
                q = quantile(D, p)
                @test cdf(D, q) >= p - 1e-10
            end
        end
        prove_dispatch_route!(:conditioning, C,
                              :generalized_quantile_identity)
    end
    @test !isempty(seen)
end

@testset "joint conditioning routes agree with normalized CDF derivatives" begin
    seen = Set{Any}()
    conditioned = 0.41
    h = 2e-5
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        d > 2 || continue
        key = dispatch_route_key(:conditional_joint, C)
        key in seen && continue
        push!(seen, key)

        H = condition(C, (1,), (conditioned,))
        targets = collect(range(0.53, 0.71; length=d - 1))
        conditional_scale = [cdf(H.m[i], targets[i]) for i in 1:(d - 1)]
        if C isa Union{GaussianCopula,TCopula}
            J, I = [1], collect(2:d)
            Σcond = C.Σ[I, I] - C.Σ[I, J] * (C.Σ[J, J] \ C.Σ[J, I])
            σ = sqrt.(diag(Σcond))
            expected_R = Σcond ./ (σ * σ')
            @test H.C.Σ ≈ expected_R atol=2e-12 rtol=2e-12
        elseif C isa LiouvilleCopula
            @test H.C isa LiouvilleCopula{d - 1}
            @test H.C.α == ntuple(i -> C.α[i + 1], d - 1)
        else
            upper = vcat(conditioned + h, targets)
            lower = vcat(conditioned - h, targets)
            numerator = (cdf(C, upper) - cdf(C, lower)) / (2h)
            normalizer = (cdf(C, vcat(conditioned + h, ones(d - 1))) -
                          cdf(C, vcat(conditioned - h, ones(d - 1)))) / (2h)
            tolerance = is_absolutely_continuous(C) ? 5e-4 : 3e-3
            @test isapprox(cdf(H.C, conditional_scale), numerator / normalizer;
                           atol=tolerance, rtol=tolerance)
        end
        prove_dispatch_route!(:conditional_joint, C,
                              :normalized_joint_cdf_derivative)
    end
    @test !isempty(seen)
end

@testset "subsetting routes preserve parent margins" begin
    seen = Set{Any}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        dims = d == 2 ? (2, 1) : (1, d)
        key = dispatch_route_key(:subsetting, C)
        key in seen && continue
        push!(seen, key)
        S = subsetdims(C, dims)
        u = [0.37, 0.68]
        parent_point = ones(d)
        parent_point[collect(dims)] .= u
        @test cdf(S, u) ≈ cdf(C, parent_point)
        prove_dispatch_route!(:subsetting, C, :parent_margin_identity)
    end
    @test !isempty(seen)
end

@testset "rectangle-measure routes equal CDF inclusion-exclusion" begin
    seen = Set{Any}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        key = dispatch_route_key(:measure, C)
        key in seen && continue
        push!(seen, key)
        d = length(C)
        lower = collect(range(0.13, 0.19; length=d))
        upper = collect(range(0.71, 0.79; length=d))
        expected = 0.0
        for mask in Iterators.product(ntuple(_ -> (false, true), d)...)
            point = [mask[i] ? lower[i] : upper[i] for i in 1:d]
            expected += (-1)^count(identity, mask) * cdf(C, point)
        end
        @test Copulas.measure(C, lower, upper) ≈ expected atol=1e-10
        prove_dispatch_route!(:measure, C, :cdf_inclusion_exclusion)
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
        IndependentCopula{3}(),
    )
    generic_method = which(Copulas.rosenblatt,
        Tuple{Copulas.Copula{3},Matrix{Float64}})
    candidates = Any[checked[3]]
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        length(C) == 3 && is_absolutely_continuous(C) && push!(candidates, C)
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

@testset "every Rosenblatt route equals sequential conditioning" begin
    seen_forward = Set{Any}()
    seen_inverse = Set{Any}()
    for fixture in ROUTING_COPULA_FIXTURES
        case, C = fixture.case, fixture.copula
        d = length(C)
        u = collect(range(0.31, 0.73; length=d))
        forward_key = dispatch_route_key(:rosenblatt, C)
        inverse_key = dispatch_route_key(:inverse_rosenblatt, C)
        forward_done = forward_key in seen_forward
        inverse_done = isnothing(inverse_key) || inverse_key in seen_inverse
        forward_done && inverse_done && continue

        R = rosenblatt(C, u)
        expected = similar(R)
        expected[1] = u[1]
        for i in 2:d
            js = Tuple(1:(i - 1))
            values = Tuple(u[1:(i - 1)])
            expected[i] = cdf(Copulas.DistortionFromCop(C, js, values, i),
                              u[i])
        end
        @test R ≈ expected atol=2e-6 rtol=2e-6
        prove_dispatch_route!(:rosenblatt, C, :sequential_conditioning)
        push!(seen_forward, forward_key)
        if !isnothing(inverse_key)
            @test inverse_rosenblatt(C, R) ≈ u atol=2e-6 rtol=2e-6
            prove_dispatch_route!(:inverse_rosenblatt, C,
                                  :sequential_conditioning_inverse)
            push!(seen_inverse, inverse_key)
        end
    end
    @test !isempty(seen_forward)
    @test !isempty(seen_inverse)
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
    @test 0 < cdf(D, 0.63) < 1
    @test pdf(D, 0.63) > 0
    u = [0.21, 0.53, 0.74]
    @test all(x -> 0 < x < 1, rosenblatt(C, u))
end

@testset "conditioning preserves non-Float64 paths" begin
    C = ClaytonCopula{4}(2.0)
    xf = [0.3, 0.5, 0.4, 0.6]
    xb = big.(xf)

    df = condition(C, (1, 3, 4), Tuple(xf[[1, 3, 4]]))
    db = condition(C, (1, 3, 4), Tuple(xb[[1, 3, 4]]))
    @test db.den isa BigFloat
    @test eltype(db.uⱼₛ) === BigFloat
    cdf_db = cdf(db, xb[2])
    @test cdf_db isa BigFloat
    @test Float64(cdf_db) ≈ cdf(df, xf[2]) atol=1e-9

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

@testset "bivariate EV matrix and scalar representations agree" begin
    point = [0.4, 0.7]
    pairs = (
        (HuslerReissCopula{2}([0.0 1.0; 1.0 0.0]),
         HuslerReissCopula{2}(2.0), 4101),
        (tEVCopula{2}(4.0, [1.0 0.3; 0.3 1.0]),
         tEVCopula{2}(4.0, 0.3), 4102),
    )
    for (matrix_model, scalar_model, seed) in pairs
        @test cdf(matrix_model, point) ≈ cdf(scalar_model, point)
        @test pdf(matrix_model, point) ≈ pdf(scalar_model, point)
        for measure in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)
            @test measure(matrix_model) ≈ measure(scalar_model)
        end
        @test rand(Random.Xoshiro(seed), matrix_model, 16) ==
              rand(Random.Xoshiro(seed), scalar_model, 16)
    end
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
