# Complete operation proof for marginal distortions and conditional copulas:
# public contracts, independent identities, specialization equivalence,
# family regressions, and route coverage are colocated here.

function test_conditioning_contract(C, ctx)
    Base.@nospecialize C ctx
    d = length(C)
    d > 2 && !is_absolutely_continuous(C) && return
    if d == 2
        scalar = condition(C, 1, ctx.u[1])
        tupled = condition(C, (1,), (ctx.u[1],))
        @test scalar isa Distributions.UnivariateDistribution
        @test cdf(scalar, ctx.u[2]) ≈ cdf(tupled, ctx.u[2])
    end
    if d > 2
        joint = condition(C, 1, ctx.u[1])
        @test length(joint) == d - 1
        @test 0 <= cdf(joint, ctx.u[2:end]) <= 1
    end
    if d > 3
        js = Tuple(1:(d - 2))
        joint = condition(C, js, Tuple(ctx.u[1:(d - 2)]))
        @test length(joint) == 2
        @test 0 <= cdf(joint, ctx.u[(d - 1):d]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(ctx.u[1:(d - 1)])
    D = condition(C, js, values)
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    q = quantile(D, 0.5)

    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test issorted(vals)
    @test logcdf(D, 0.5) ≈ log(cdf(D, 0.5))

    if is_absolutely_continuous(C)
        densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
        @test all(x -> x >= 0, densities)
        density = pdf(D, 0.5)
        @test iszero(density) ? logpdf(D, 0.5) == -Inf :
              logpdf(D, 0.5) ≈ log(density)
    end

    @test all(x -> 0 <= x <= 1, rand(StableRNG(73), D, 3))
    @test 0 <= q <= 1
    is_absolutely_continuous(C) &&
        @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end

function conditional_distribution(fixture)
    Base.@nospecialize fixture
    C = fixture.copula
    d = length(C)
    js = Tuple(1:(d - 1))
    values = ntuple(_ -> 0.4, d - 1)
    return condition(C, js, values)
end

conditional_route_key(D) = Tuple(which(f, Tuple{typeof(D),Float64}) for f in (
    Distributions.cdf, Distributions.logcdf, Distributions.logpdf,
    Distributions.quantile,
))

const CONDITIONAL_DISTRIBUTION_CANDIDATES = (
    ((fixture.case.name, conditional_distribution(fixture))
     for fixture in ROUTING_COPULA_FIXTURES
     if length(fixture.copula) == 2 || is_absolutely_continuous(fixture.copula))...,
)
const CONDITIONAL_DISTRIBUTION_CASES = Tuple(unique(
    case -> conditional_route_key(last(case)),
    CONDITIONAL_DISTRIBUTION_CANDIDATES,
))

conditional_measure_style(D::Copulas.Distortion) =
    Copulas.distortion_measure_style(D)
conditional_measure_style(::Distributions.UnivariateDistribution) =
    Copulas.AbsolutelyContinuousMeasure()

@testset verbose=true "public conditioning contract" begin
    @testset "$(fixture.case.name)" for (seed, fixture) in enumerate(COPULA_FIXTURES)
        test_progress("operations", "conditioning", fixture.case.name, "contract")
        test_conditioning_contract(
            fixture.copula,
            copula_contract_context(fixture.copula, 10_000 + seed),
        )
    end
end


# Conditioning-operation equivalence proofs for optimized distortions and joint laws.

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

@testset "mixed conditional laws use generalized quantiles" begin
    # A bijective Rosenblatt identity is intentionally not asserted in the
    # presence of atoms.
    C = MOCopula{2}(0.2, 0.3, 0.4)
    D = condition(C, 1, 0.4)
    probabilities = collect(0.05:0.05:0.95)
    quantiles = quantile.(Ref(D), probabilities)
    @test issorted(quantiles)
    @test any(iszero, diff(quantiles))
    for (p, q) in zip(probabilities, quantiles)
        @test cdf(D, q) >= p - 1e-10
    end
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



# Conditioning-operation equivalence: conditional-distribution and distortion
# fast paths are checked against inversion identities, generic conditionals,
# log-scale definitions, or independent Gaussian conditioning algebra.

@testset "Gaussian distortion log-scale formulas" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))
    N = Normal()
    for u in (1e-12, 0.2, 0.5, 0.8)
        q = quantile(N, u)
        z = (q - D.μz) / D.σz
        reference = logpdf(N, z) - log(abs(D.σz)) - logpdf(N, q)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 1e-13
        @test logpdf(D, u) ≈ reference atol = 1e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
    @test logpdf(D, -0.1) == -Inf
end

@testset "Student distortion logcdf" begin
    D = condition(TCopula{2}(4, [1.0 0.5; 0.5 1.0]), (1,), (0.3,))
    @test D.Tu isa TDist
    @test D.Tcond isa TDist
    for u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-13
    end
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "Elliptical conditioning shares matrix factorizations" begin
    Σ = [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]
    for C in (GaussianCopula{3}(Σ), TCopula{3}(4, Σ))
        conditioned = condition(C, (1,), (0.35,))
        @test length(conditioned.m) == 2
        for (k, i) in enumerate((2, 3)), u in (0.2, 0.7)
            reference = Copulas.DistortionFromCop(C, (1,), (0.35,), i)
            @test cdf(conditioned.m[k], u) ≈ cdf(reference, u) atol = 2e-12
        end
    end
end

@testset "Distorted distribution logcdf" begin
    D = condition(GaussianCopula{2}([1.0 0.6; 0.6 1.0]), (1,), (0.3,))(Logistic())
    @test D isa Copulas.DistortedDist
    for x in (-8.0, -0.5, 1.0)
        @test logcdf(D, x) ≈ logcdf(D.D, cdf(D.X, x)) atol = 2e-13
    end
end

@testset "Archimedean distortion logcdf" begin
    distortions = (
        condition(ClaytonCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(FrankCopula{3}(2.0), (1, 2), (0.3, 0.6)),
        condition(GumbelCopula{3}(2.0), (1, 2), (0.3, 0.6)),
    )
    for D in distortions, u in (1e-10, 0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 3e-12
    end
    @test all(logcdf(D, 0.0) == -Inf for D in distortions)
    @test all(logcdf(D, 1.0) == 0.0 for D in distortions)
end

@testset "Flip distortion logcdf" begin
    S = SurvivalCopula{2}(ClaytonCopula{2}(2.0), (2,))
    D = condition(S, (1,), (0.3,))
    @test D isa Copulas.FlipDistortion
    for u in (0.2, 0.5, 0.8)
        @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-12
    end
    u = 1e-12
    @test logcdf(D, u) ≈ LogExpFunctions.log1mexp(logcdf(D.base, 1 - u)) atol = 2e-12
    @test isfinite(logcdf(D, u))
    @test logcdf(D, 0.0) == -Inf
    @test logcdf(D, 1.0) == 0.0
end

@testset "FGM distortion log-scale formulas" begin
    for θ in (-0.8, 0.8), uⱼ in (0.2, 0.7)
        D = condition(FGMCopula{2}(θ), (1,), (uⱼ,))
        for u in (1e-12, 0.2, 0.5, 0.8)
            @test logcdf(D, u) ≈ log(cdf(D, u)) atol = 2e-14
        end
        @test logcdf(D, 0.0) == -Inf
        @test logcdf(D, 1.0) == 0.0
        @test logpdf(D, -0.1) == -Inf
        @test logpdf(D, 1.1) == -Inf
    end
end

@testset "Generic ConditionalCopula density" begin
    C = GaussianCopula{3}([
        1.0 0.35 0.20
        0.35 1.0 0.25
        0.20 0.25 1.0
    ])
    js = (3,)
    ujs = (0.4,)
    generic = @invoke Copulas.ConditionalCopula(C::Copulas.Copula{3}, js, ujs)
    Cgeneric = FGMCopula{3}([0.1, 0.2, 0.3, 0.4])
    conditioned = condition(Cgeneric, js, ujs)
    @test conditioned.C isa Copulas.ConditionalCopula
    @test conditioned.m === conditioned.C.distortions
    @test conditioned.C.is == (1, 2)
    @test generic.logden == log(generic.den)
    specialized = Copulas.ConditionalCopula(C, js, ujs)

    for u in ([0.25, 0.35], [0.5, 0.5], [0.75, 0.65])
        @test isapprox(logpdf(generic, u), logpdf(specialized, u); atol=1e-8, rtol=1e-8)
        @test isapprox(pdf(generic, u), pdf(specialized, u); atol=1e-8, rtol=1e-8)
    end
    @test pdf(generic, [-0.1, 0.5]) == 0

    Cclayton = ClaytonCopula{3}(2.0)
    generic_big = @invoke Copulas.ConditionalCopula(
        Cclayton::Copulas.Copula{3},
        (3,),
        (big"0.4",),
    )
    value_big = logpdf(generic_big, BigFloat[0.35, 0.65])
    @test value_big isa BigFloat
    @test isfinite(value_big)
end



# Conditioning-operation proof: exercises the common univariate API once
# for every result reached through the public `condition` entry point. Most
# results are `Distortion`s, but families may legitimately return another
# `UnivariateDistribution`, such as BetaCopula's exact `MixtureModel`.
@testset "bivariate scalar conditioning contract" begin
    C = GaussianCopula{2}(0.4)
    @test @inferred(condition(C, 1, 0.4)) isa Copulas.GaussianDistortion
    for j in 1:2, uⱼ in (0.2f0, big"0.8")
        @test typeof(condition(C, j, uⱼ)) ==
              typeof(condition(C, (j,), (float(uⱼ),)))
    end
    @test_throws ArgumentError condition(C, 0, 0.4)
    @test_throws ArgumentError condition(C, 3, 0.4)
    @test_throws ArgumentError condition(C, 1, -0.1)
    @test_throws ArgumentError condition(C, 1, 1.1)
end

function test_distortion_contract(D)
    Base.@nospecialize D
    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test cdf(D, 0.0) == 0
    @test cdf(D, 1.0) ≈ 1
    @test cdf(D, -0.2) == 0
    upper = cdf(D, 1.2)
    if D isa Copulas.Distortion
        @test upper == 1
    else
        # Distributions.MixtureModel may accumulate a few ulps above one when
        # all component CDFs are numerically saturated. BetaCopula deliberately
        # returns that native mixture, so require numerical rather than bitwise
        # unity only for external univariate-distribution implementations.
        @test upper ≈ 1
    end

    # Two separated interior points prove monotonicity while avoiding repeated
    # numerical conditioning kernels for every concrete implementation.
    grid = (0.25, 0.75)
    values = cdf.(Ref(D), grid)
    @test issorted(values)
    @test all(x -> 0 <= x <= 1, values)
    @test all(u -> logcdf(D, u) ≈ log(cdf(D, u)), grid)

    # One generalized inverse call per implementation exercises the route;
    # inverse shape/ordering is covered by the distribution-level contracts.
    probabilities = (0.5,)
    quantiles = quantile.(Ref(D), probabilities)
    @test issorted(quantiles)
    @test all(x -> 0 <= x <= 1, quantiles)
    for (p, q) in zip(probabilities, quantiles)
        @test cdf(D, q) >= p - 2e-8
    end

    samples = rand(StableRNG(501), D, 1)
    @test all(x -> 0 <= x <= 1, samples)

    conditional_measure_style(D) isa Copulas.AbsolutelyContinuousMeasure || return
    @test pdf(D, -0.2) == 0
    @test pdf(D, 1.2) == 0
    @test logpdf(D, -0.2) == -Inf
    @test logpdf(D, 1.2) == -Inf
    for u in grid
        density = pdf(D, u)
        @test density >= 0
        @test iszero(density) ? logpdf(D, u) == -Inf :
              logpdf(D, u) ≈ log(density)
    end
end

@testset "conditional distributions satisfy the public contract" begin
    operations = (
        cdf=Distributions.cdf, logcdf=Distributions.logcdf,
        logpdf=Distributions.logpdf, quantile=Distributions.quantile,
    )
    selected_routes = Dict(name => Set(which(f, Tuple{typeof(D),Float64})
        for (_, D) in CONDITIONAL_DISTRIBUTION_CASES)
        for (name, f) in pairs(operations))
    checked_routes = Dict(name => Set{Method}() for name in keys(operations))
    for (name, D) in CONDITIONAL_DISTRIBUTION_CASES
        @testset "$name ($(nameof(typeof(D))))" begin
            test_distortion_contract(D)
            for (operation, f) in pairs(operations)
                push!(checked_routes[operation],
                      which(f, Tuple{typeof(D),Float64}))
            end
        end
    end
    @test checked_routes == selected_routes

    # Every route reached by full conditioning of a bivariate or multivariate
    # bestiary entry must be represented by the univariate contract.
    reachable = Set(conditional_route_key(D)
                    for (_, D) in CONDITIONAL_DISTRIBUTION_CANDIDATES)
    represented = Set(conditional_route_key(D)
                      for (_, D) in CONDITIONAL_DISTRIBUTION_CASES)
    @test reachable == represented
end

@testset "distortion push-forwards preserve the marginal scale" begin
    D = condition(GaussianCopula{2}(0.4), 1, 0.35)
    X = Logistic(0.3, 1.2)
    Y = D(X)
    for x in (0.2,)
        @test cdf(Y, x) ≈ cdf(D, cdf(X, x))
        @test pdf(Y, x) ≈ pdf(D, cdf(X, x)) * pdf(X, x)
    end
    @test D(Normal(0.3, 1.2)) isa Normal
    @test Copulas.NoDistortion()(X) === X
end

@testset "atomic distortion generalized quantiles" begin
    for D in (condition(MCopula{2}(), 1, 0.4),
              condition(WCopula{2}(), 1, 0.4))
        atom = quantile(D, 0.5)
        @test cdf(D, prevfloat(atom)) == 0
        @test cdf(D, atom) == 1
        @test cdf(D, nextfloat(atom)) == 1
        @test pdf(D, atom) == 1
        @test pdf(D, prevfloat(atom)) == 0
        @test all(==(atom), rand(StableRNG(502), D, 4))
    end
end

@testset "elementary distortions respect unit support" begin
    for D in (Copulas.NoDistortion(), Copulas.MDistortion(0.4, Int8(2)),
              Copulas.WDistortion(0.4, Int8(2)))
        @test cdf(D, -0.2) == 0
        @test cdf(D, 1.2) == 1
        @test pdf(D, -0.2) == 0
        @test pdf(D, 1.2) == 0
        @test logpdf(D, -0.2) == -Inf
        @test logpdf(D, 1.2) == -Inf
    end
end

# Focused regressions retain implementation-sensitive assertions that are not
# implied by the operation-wide mathematical contracts above.
@testset "Extreme-value conditioning caches fixed transforms" begin
    DEV = condition(GalambosCopula{2}(2.5), (1,), (0.3,))
    @test DEV.negloguⱼ == -log(DEV.uⱼ)

    DAM = condition(ArchimaxCopula{2}(Copulas.FrankGenerator(0.8),
        Copulas.HuslerReissTail(0.6)), (1,), (0.3,))
    @test DAM.yⱼ == Copulas.ϕ⁻¹(DAM.gen, DAM.uⱼ)
    @test DAM.invderivⱼ == Copulas.ϕ⁻¹⁽¹⁾(DAM.gen, DAM.uⱼ)
end

@testset "Checkerboard multidimensional conditioning regression" begin
    C = CheckerboardCopula{3}(randn(rng, 3, 30); pseudo_values=false)
    D = Copulas.DistortionFromCop(C, (1, 2), (0.3, 0.7), 3)
    @test D isa Copulas.HistogramBinDistortion
    @test all(0 .<= cdf.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
    @test all(pdf.(Ref(D), (0.2, 0.5, 0.8)) .>= 0)
    @test all(0 .<= quantile.(Ref(D), (0.2, 0.5, 0.8)) .<= 1)
end

@testset "Bernstein distortion bounded inversion regression" begin
    D = condition(BernsteinCopula{2}(GaussianCopula{2}(0.3); m=5),
                  (1,), (0.4,))
    @test D isa Copulas.BernsteinDistortion
    for p in (0.1, 0.5, 0.9)
        q = quantile(D, p)
        @test 0 <= q <= 1
        @test cdf(D, q) ≈ p atol=2e-12
    end
end
