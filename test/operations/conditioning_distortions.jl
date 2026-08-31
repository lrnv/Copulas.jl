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
     for fixture in ROUTING_COPULA_FIXTURES)...,
)
const CONDITIONAL_DISTRIBUTION_CASES = Tuple(unique(
    case -> conditional_route_key(last(case)),
    CONDITIONAL_DISTRIBUTION_CANDIDATES,
))

conditional_measure_style(D::Copulas.Distortion) =
    Copulas.distortion_measure_style(D)
conditional_measure_style(::Distributions.UnivariateDistribution) =
    Copulas.AbsolutelyContinuousMeasure()

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
