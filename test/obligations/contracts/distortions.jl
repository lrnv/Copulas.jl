# Contract obligation: exercises the common univariate conditional API once
# for every distortion implementation reached through the public `condition`
# entry point. Family formulas remain in focused regression tests.
const DISTORTION_CASES = (
    ("identity", condition(IndependentCopula{2}(), 1, 0.4), :continuous),
    ("upper Frechet atom", condition(MCopula{2}(), 1, 0.4), :atomic),
    ("lower Frechet atom", condition(WCopula{2}(), 1, 0.4), :atomic),
    ("Gaussian", condition(GaussianCopula{2}(0.4), 1, 0.4), :continuous),
    ("Student", condition(TCopula{2}(4, [1.0 0.4; 0.4 1.0]), 1, 0.4), :continuous),
    ("Archimedean", condition(ClaytonCopula{2}(1.5), 1, 0.4), :continuous),
    ("extreme value", condition(GalambosCopula{2}(1.0), 1, 0.4), :continuous),
    ("Archimax", condition(BB4Copula{2}(1.0, 1.0), 1, 0.4), :continuous),
    ("FGM", condition(FGMCopula{2}(0.5), 1, 0.4), :continuous),
    ("Plackett", condition(PlackettCopula{2}(2.0), 1, 0.4), :continuous),
    ("histogram", condition(CheckerboardCopula{2}(_FIXTURE_DATA; m=2), 1, 0.4), :continuous),
    ("Bernstein", condition(BernsteinCopula{2}(GaussianCopula{2}(0.3); m=3), 1, 0.4), :continuous),
    ("generic", condition(RafteryCopula{2}(0.5), 1, 0.4), :continuous),
    ("Liouville", condition(LiouvilleCopula{2}(
        WilliamsonGenerator(Dirac(1.0), 3.0), (0.6, 1.1)), 1, 0.4), :continuous),
    ("nested Archimedean", condition(NestedArchimedeanCopula{4}(
        Copulas.ClaytonGenerator(1.0); leaves=[1, 2],
        children=[ClaytonCopula{2}(2.0)]), (1, 2, 3), (0.3, 0.4, 0.5)), :continuous),
    ("survival flip", condition(SurvivalCopula{2}(ClaytonCopula{2}(1.5), (2,)), 1, 0.4), :continuous),
)

function test_distortion_contract(D, kind)
    Base.@nospecialize D
    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test cdf(D, 0.0) == 0
    @test cdf(D, 1.0) == 1

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

    kind === :continuous || return
    for u in grid
        density = pdf(D, u)
        @test density >= 0
        @test iszero(density) ? logpdf(D, u) == -Inf :
              logpdf(D, u) ≈ log(density)
    end
end

@testset "distortion implementations satisfy the conditional contract" begin
    types = Set{Any}()
    for (name, D, kind) in DISTORTION_CASES
        @testset "$name ($(nameof(typeof(D))))" begin
            test_distortion_contract(D, kind)
            push!(types, typeof(D))
        end
    end
    @test length(types) == length(DISTORTION_CASES)
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
