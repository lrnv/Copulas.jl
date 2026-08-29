# Public-API contract: checks SklarDist construction and the adopted
# Distributions, conditioning, Rosenblatt, sampling, and matrix interfaces.
@testset "SklarDist public contract" begin
    # Use an analytic CDF here: this is an identity of the Sklar adapter, not
    # a test of the numerical multivariate-normal integrator (covered in the
    # elliptical tests). Calling the latter twice made this exact identity
    # depend on integration noise across Julia versions.
    C = ClaytonCopula{2}(1.0)
    D = SklarDist(C, (Normal(), Exponential()))
    x = [0.1, 1.2]
    @test length(D) == 2
    @test_throws DimensionMismatch SklarDist(C, (Normal(),))
    @test params(D) isa NamedTuple
    @test StatsBase.dof(D) == 4
    @test 0 <= cdf(D, x) <= 1
    @test logcdf(D, x) ≈ log(cdf(D, x))
    @test pdf(D, x) >= 0
    @test logpdf(D, x) ≈ log(pdf(D, x))
    X = rand(StableRNG(31), D, 4)
    @test size(X) == (2, 4)
    @test eltype(X) == eltype(D)
    @test cdf(D, X) ≈ [cdf(D, column) for column in eachcol(X)] atol=2e-4
    @test logcdf(D, X) ≈ log.(cdf(D, X)) atol=5e-4
    @test pdf(D, X) == [pdf(D, column) for column in eachcol(X)]
    @test logpdf(D, X) ≈ log.(pdf(D, X))
    @test_throws ArgumentError cdf(D, zeros(3))
    @test_throws ArgumentError cdf(D, zeros(3, 1))
    @test_throws DimensionMismatch logpdf(D, zeros(3))
    @test_throws ArgumentError logpdf(D, zeros(3, 1))
    @test loglikelihood(D, X) isa Real

    S = subsetdims(D, (2, 1))
    @test length(S) == 2
    @test S.C == subsetdims(C, (2, 1))
    @test subsetdims(D, (1,)) == D.m[1]
    conditional = condition(D, 1, x[1])
    @test minimum(conditional) == 0
    @test maximum(conditional) == Inf
    @test cdf(conditional, quantile(conditional, 0.5)) >= 0.5 - sqrt(eps())
    @test pdf(conditional, 1.0) >= 0
    @test logpdf(conditional, 1.0) ≈ log(pdf(conditional, 1.0))
    @test rand(StableRNG(32), conditional) >= 0

    R = rosenblatt(D, X)
    @test size(R) == size(X)
    @test inverse_rosenblatt(D, R) ≈ X atol=2e-5 rtol=2e-5
    @test rosenblatt(D, x) ≈ vec(rosenblatt(D, reshape(x, :, 1)))
    @test inverse_rosenblatt(D, rosenblatt(D, x)) ≈ x atol=2e-5 rtol=2e-5

    clayton_joint = SklarDist(ClaytonCopula{2}(1.0), (Normal(), Exponential()))
    @test StatsBase.dof(clayton_joint) == 4

    D3 = SklarDist(GaussianCopula{3}(0.3), (Normal(), Exponential(), Gamma(2, 1)))
    x3 = [0.1, 1.2, 0.8]
    joint = condition(D3, 1, x3[1])
    @test length(joint) == 2
    @test 0 <= cdf(joint, x3[2:3]) <= 1
    @test pdf(joint, x3[2:3]) >= 0
    @test size(rand(StableRNG(33), joint, 2)) == (2, 2)
    @test length(subsetdims(D3, (3, 1))) == 2

    independent = SklarDist(
        IndependentCopula{3}(), (Normal(), Exponential(), LogNormal()))
    independent_conditional = condition(independent, 2, 0.7)
    independent_subset = subsetdims(independent, (1, 3))
    @test independent_conditional.C == independent_subset.C
    @test independent_conditional.m == independent_subset.m

    uniform_conditional = condition(IndependentCopula{2}(), 1, 0.3)
    @test cdf(uniform_conditional, 0.37) == 0.37
    original_scale = condition(
        SklarDist(IndependentCopula{2}(), (Normal(), Exponential())),
        1, 0.0)
    for t in (-1.0, 0.0, 1.2)
        @test cdf(original_scale, t) ≈ cdf(Exponential(), t)
    end

    # The Sklar wrapper has one implementation route per public operation;
    # variation in copula, dimension and margins is delegated to components
    # whose own routes are proved independently.
    compositions = (D, D3, independent)
    route_functions = (
        cdf = S -> which(Distributions.cdf,
                         Tuple{typeof(S),Vector{Float64}}),
        logpdf = S -> which(Distributions._logpdf,
                            Tuple{typeof(S),Vector{Float64}}),
        sampling = S -> which(Distributions._rand!,
            Tuple{typeof(StableRNG(34)),typeof(S),Matrix{Float64}}),
        subsetting = S -> which(Copulas.subsetdims,
                                Tuple{typeof(S),Tuple{Int,Int}}),
        conditioning = S -> which(Copulas.condition,
                                   Tuple{typeof(S),Int,Float64}),
        rosenblatt = S -> which(Copulas.rosenblatt,
                                Tuple{typeof(S),Matrix{Float64}}),
        inverse_rosenblatt = S -> which(Copulas.inverse_rosenblatt,
            Tuple{typeof(S),Matrix{Float64}}),
    )
    for route in values(route_functions)
        @test length(Set(route(S) for S in compositions)) == 1
    end
end
