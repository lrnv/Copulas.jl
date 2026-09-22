# Public-API proof: checks SklarDist construction and the adopted
# Distributions, conditioning, Rosenblatt, sampling, and matrix interfaces.
@testset "SklarDist public contract" begin
    # Use an analytic CDF here: this is an identity of the Sklar adapter, not
    # a test of the numerical multivariate-normal integrator (covered in the
    # elliptical tests). Calling the latter twice made this exact identity
    # depend on integration noise across Julia versions.
    C = ClaytonCopula{2}(1.0)
    D = SklarDist(C, (Normal(), Exponential()))
    D_from_vector = SklarDist(C, [Normal(), Exponential()])
    @test D_from_vector.m isa Tuple
    @test typeof(D_from_vector) === typeof(D)
    x = [0.1, 1.2]
    @test length(D) == 2
    @test_throws DimensionMismatch SklarDist(C, (Normal(),))
    @test params(D) isa Tuple
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
    @test_throws DimensionMismatch cdf(D, zeros(3))
    @test_throws DimensionMismatch cdf(D, zeros(3, 1))
    @test_throws DimensionMismatch logpdf(D, zeros(3))
    @test_throws DimensionMismatch logpdf(D, zeros(3, 1))
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
    @test StatsBase.dof(D3) == 9
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

@testset "Sklar work buffers promote all numeric inputs" begin
    S = SklarDist(IndependentCopula{2}(), (Normal(), Normal()))
    @test cdf(S, [0, 0]) ≈ 0.25

    Smixed = SklarDist(
        IndependentCopula{2}(),
        (Normal(0f0, 1f0), Normal(0.0, 1.0)),
    )
    @test cdf(Smixed, Float32[0, 0]) isa Float64
    @test logpdf(Smixed, Float32[0, 0]) isa Float64

    integer_data = [-2 -1 0 1 2; 2 1 0 -1 -2]
    Sinteger = fit(
        SklarDist{typeof(S.C),Tuple{Normal,Normal}},
        integer_data,
    )
    @test Sinteger isa SklarDist
    @test all(margin -> margin isa Normal, Sinteger.m)

    Sbig = SklarDist(
        IndependentCopula{2}(),
        (Normal(big"0", big"1"), Normal(big"0", big"1")),
    )
    xbig = BigFloat[0, 0]
    @test cdf(Sbig, xbig) isa BigFloat
    @test logpdf(Sbig, xbig) isa BigFloat
end

@testset "Sklar likelihood with atoms is a probability mass" begin
    # Bernoulli × Bernoulli under FGM: P(0, 0) = C(1/2, 1/2), not 1/4 · c(1/2, 1/2).
    S = SklarDist(FGMCopula{2}(1.0), (Bernoulli(0.5), Bernoulli(0.5)))
    cells = [pdf(S, [i, j]) for i in 0:1, j in 0:1]
    @test cells[1, 1] ≈ cdf(FGMCopula{2}(1.0), [0.5, 0.5]) atol=1e-12
    @test cells[1, 1] ≈ 0.3125 atol=1e-12
    @test sum(cells) ≈ 1 atol=1e-12
    @test cells[1, 2] ≈ 0.5 - cells[1, 1] atol=1e-12
    @test logpdf(S, [0, 0]) ≈ log(cells[1, 1])
    @test pdf(S, [0.5, 0.5]) == 0

    # Normal × Poisson under Clayton: integrating out the continuous coordinate
    # recovers the Poisson pmf, and the conditional mean matches a sample.
    X = SklarDist(ClaytonCopula(2, 2.0), (Normal(), Poisson(3.0)))
    for k in 0:8
        @test quadgk(x -> pdf(X, [x, k]), -8, 8)[1] ≈ pdf(Poisson(3.0), k) atol=1e-12
    end
    @test pdf(X, [0.3, 2.5]) == 0
    sample = rand(StableRNG(491), X, 1_000_000)
    keep = sample[2, :] .== 2
    numerator = quadgk(x -> x * pdf(X, [x, 2]), -8, 8)[1]
    @test numerator / pdf(Poisson(3.0), 2) ≈ mean(view(sample, 1, keep)) atol=1e-2

    # Continuous margins recover the density factorization exactly.
    Y = SklarDist(ClaytonCopula(2, 2.0), (Normal(), Normal()))
    x = [0.3, -0.2]
    @test logpdf(Y, x) == logpdf(Normal(), 0.3) + logpdf(Normal(), -0.2) +
                          logpdf(ClaytonCopula(2, 2.0), cdf.(Normal(), x))
    # ... and the atom route reduces to the same number when every margin is continuous.
    @test Copulas._sklar_logpdf_atoms(Y, x) ≈ logpdf(Y, x) atol=1e-12

    # Three discrete margins: the masses over a truncated support sum to one.
    T3 = SklarDist(GaussianCopula([1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
                   (Bernoulli(0.3), Poisson(1.0), Bernoulli(0.6)))
    total = sum(pdf(T3, [i, k, j]) for i in 0:1, k in 0:30, j in 0:1)
    @test total ≈ 1 atol=1e-8
end
