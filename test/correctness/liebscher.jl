@testset "Liebscher defining representation" begin
    C1 = ClaytonCopula{2}(2.0)
    C2 = GumbelCopula{2}(1.5)
    W = [0.3 0.8; 0.7 0.2]
    C = LiebscherCopula(2, (C1, C2), W)

    @test C isa LiebscherCopula{2}
    @test length(C) == 2
    @test eltype(C) == Float64
    @test params(C).copulas == (C1, C2)
    @test params(C).weights == W
    @test params(C).weights !== C.weights

    reconstructed = typeof(C)(values(params(C))...)

    @test typeof(reconstructed) === typeof(C)
    @test params(reconstructed) == params(C)

    for u in ([0.31, 0.47], [0.52, 0.73], [0.81, 0.64])
        expected = cdf(C1, u .^ W[1, :]) * cdf(C2, u .^ W[2, :])
        @test cdf(C, u) ≈ expected
        @test cdf(C, [u[1], 1.0]) ≈ u[1]
        @test cdf(C, [1.0, u[2]]) ≈ u[2]
    end

    @test cdf(C, [0.0, 0.6]) == 0
    @test cdf(C, [0.6, 0.0]) == 0
    @test cdf(C, ones(2)) == 1
end

@testset "Liebscher multivariate defining representation" begin
    C1 = ClaytonCopula{3}(1.5)
    C2 = GumbelCopula{3}(1.4)
    W = [0.2 0.5 0.7; 0.8 0.5 0.3]
    C = LiebscherCopula(3, (C1, C2), W)
    u = [0.31, 0.57, 0.82]
    expected = cdf(C1, u .^ W[1, :]) * cdf(C2, u .^ W[2, :])

    @test C isa LiebscherCopula{3}
    @test cdf(C, u) ≈ expected

    for j in 1:3
        margin = ones(3)
        margin[j] = u[j]
        @test cdf(C, margin) ≈ u[j]
    end
end

@testset "Liebscher inactive components" begin
    C1 = ClaytonCopula{2}(2.0)
    C2 = GumbelCopula{2}(1.5)
    u = [0.37, 0.68]

    first_only = LiebscherCopula(2, (C1, C2), [1.0 1.0; 0.0 0.0])
    second_only = LiebscherCopula(2, (C1, C2), [0.0 0.0; 1.0 1.0])
    independent = LiebscherCopula(2, (IndependentCopula{2}(), IndependentCopula{2}()), [0.3 0.8; 0.7 0.2])

    @test cdf(first_only, u) ≈ cdf(C1, u)
    @test cdf(second_only, u) ≈ cdf(C2, u)
    @test cdf(independent, u) ≈ prod(u)
    @test pdf(independent, u) ≈ 1
end

@testset "Liebscher measure semantics" begin
    Π = IndependentCopula{2}()
    M = MCopula{2}()

    regular = LiebscherCopula(2, (ClaytonCopula{2}(2.0), GumbelCopula{2}(1.5)), [0.3 0.8; 0.7 0.2])
    singular = LiebscherCopula(2, (M, Π), [0.5 0.5; 0.5 0.5])
    one_active_coordinate = LiebscherCopula(2, (M, Π), [0.5 0.0; 0.5 1.0])
    inactive_singular = LiebscherCopula(2, (M, Π), [0.0 0.0; 1.0 1.0])

    @test Copulas.copula_measure_style(regular) isa Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(singular) isa Copulas.NonAbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(one_active_coordinate) isa Copulas.AbsolutelyContinuousMeasure
    @test Copulas.copula_measure_style(inactive_singular) isa Copulas.AbsolutelyContinuousMeasure
    @test cdf(one_active_coordinate, [0.37, 0.68]) ≈ 0.37 * 0.68
    @test cdf(inactive_singular, [0.37, 0.68]) ≈ 0.37 * 0.68
end

@testset "Liebscher mixed partials and density" begin
    C = LiebscherCopula(2, (ClaytonCopula{2}(2.0), GumbelCopula{2}(1.5)), [0.3 0.8; 0.7 0.2])
    u = [0.37, 0.68]

    d1 = ForwardDiff.derivative(x -> cdf(C, [x, u[2]]), u[1])
    d2 = ForwardDiff.derivative(y -> cdf(C, [u[1], y]), u[2])
    d12 = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> cdf(C, [x, y]), u[2]), u[1])

    @test Copulas._partial_cdf(C, (2,), (1,), (u[2],), (u[1],)) ≈ d1 rtol=1e-8
    @test Copulas._partial_cdf(C, (1,), (2,), (u[1],), (u[2],)) ≈ d2 rtol=1e-8
    @test pdf(C, u) ≈ d12 rtol=1e-8
    @test logpdf(C, u) ≈ log(d12) rtol=1e-8
    @test exp(logpdf(C, u)) ≈ pdf(C, u) rtol=1e-8
end

@testset "Liebscher multivariate density" begin
    C = LiebscherCopula(3, (ClaytonCopula{3}(1.5), GumbelCopula{3}(1.4)), [0.2 0.5 0.7; 0.8 0.5 0.3])
    u = [0.31, 0.57, 0.82]
    oracle = ForwardDiff.derivative(x -> ForwardDiff.derivative(y -> ForwardDiff.derivative(z -> cdf(C, [x, y, z]), u[3]), u[2]), u[1])

    @test isfinite(pdf(C, u))
    @test pdf(C, u) > 0
    @test pdf(C, u) ≈ oracle rtol=1e-7
    @test logpdf(C, u) ≈ log(oracle) rtol=1e-7
end

@testset "Liebscher numerical CDF components" begin
    G = GaussianCopula([1.0 0.5; 0.5 1.0])
    C = LiebscherCopula(2, (G, ClaytonCopula{2}(1.7)), [0.4 0.7; 0.6 0.3])
    u = [0.41, 0.73]

    d1 = Copulas._partial_cdf(C, (2,), (1,), (u[2],), (u[1],))
    d2 = Copulas._partial_cdf(C, (1,), (2,), (u[1],), (u[2],))

    @test isfinite(pdf(C, u))
    @test pdf(C, u) > 0
    @test isfinite(logpdf(C, u))
    @test isfinite(d1)
    @test isfinite(d2)
    @test d1 > 0
    @test d2 > 0
end

@testset "Liebscher subsetting identity" begin
    C1 = ClaytonCopula{3}(1.5)
    C2 = GumbelCopula{3}(1.4)
    W = [0.3 0.6 0.4; 0.7 0.4 0.6]
    C = LiebscherCopula(3, (C1, C2), W)
    S = subsetdims(C, (3, 1))
    u = [0.43, 0.71]
    expected = cdf(subsetdims(C1, (3, 1)), u .^ W[1, [3, 1]]) * cdf(subsetdims(C2, (3, 1)), u .^ W[2, [3, 1]])

    @test S isa LiebscherCopula{2}
    @test S.weights ≈ W[:, [3, 1]]
    @test cdf(S, u) ≈ cdf(C, [u[2], 1.0, u[1]])
    @test cdf(S, u) ≈ expected
    @test isfinite(pdf(S, u))
    @test pdf(S, u) > 0

    reordered = subsetdims(C, (3, 2, 1))
    v = [0.29, 0.58, 0.81]

    @test reordered isa LiebscherCopula{3}
    @test reordered.weights ≈ W[:, [3, 2, 1]]
    @test cdf(reordered, v) ≈ cdf(C, [v[3], v[2], v[1]])
    @test subsetdims(C, (1, 2, 3)) === C
    @test subsetdims(C, (2,)) isa Uniform
end

@testset "Liebscher conditioning identity" begin
    C = LiebscherCopula(2, (ClaytonCopula{2}(2.0), GumbelCopula{2}(1.5)), [0.3 0.8; 0.7 0.2])
    u1 = 0.41
    u2 = 0.67
    D = condition(C, 1, u1)
    numerator = Copulas._partial_cdf(C, (2,), (1,), (u2,), (u1,))
    denominator = Copulas._partial_cdf(C, (), (1,), (), (u1,))
    density = pdf(D, u2)
    q = quantile(D, 0.63)

    @test D isa UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    @test cdf(D, u2) ≈ numerator / denominator rtol=1e-9
    @test 0 <= cdf(D, u2) <= 1
    @test isfinite(density)
    @test density > 0
    @test logpdf(D, u2) ≈ log(density)
    @test 0 <= q <= 1
    @test cdf(D, q) ≈ 0.63 atol=1e-8
end

@testset "Liebscher multivariate conditioning" begin
    C = LiebscherCopula(3, (ClaytonCopula{3}(1.5), GumbelCopula{3}(1.4)), [0.2 0.5 0.7; 0.8 0.5 0.3])
    u1 = 0.37
    target = [0.56, 0.74]
    D = condition(C, 1, u1)
    value = cdf(D, target)
    numerator = Copulas._partial_cdf(C, (2, 3), (1,), Tuple(target), (u1,))
    denominator = Copulas._partial_cdf(C, (), (1,), (), (u1,))

    @test length(D) == 2
    @test isfinite(value)
    @test 0 <= value <= 1
    @test value ≈ numerator / denominator rtol=1e-8
end

@testset "Liebscher Rosenblatt round trip" begin
    C = LiebscherCopula(3, (ClaytonCopula{3}(1.5), GumbelCopula{3}(1.4)), [0.2 0.5 0.7; 0.8 0.5 0.3])
    U = [0.21 0.43 0.72; 0.37 0.61 0.52; 0.68 0.79 0.34]
    R = rosenblatt(C, U)
    U2 = inverse_rosenblatt(C, R)

    @test size(R) == size(U)
    @test all(0 .<= R .<= 1)
    @test R[1, :] ≈ U[1, :]
    @test U2 ≈ U atol=1e-8 rtol=1e-8
end

@testset "Liebscher Rosenblatt with numerical CDF component" begin
    G = GaussianCopula([1.0 0.4 0.2; 0.4 1.0 0.35; 0.2 0.35 1.0])
    C = LiebscherCopula(3, (G, ClaytonCopula{3}(1.4)), [0.4 0.6 0.3; 0.6 0.4 0.7])
    U = [0.28 0.47; 0.63 0.39; 0.74 0.58]
    R = rosenblatt(C, U)
    U2 = inverse_rosenblatt(C, R)

    @test all(isfinite, R)
    @test all(0 .<= R .<= 1)
    @test U2 ≈ U atol=2e-7 rtol=2e-7
end

@testset "Khoudraji is Liebscher sugar" begin
    C1 = ClaytonCopula{2}(2.0)
    C2 = GumbelCopula{2}(1.5)
    shapes = [0.25, 0.80]
    K = KhoudrajiCopula(2, (C1, C2), shapes)
    W = [1 - shapes[1] 1 - shapes[2]; shapes[1] shapes[2]]
    L = LiebscherCopula(2, (C1, C2), W)

    @test K isa LiebscherCopula{2}
    @test typeof(K) === typeof(L)
    @test params(K).copulas == params(L).copulas
    @test params(K).weights ≈ params(L).weights
    @test K.weights ≈ W

    for u in ([0.31, 0.47], [0.52, 0.73], [0.81, 0.64])
        expected = cdf(C1, u .^ (1 .- shapes)) * cdf(C2, u .^ shapes)
        @test cdf(K, u) ≈ expected
        @test cdf(K, u) ≈ cdf(L, u)
        @test pdf(K, u) ≈ pdf(L, u)
    end
end

@testset "Khoudraji single-component convenience" begin
    C = ClaytonCopula{2}(1.7)
    shapes = [0.30, 0.75]
    K = KhoudrajiCopula(2, C, shapes)
    L = LiebscherCopula(2, (IndependentCopula{2}(), C), [1 - shapes[1] 1 - shapes[2]; shapes[1] shapes[2]])
    u = [0.42, 0.69]
    expected = prod(u .^ (1 .- shapes)) * cdf(C, u .^ shapes)

    @test K isa LiebscherCopula{2}
    @test K.copulas[1] isa IndependentCopula{2}
    @test K.copulas[2] === C
    @test cdf(K, u) ≈ expected
    @test cdf(K, u) ≈ cdf(L, u)
    @test pdf(K, u) ≈ pdf(L, u)

    independent_limit = KhoudrajiCopula(2, C, zeros(2))
    copula_limit = KhoudrajiCopula(2, C, ones(2))

    @test cdf(independent_limit, u) ≈ prod(u)
    @test cdf(copula_limit, u) ≈ cdf(C, u)
end

@testset "Khoudraji multivariate construction" begin
    C1 = ClaytonCopula{3}(1.4)
    C2 = GumbelCopula{3}(1.3)
    shapes = [0.2, 0.6, 0.85]
    K = KhoudrajiCopula(3, (C1, C2), shapes)
    u = [0.37, 0.58, 0.76]
    expected = cdf(C1, u .^ (1 .- shapes)) * cdf(C2, u .^ shapes)

    @test K isa LiebscherCopula{3}
    @test K.weights ≈ [0.8 0.4 0.15; 0.2 0.6 0.85]
    @test cdf(K, u) ≈ expected
    @test isfinite(pdf(K, u))
    @test pdf(K, u) > 0

    S = subsetdims(K, (3, 1))

    @test S isa LiebscherCopula{2}
    @test S.weights ≈ K.weights[:, [3, 1]]
end

@testset "Khoudraji inherited operations" begin
    K = KhoudrajiCopula(3, (ClaytonCopula{3}(1.5), GumbelCopula{3}(1.4)), [0.2, 0.5, 0.8])
    U = [0.28 0.47; 0.63 0.39; 0.74 0.58]
    R = rosenblatt(K, U)
    U2 = inverse_rosenblatt(K, R)
    D = condition(K, 1, 0.41)
    sample = rand(K, 1_000)

    @test U2 ≈ U atol=1e-8 rtol=1e-8
    @test length(D) == 2
    @test 0 <= cdf(D, [0.53, 0.71]) <= 1
    @test size(sample) == (3, 1_000)
    @test all(0 .<= sample .<= 1)
end

@testset "Liebscher 3D parameter differentiation" begin
    C0 = LiebscherCopula(
        3,
        (ClaytonCopula(3, 0.8), GumbelCopula(3, 1.2)),
        [0.5 0.5 0.5; 0.5 0.5 0.5],
    )

    α0 = Copulas._liebscher_unbound(C0)
    u = [0.3, 0.5, 0.8]

    f(α) = logpdf(Copulas._liebscher_rebound(C0, α), u)

    g = ForwardDiff.gradient(f, α0)

    @test length(g) == 5
    @test all(isfinite, g)
end