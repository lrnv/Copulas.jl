@testset "EmpiricalCopula CDF includes boundary atoms" begin
    example = EmpiricalCopula([0.0 0.5; 0.2 0.8])
    @test cdf(example, [0.0, 0.5]) == 0.5
    @test cdf(example, [-eps(), 0.5]) == 0.0
    @test logcdf(example, [0.0, 0.5]) == log(0.5)

    # Binary-exact atom coordinates keep the Float32 and Float64 boundary
    # queries identical; decimal rounding must not change which atom is hit.
    points = [0.0 0.0 0.0 0.5 1.0; 0.0 0.0 0.25 0.75 1.0]
    queries = [
        [0.0, 0.0], [0.0, 0.1], [0.0, 0.25], [0.0, 1.0],
        [1.0, 0.0], [0.5, 0.75], [prevfloat(1.0), prevfloat(1.0)],
        [1.0, 1.0], [-0.1, 1.0], [1.0, -0.1], [2.0, 0.25],
        [0.0, 2.0], [2.0, 2.0], [-Inf, Inf], [Inf, Inf],
    ]
    # Independent oracle: the CDF of the equally weighted atomic sample.
    reference = [count(point -> all(point .<= query), eachcol(points)) /
                 size(points, 2) for query in queries]
    for T in (Float32, Float64), d in (2, 3)
        data = T.(d == 2 ? points : vcat(points, points[1:1, :]))
        C = EmpiricalCopula(data)
        query_matrix = hcat((d == 2 ? query : [query; query[1]] for query in queries)...)
        @test cdf(C, query_matrix) == reference
        @test logcdf(C, query_matrix) == log.(reference)
        for (query, expected) in zip(eachcol(query_matrix), reference)
            @test cdf(C, query) == expected
        end
        @test cdf(C, zeros(d)) == 2 / 5
        @test pdf(C, zeros(d)) == 2 / 5
        @test C.u == data
        @test_throws DimensionMismatch cdf(C, zeros(d - 1))
        @test_throws DimensionMismatch cdf(C, zeros(d + 1, 2))
    end
    # Step margins remain empirical margins, rather than forced uniforms.
    @test cdf(EmpiricalCopula(points), [0.0, 1.0]) == 3 / 5
    @test cdf(EmpiricalCopula(points), [0.5, 1.0]) == 4 / 5
    S = @test_logs (:warn, r"EmpiricalCopula has finite-sample step margins") begin
        SklarDist(example, (Uniform(), Uniform()))
    end
    @test cdf(S, [0.0, 0.5]) == 0.5
    # Genuine copulas retain their zero-face CDF behavior.
    @test cdf(IndependentCopula(2), [0.0, 0.5]) == 0
    @test cdf(ClaytonCopula(2, 1.0), [0.0, 0.5]) == 0
end
