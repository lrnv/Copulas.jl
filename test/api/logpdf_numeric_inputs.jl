@testset "copula logpdf outside support accepts exact numeric inputs" begin
    for T in (Int8, Int, Int128, BigInt, Rational{Int}, Float32, Float64, BigFloat)
        for C in (IndependentCopula(2), ClaytonCopula(2, 1.0))
            lower = T[-1, 0]
            upper = T[0, 2]
            @test logpdf(C, lower) == -Inf
            @test logpdf(C, upper) == -Inf
            @test logpdf(C, lower) isa typeof(float(one(T)))
            @test pdf(C, lower) == 0
            @test pdf(C, upper) == 0
            @test logpdf(C, hcat(lower, upper)) == [-Inf, -Inf]
            @test lower == T[-1, 0]
            @test upper == T[0, 2]
            @test_throws DimensionMismatch logpdf(C, T[-1])
        end
    end
    C = IndependentCopula(3)
    @test logpdf(C, [0, 1, 0]) == 0
    @test logpdf(C, [-1, 1, 0]) == -Inf
    @test logpdf(C, [0, 1, 2]) == -Inf
    @test logpdf(C, [0 0 -1; 1 1 1; 0 2 0]) == [0, -Inf, -Inf]
    # Abstractly typed real vectors still have a concrete floating result.
    @test logpdf(C, Real[-1, 0, 1]) === -Inf
end
