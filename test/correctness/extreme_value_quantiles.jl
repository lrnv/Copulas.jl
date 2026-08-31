# Correctness obligation: bounded generalized inversion for singular
# extreme-value radial distributions.
@testset "extreme-value quantiles use bounded bisection" begin
    for C in (
        BC2Copula{2}(0.5, 0.3),
        BC2Copula{2}(0.5516353577049822, 0.33689370624999193),
        BC2Copula{2}(0.6, 0.8),
    )
        D = Copulas.ExtremeDist(C.tail)
        for p in (0.1, 0.5, 0.9)
            q = quantile(D, p)
            @test 0 <= q <= 1
            @test cdf(D, q) >= p
        end
    end
end
