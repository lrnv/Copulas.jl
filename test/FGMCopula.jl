@testitem "Generic" tags=[:FGMCopula] setup=[M] begin M.check(FGMCopula(2, 0.0)) end
@testitem "Generic" tags=[:FGMCopula] setup=[M] begin M.check(FGMCopula(2, rand(M.rng))) end
@testitem "Generic" tags=[:FGMCopula] setup=[M] begin M.check(FGMCopula(2,1)) end
@testitem "Generic" tags=[:FGMCopula] setup=[M] begin M.check(FGMCopula(3, [0.3,0.3,0.3,0.3])) end
@testitem "Generic" tags=[:FGMCopula] setup=[M] begin M.check(FGMCopula(3,[0.1,0.2,0.3,0.4])) end

@testset "Fixing values of FGMCopula - cdf, pdf, constructor" begin

    @test isa(FGMCopula(2,0.0), IndependentCopula)

    using StableRNGs
    rng = StableRNG(123)

    cdf_exs = [
        ([0.1, 0.2, 0.3], [0.0100776123, 1e-4], [0.1,0.2,0.5,0.4]),
        ([0.5, 0.4, 0.3], [0.0830421321, 1e-4], [0.3,0.3,0.3,0.3]),
        ([0.1, 0.1], [0.010023, 1e-4], 0.0),
        ([0.5, 0.4], [0.224013, 1e-4], 0.5),
    ]
    
    for (u, expected) in cdf_exs
        copula = FGMCopula(length(u), expected[3])
        @test cdf(copula, u) ≈ expected[1] atol=expected[2]
    end

    pdf_exs = [
        ([0.1, 0.2, 0.3], [1.308876232, 1e-4], [0.1,0.2,0.5,0.4]),
        ([0.5, 0.4, 0.3], [1.024123232, 1e-4], [0.3,0.3,0.3,0.3]),
        ([0.1, 0.1], [0.01, 1e-4], 0.0),
        ([0.5, 0.4], [1, 1e-4], rand(rng)),
    ]
    
    for (u, expected) in pdf_exs
        copula = FGMCopula(length(u), expected[3])
        @test cdf(copula, u) ≈ expected[1] atol=expected[2]
    end
end