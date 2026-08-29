# Focused conditioning regressions that inspect implementation state or
# reproduce family-specific numerical bugs; shared identities live under the
# contract and equivalence obligations.

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
