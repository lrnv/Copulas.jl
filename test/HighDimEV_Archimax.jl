@testitem "High-d EV and Archimax basics" tags=[:HighDim, :ExtremeValueCopula, :ArchimaxCopula] setup=[M] begin
    using Copulas, Distributions, StableRNGs, Test
    rng = StableRNG(42)

    # 3D Logistic EV (aka Gumbel EV) via LogTail(θ)
    θs = (1.2, 2.5)
    for θ in θs
        Cev = Copulas.ExtremeValueCopula(3, Copulas.LogTail(θ))
        U = rand(rng, Cev, 200)
        @test all(0 .<= U .<= 1)
        @test 0.0 ≤ cdf(Cev, [0.5,0.5,0.5]) ≤ 1.0
        @test isfinite(logpdf(Cev, [0.3,0.6,0.7]))
        # small smoke on rosenblatt ∘ inverse
        U2 = Copulas.inverse_rosenblatt(Cev, Copulas.rosenblatt(Cev, U))
        @test isapprox(U, U2; atol=1e-2)
    end

    # 4D Galambos EV (negative logistic)
    for θ in (0.7, 2.0)
        Cev = Copulas.ExtremeValueCopula(4, Copulas.GalambosTail(θ))
        U = rand(rng, Cev, 100)
        @test all(0 .<= U .<= 1)
        @test 0.0 ≤ cdf(Cev, fill(0.5, 4)) ≤ 1.0
        @test isfinite(logpdf(Cev, fill(0.6, 4)))
    end

    # 3D Archimax: Clayton × Logistic tail
    for (θg, θt) in ((1.5, 1.3), (3.0, 2.0))
        Cax = Copulas.ArchimaxCopula(3, Copulas.ClaytonGenerator(θg), Copulas.LogTail(θt))
        U = rand(rng, Cax, 150)
        @test all(0 .<= U .<= 1)
        @test 0.0 ≤ cdf(Cax, [0.2,0.7,0.5]) ≤ 1.0
        @test isfinite(logpdf(Cax, [0.4,0.6,0.8]))
    end
end

@testitem "Logistic EV specialized sampler & distortion" tags=[:ExtremeValueCopula, :LogisticSampler] setup=[M] begin
    using Copulas, Distributions, StableRNGs, Test, Statistics
    rng = StableRNG(84)
    for d in (3,5)
        θ = 2.5
        C = Copulas.ExtremeValueCopula(d, Copulas.LogTail(θ))
        n = 1500
        U = rand(rng, C, n)
        @test all(0 .<= U .<= 1)
        # Basic marginal sanity: mean(U_i) should be close to 0.5 (uniform margins)
        μ = mean(U[1,:])
        @test 0.4 < μ < 0.6
        # Distortion specialization check
        j = 1; i = 2
        u_j = U[j, 10]
        D = Copulas.DistortionFromCop(C, (j,), (u_j,), i)
        val = cdf(D, U[i,10])
        @test 0.0 <= val <= 1.0
    end
end
