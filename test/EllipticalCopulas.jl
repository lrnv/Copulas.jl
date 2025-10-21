@testset "GaussianCopula" begin
    # [GenericTests integration]: Maybe. The broken fit on mixed marginals is out-of-scope for generic copula properties; keep here.
    Random.seed!(rng,123)
    C = GaussianCopula([1 -0.1; -0.1 1])
    M1 = Beta(2,3)
    M2 = LogNormal(2,3)
    D = SklarDist(C,(M1,M2))
    X = rand(rng,D,10)
    loglikelihood(D,X)
    @test true
end

@testset "Fix value Gaussian Copula & SklarDist" begin
    # [GenericTests integration]: Yes. This is a regression value test for cdf(SklarDist(...)); can be moved to a generic Sklar fixture tests.

    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula([1 0.5; 0.5 1])
    D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
end

@testset "GaussianCopula equicorrelation constructor" begin
    Cρ = GaussianCopula(2, 0.5)
    @test Cρ isa GaussianCopula{2}
    # Theoretical Kendall tau for bivariate Gaussian: τ = 2/π asin(ρ)
    @test isapprox(Copulas.τ(Cρ), 2*asin(0.5)/π; rtol=1e-12)
    # Zero correlation gives independent copula
    C0 = GaussianCopula(2, 0.0)
    @test C0 == IndependentCopula(2)
    # PD lower bound check (just above boundary for d=3: lower = -0.5)
    Cneg = GaussianCopula(3, -0.49)
    @test Cneg isa GaussianCopula{3}
    # Boundary should throw
    @test_throws ArgumentError GaussianCopula(3, -0.5)
end

@testset "GaussianCopula vs MvNormalCDF" begin
    Random.seed!(rng, 42)
    for d in (2,3,4,6)
        for trial in 1:20
            # random correlation matrix via LKJ-like construction: random orthonormal and diag 1
            A = randn(d,d)
            Q, R = qr(A)
            Σ = Matrix(Q*Q')
            # mix with identity to keep PD and not too extreme
            ρ = rand() * 0.8
            Σ = (1-ρ)*I + ρ*Σ
            # ensure PD and correlation-like scaling
            D = sqrt.(diag(Σ))
            Σ = Σ ./ (D*D')

            C = Copulas.GaussianCopula(Σ)
            u = rand(rng, d)
            # our cdf uses QMC; reduce m/r for speed
            p_cop = Distributions.cdf(C, u; m=2000 * d, r=8, rng=rng)

            x = Statistics.quantile.(Distributions.Normal(), u)
            p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, d), x)[1]

            @test isfinite(p_cop) && isfinite(p_ref)
            # allow small relative error; QMC has randomness but we seeded rng
            @test isapprox(p_cop, p_ref; atol=1e-6, rtol=5e-2)
        end
    end
end

# Extreme/structured correlation matrices vs MvNormalCDF
@testset "GaussianCopula vs MvNormalCDF (structured/extreme)" begin
    Random.seed!(rng, 777)
    # Equicorrelation near boundaries (still PD): ρ in {0.95, 0.99}
    for d in (3, 5, 10)
        for ρ in (0.95, 0.99)
            Σ = fill(ρ, d, d); Σ[diagind(Σ)] .= 1.0
            C = Copulas.GaussianCopula(Σ)
            u = rand(rng, d)
            p_cop = Distributions.cdf(C, u; m=2500 * min(d,6), r=8, rng=rng)
            x = Statistics.quantile.(Distributions.Normal(), u)
            p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, d), x)[1]
            @test isfinite(p_cop) && isfinite(p_ref)
            @test isapprox(p_cop, p_ref; atol=2e-6, rtol=8e-2)
        end
    end

    # AR(1) Toeplitz with strong positive and negative correlations
    for d in (4, 8)
        for ρ in (0.95, -0.95)
            Σ = [ρ^(abs(i-j)) for i in 1:d, j in 1:d]
            C = Copulas.GaussianCopula(Σ)
            u = rand(rng, d)
            p_cop = Distributions.cdf(C, u; m=2500 * min(d,6), r=8, rng=rng)
            x = Statistics.quantile.(Distributions.Normal(), u)
            p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, d), x)[1]
            @test isfinite(p_cop) && isfinite(p_ref)
            @test isapprox(p_cop, p_ref; atol=2e-6, rtol=1e-1)
        end
    end

    # Block-diagonal with diverse blocks
    let
        # 2x2 block with moderate positive corr, 3x3 equicorr near boundary
        B1 = [1.0 0.7; 0.7 1.0]
        ρ = 0.9; B2 = fill(ρ, 3, 3); B2[diagind(B2)] .= 1.0
        Σ = zeros(5,5); Σ[1:2,1:2] .= B1; Σ[3:5,3:5] .= B2
        C = Copulas.GaussianCopula(Σ)
        u = rand(rng, 5)
        p_cop = Distributions.cdf(C, u; m=8000, r=8, rng=rng)
        x = Statistics.quantile.(Distributions.Normal(), u)
        p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, 5), x)[1]
        @test isfinite(p_cop) && isfinite(p_ref)
        @test isapprox(p_cop, p_ref; atol=2e-6, rtol=8e-2)
    end
end

# High-dimensional random correlations (smaller trials, looser tol)
@testset "GaussianCopula vs MvNormalCDF (high-d)" begin
    Random.seed!(rng, 31415)
    for d in (8, 12, 16)
        for trial in 1:3
            A = randn(rng, d, d)
            Σ = A * A'  # SPD
            D = sqrt.(diag(Σ)); Σ = Σ ./ (D * D') # scale to correlation
            C = Copulas.GaussianCopula(Σ)
            u = rand(rng, d)
            # Fewer QMC reps to keep CI time under control for CI
            p_cop = Distributions.cdf(C, u; m=1500, r=6, rng=rng)
            x = Statistics.quantile.(Distributions.Normal(), u)
            p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, d), x)[1]
            @test isfinite(p_cop) && isfinite(p_ref)
            @test isapprox(p_cop, p_ref; atol=5e-6, rtol=1.5e-1)
        end
    end
end

# Tail u values near 0/1 to stress numerical stability
@testset "GaussianCopula vs MvNormalCDF (tails)" begin
    Random.seed!(rng, 2024)
    for d in (3, 6)
        # Use equicorrelation and AR(1) variants
        for Σ in begin
            ρ = 0.8; S1 = fill(ρ, d, d); S1[diagind(S1)] .= 1.0
            ρ2 = 0.9; S2 = [ρ2^(abs(i-j)) for i in 1:d, j in 1:d]
            (S1, S2)
        end
            # Construct u with extremes and interior values
            u = fill(0.0, d)
            for i in 1:d
                u[i] = i % 3 == 1 ? 1e-6 : (i % 3 == 2 ? (1 - 1e-6) : rand(rng))
            end
            C = Copulas.GaussianCopula(Σ)
            p_cop = Distributions.cdf(C, u; m=6000, r=8, rng=rng)
            x = Statistics.quantile.(Distributions.Normal(), u)
            p_ref = MvNormalCDF.mvnormcdf(x, Matrix(Σ), fill(-Inf, d), x)[1]
            @test isfinite(p_cop) && isfinite(p_ref)
            @test isapprox(p_cop, p_ref; atol=1e-5, rtol=2e-1)
        end
    end
end

# Compare TCopula.cdf to a crude Monte Carlo for a few small dimensions
@testset "TCopula vs MonteCarlo" begin
    Random.seed!(rng, 123)
    for d in (2,3)
        for ν in (3,5,10)
            for trial in 1:10
                # random correlation matrix
                A = randn(d,d)
                Q, R = qr(A)
                Σ = Matrix(Q*Q')
                ρ = rand() * 0.6
                Σ = (1-ρ)*I + ρ*Σ
                D = sqrt.(diag(Σ))
                Σ = Σ ./ (D*D')
                C = Copulas.TCopula(ν, Σ)
                u = rand(rng, d)
                p_cop = Distributions.cdf(C, u; m=2000 * d, r=8, rng=rng)

                # crude Monte Carlo estimate using t multivariate draws
                mv = Distributions.MvTDist(ν, Σ)
                nmc = 20_000
                z = rand(rng, mv, nmc)
                # compute fraction with all coords <= quantiles
                quants = Distributions.quantile.(Distributions.TDist(ν), u)
                hits = sum(all(z[:,i] .<= quants[i] for i in 1:d) for j in 1:nmc)
                p_mc = hits / nmc
                @test isfinite(p_cop) && isfinite(p_mc)
                @test isapprox(p_cop, p_mc; atol=3e-2, rtol=1e-1)
            end
        end
    end
end
