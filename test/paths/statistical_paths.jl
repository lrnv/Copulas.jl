# Statistical-path layer: validates representative samplers and Rosenblatt
# transforms statistically without repeating Monte Carlo checks for every family.
@testset "representative sampler and Rosenblatt statistics" begin
    for C in (ClaytonCopula{2}(1.5), GaussianCopula{2}(0.3),
              GalambosCopula{2}(1.0), FGMCopula{2}(0.4))
        U = rand(StableRNG(101), C, 400)
        point = [0.7, 0.8]
        theoretical = cdf(C, point)
        empirical = mean(all(U .<= point; dims=1))
        se = sqrt(theoretical * (1 - theoretical) / size(U, 2))
        @test abs(empirical - theoretical) <= max(5se, 0.03)

        R = rosenblatt(C, U)
        @test abs(StatsBase.corkendall(transpose(R))[1, 2]) <= 0.15
    end
end

@testset "singular spectral sampler structure" begin
    C = CuadrasAugeCopula{2}(0.5)
    U = rand(StableRNG(102), C, 400)
    observed = mean(U[1, :] .== U[2, :])
    expected = 0.5 / (2 - 0.5)
    se = sqrt(expected * (1 - expected) / size(U, 2))
    @test abs(observed - expected) <= max(5se, 0.03)

    for C in (BC2Copula{2}(0.5, 0.3), MOCopula{2}(0.2, 0.3, 0.4))
        U = rand(StableRNG(103), C, 400)
        x, y = -log.(U[1, :]), -log.(U[2, :])
        if C isa BC2Copula
            a, b = params(C).a, params(C).b
            atom = isapprox.(a .* x, b .* y; atol=1e-10, rtol=1e-7) .|
                   isapprox.((1 - a) .* x, (1 - b) .* y; atol=1e-10, rtol=1e-7)
            expected = 1 - abs(a - b)
        else
            p = params(C)
            atom = isapprox.((p.λ₁ + p.λ₃) .* x, (p.λ₂ + p.λ₃) .* y;
                            atol=1e-10, rtol=1e-7)
            expected = p.λ₃ / (p.λ₁ + p.λ₂ + p.λ₃)
        end
        observed = mean(atom)
        se = sqrt(expected * (1 - expected) / size(U, 2))
        @test abs(observed - expected) <= max(5se, 0.03)
    end
end
