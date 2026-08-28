# Correctness obligation: validates samplers and Rosenblatt transforms
# statistically once per distinct implementation route.
@testset verbose=true "one distributional identity per sampler dispatch" begin
    seen = Set{Any}()
    for (index, case) in pairs(ROUTING_COPULA_CASES)
        C = case.build()
        d = length(C)
        route_rng = StableRNG(400 + index)
        method = which(Distributions._rand!,
            Tuple{typeof(route_rng),typeof(C),Matrix{Float64}})
        key = (method, d == 2 ? :bivariate : :multivariate)
        key in seen && continue
        push!(seen, key)

        @testset "$(case.name)" begin
            n = 160
            U = rand(route_rng, C, n)
            point = fill(0.72, d)
            theoretical = cdf(C, point)
            empirical = mean(all(U .<= point; dims=1))
            se = sqrt(max(theoretical * (1 - theoretical), eps()) / n)
            @test abs(empirical - theoretical) <= max(6se, 0.08)
            @test all(abs(mean(view(U, i, :)) - 0.5) <= 0.12 for i in 1:d)
        end
    end
    @test !isempty(seen)
end

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

@testset "empirical dependence estimators match their theoretical targets" begin
    C = ClaytonCopula{2}(2.0)
    U = rand(StableRNG(104), C, 2_000)
    for measure in (Copulas.τ, Copulas.ρ, Copulas.β)
        @test measure(U) ≈ measure(C) atol=0.1
    end
    @test Copulas.γ(U) ≈ Copulas.γ(C) atol=0.15
    @test Copulas.ι(U) ≈ Copulas.ι(C) atol=0.15

    observations = transpose(U)
    @test Copulas.corblomqvist(observations)[1, 2] ≈ Copulas.β(C) atol=0.1
    @test Copulas.corgini(observations)[1, 2] ≈ Copulas.γ(C) atol=0.1
    entropy = Copulas.corentropy(observations)
    @test diag(entropy) == zeros(2)
    @test isfinite(entropy[1, 2])
end
