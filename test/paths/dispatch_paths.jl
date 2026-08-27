# Mechanism-path layer: exercises one representative of each important generic
# or specialized sampling, conditioning, subsetting, and numerical dispatch path.
@testset "representative dispatch paths" begin
    for (name, C) in pairs(PATH_CASES)
        @testset "$name" begin
            d = length(C)
            u = fill(0.6, d)
            @test 0 <= cdf(C, u) <= 1
            @test size(rand(StableRNG(51), C, 2)) == (d, 2)
            js = Tuple(1:(d - 1))
            D = condition(C, js, ntuple(_ -> 0.4, d - 1))
            @test 0 <= cdf(D, 0.6) <= 1
        end
    end
end

@testset "specialized FGM paths agree with the generic polynomial oracle" begin
    θ = 0.4
    generic = PolynomialOracleCopula(θ)
    specialized = FGMCopula{2}(θ)
    u = [0.37, 0.68]

    generic_integrated_cdf =
        invoke(Copulas._cdf, Tuple{Copulas.Copula,Any}, generic, u)
    @test cdf(specialized, u) ≈ generic_integrated_cdf atol=2e-5
    @test pdf(specialized, u) ≈ pdf(generic, u)
    @test Copulas.measure(specialized, [0.15, 0.25], [0.55, 0.65]) ≈
          Copulas.measure(generic, [0.15, 0.25], [0.55, 0.65])

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2])
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6

    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
    @test inverse_rosenblatt(specialized, rosenblatt(specialized, u)) ≈
          inverse_rosenblatt(generic, rosenblatt(generic, u)) atol=2e-6
    @test Copulas.ρ(specialized) ≈ Copulas.ρ(generic) atol=2e-5
    @test Copulas.β(specialized) ≈ Copulas.β(generic)
end

@testset "specialized Gumbel generator agrees with its generic oracle" begin
    θ = 1.5
    generic = PowerExponentialOracleGenerator(θ)
    specialized = Copulas.GumbelGenerator(θ)
    for t in (0.2, 0.7, 1.4)
        p = Copulas.ϕ(generic, t)
        @test Copulas.ϕ(specialized, t) ≈ p
        @test Copulas.ϕ⁻¹(specialized, p) ≈ Copulas.ϕ⁻¹(generic, p)
        @test Copulas.ϕ⁽¹⁾(specialized, t) ≈ Copulas.ϕ⁽¹⁾(generic, t)
        @test Copulas.ϕ⁽ᵏ⁾(specialized, 2, t) ≈
              Copulas.ϕ⁽ᵏ⁾(generic, 2, t)
        @test Copulas.ϕ⁻¹⁽¹⁾(specialized, p) ≈ Copulas.ϕ⁻¹⁽¹⁾(generic, p)
    end
end

@testset "specialized logistic tail agrees with its generic oracle" begin
    θ = 1.5
    generic_tail = LogisticOracleTail(θ)
    specialized_tail = Copulas.LogTail(θ)
    x = [0.4, 0.7]
    weight = Tuple(x ./ sum(x))
    @test Copulas.ℓ(specialized_tail, x) ≈ Copulas.ℓ(generic_tail, x)
    @test Copulas.A(specialized_tail, weight) ≈ Copulas.A(generic_tail, weight)
    for indices in ((), (1,), (2,), (1, 2))
        @test Copulas.ellpartial(specialized_tail, x, indices) ≈
              Copulas.ellpartial(generic_tail, x, indices) atol=2e-6
    end

    generic = ExtremeValueCopula{2}(generic_tail)
    specialized = LogCopula{2}(θ)
    u = [0.37, 0.68]
    @test cdf(specialized, u) ≈ cdf(generic, u)
    @test pdf(specialized, u) ≈ pdf(generic, u) atol=2e-6

    generic_D = condition(generic, 1, u[1])
    specialized_D = condition(specialized, 1, u[1])
    @test cdf(specialized_D, u[2]) ≈ cdf(generic_D, u[2])
    @test pdf(specialized_D, u[2]) ≈ pdf(generic_D, u[2]) atol=2e-6
    @test quantile(specialized_D, 0.6) ≈ quantile(generic_D, 0.6) atol=2e-6
    @test rosenblatt(specialized, u) ≈ rosenblatt(generic, u)
end


@testset "generic numeric sampler buffers" begin
    C = ClaytonCopula{3}(1.0)
    storage = fill(Float32(NaN), 5, 2)
    buffer = @view storage[2:4, :]
    @test rand!(StableRNG(52), C, buffer) === buffer
    @test all(x -> 0 <= x <= 1, buffer)
    @test all(isnan, storage[[1, 5], :])
    @test_throws DimensionMismatch rand!(StableRNG(52), C, zeros(Float32, 2, 1))
end
