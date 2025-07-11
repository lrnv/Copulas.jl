
@testitem "Gumbel pdf" begin
    using HCubature
    using Distributions

    C = GumbelCopula(2, 2.5)
    d = size(C)[1]
    @test hcubature(x -> pdf(C, x), zeros(d), ones(d))[1] ≈ 1.0
end

@testitem "Gumbel cdf" begin
    using HCubature
    using Distributions

    C = GumbelCopula(2, 2.5)
    d = size(C)[1]
    u = fill(0.5, d)
    @test hcubature(x -> pdf(C, x), zeros(d), u)[1] ≈ cdf(C, u)
end

@testitem "Gumbel generator derivative" begin
    using ForwardDiff
    G = Copulas.GumbelGenerator(1.25)

    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, 1, 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, 2, 10.0)
    # reference value taken from the paper
    @test Copulas.ϕ⁽ᵏ⁾(G, 50, 15.0) ≈ 1057 rtol = 0.01

    @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
end

@testitem "Gumbel rand" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [2.5, 5.0, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = GumbelCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^5)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "Gumbel Rosenblatt" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [2.5, 5.0, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = GumbelCopula(d, θ)
        u = rand(C, 10^5)
        U = rosenblatt(C, u)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.01
            end
        end

        u = rand(C, 10^4)
        U = rosenblatt(C, u)
        @test u ≈ inverse_rosenblatt(C, U)
    end
end

@testitem "Boundary test for bivariate Gumbel" begin
    using Distributions
    G = GumbelCopula(2, 2.5)
    @test pdf(G, [0.1, 0.0]) == 0.0
    @test pdf(G, [0.0, 0.1]) == 0.0
    @test pdf(G, [0.0, 0.0]) == 0.0
end
