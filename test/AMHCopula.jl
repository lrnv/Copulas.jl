@testitem "AMH derivatives" begin
    using ForwardDiff

    for G in [Copulas.AMHGenerator(0.5), Copulas.FrankGenerator(-0.5)]
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, 1, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈
            Copulas.ϕ⁽ᵏ⁾(G, 2, 10.0)

        @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
    end
end

@testitem "AMH pdf" begin
    using HCubature
    using Distributions

    for C in [AMHCopula(2, 0.5), AMHCopula(2, -0.5)]
        d = size(C)[1]
        @test hcubature(x -> pdf(C, x), zeros(d), ones(d))[1] ≈ 1.0
    end
end

@testitem "AMH cdf" begin
    using HCubature
    using Distributions

    for C in [AMHCopula(2, 0.5), AMHCopula(2, -0.5)]
        d = size(C)[1]
        u = fill(0.5, d)
        @test hcubature(x -> pdf(C, x), zeros(d), u)[1] ≈ cdf(C, u)
    end
end

@testitem "AMH rand" begin
    using StatsBase

    dimensions = [2, 2, 3, 5]
    parameters = [0.5, -0.5, -0.3, 0.8]

    for (d, θ) in zip(dimensions, parameters)
        C = AMHCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^5)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "AMH Rosenblatt" begin
    using StatsBase

    dimensions = [2, 2, 3, 5]
    parameters = [0.5, -0.5, -0.3, 0.8]

    for (d, θ) in zip(dimensions, parameters)
        C = AMHCopula(d, θ)
        u = rand(C, 10^6)
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
