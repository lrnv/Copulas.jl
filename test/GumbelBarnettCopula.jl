@testitem "GumbelBarnett generator derivative" begin
    using ForwardDiff
    G = Copulas.GumbelBarnettGenerator(0.5)

    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(1), 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(2), 10.0)

    @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
end

@testitem "GumbelBarnett rand" begin
    using StatsBase

    dimensions = [2]
    parameters = [0.5]

    for (d, θ) in zip(dimensions, parameters)
        C = GumbelBarnettCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^5)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "GumbelBarnett Rosenblatt" begin
    using StatsBase

    dimensions = [2]
    parameters = [0.5]

    for (d, θ) in zip(dimensions, parameters)
        C = GumbelBarnettCopula(d, θ)
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
