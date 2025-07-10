@testitem "Joe rand" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [1.5, 2.5, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = JoeCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^5)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "Joe derivatives" begin
    using ForwardDiff
    C = Copulas.JoeGenerator(2.5)

    @test ForwardDiff.derivative(x -> Copulas.ϕ(C, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(C, 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ(C, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(C,1, 10.0)

    @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(C, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(C,2, 10.0)
end

@testitem "Joe Rosenblatt" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [1.5, 2.5, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = JoeCopula(d, θ)
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
