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
