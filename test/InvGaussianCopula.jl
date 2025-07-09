@testitem "InvGaussian rand" begin
    using StatsBase

    dimensions = [2, 3, 4]
    parameters = [0.05, 0.7, 8]

    for (d, θ) in zip(dimensions, parameters)
        C = InvGaussianCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^6)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ rtol = 0.1
            end
        end
    end
end

@testitem "InvGaussian Rosenblatt" begin
    using StatsBase

    dimensions = [2, 3, 4]
    parameters = [0.05, 0.7, 8]

    for (d, θ) in zip(dimensions, parameters)
        C = InvGaussianCopula(d, θ)
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
