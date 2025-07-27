@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,1.0)) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,rand(M.rng))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,0.05)) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,1.0)) end

@testitem "InvGaussian rand" begin
    using StatsBase

    dimensions = [2, 3, 4]
    parameters = [0.05, 0.7, 8]

    for (d, θ) in zip(dimensions, parameters)
        C = InvGaussianCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 1000)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ atol = 0.1
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
        u = rand(C, 1000)
        U = rosenblatt(C, u)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(U[i, :], U[j, :]) ≈ 0.0 atol = 0.1
            end
        end
        @test u ≈ inverse_rosenblatt(C, U)
    end
end
