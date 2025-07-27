
@testitem "Generic" setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,-rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,0.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,0.7)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,-rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,0.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,0.6)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,rand(M.rng))) end

@testitem "AMH derivatives" begin
    using ForwardDiff

    for G in [Copulas.AMHGenerator(0.5), Copulas.FrankGenerator(-0.5)]
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(1), 10.0)
        @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈
            Copulas.ϕ⁽ᵏ⁾(G, Val(2), 10.0)

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
        u = rand(C, 10^4)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ atol = 0.1
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
