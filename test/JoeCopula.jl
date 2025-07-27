@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,1.0)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,3)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,Inf)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(3,1.0)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(3,7)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(4,1-log(rand(M.rng)))) end

@testitem "Joe pdf" begin
    using HCubature
    using Distributions

    C = GumbelCopula(2, 1.5)
    d = size(C)[1]
    @test hcubature(x -> pdf(C, x), zeros(d), ones(d))[1] ≈ 1.0
end

@testitem "Joe cdf" begin
    using HCubature
    using Distributions

    C = JoeCopula(2, 1.5)
    d = size(C)[1]
    u = fill(0.5, d)
    @test hcubature(x -> pdf(C, x), zeros(d), u)[1] ≈ cdf(C, u)
end

@testitem "Joe rand" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [1.5, 2.5, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = JoeCopula(d, θ)
        τ = Copulas.τ(C)
        u = rand(C, 10^4)
        for i in 1:(d - 1)
            for j in (i + 1):d
                @test corkendall(u[i, :], u[j, :]) ≈ τ atol = 0.1
            end
        end
    end
end

@testitem "Joe derivatives" begin
    using ForwardDiff
    G = Copulas.JoeGenerator(2.5)

    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽¹⁾(G, 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(1), 10.0)
    @test ForwardDiff.derivative(x -> Copulas.ϕ⁽¹⁾(G, x), 10.0) ≈ Copulas.ϕ⁽ᵏ⁾(G, Val(2), 10.0)

    @test ForwardDiff.derivative(x -> Copulas.ϕ⁻¹(G, x), 0.5) ≈ Copulas.ϕ⁻¹⁽¹⁾(G, 0.5)
end

@testitem "Joe Rosenblatt" begin
    using StatsBase

    dimensions = [2, 3, 5]
    parameters = [1.5, 2.5, 10.0]

    for (d, θ) in zip(dimensions, parameters)
        C = JoeCopula(d, θ)
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

@testitem "Boundary test for bivariate Joe" begin
    using Distributions
    θ = 1.1
    C = JoeCopula(2, θ)

    # Joe copula is zero on all borders and corners of the hypercube.
    # so as soon as there is a zero or a one it should be zero.
    us = [0,1,rand(10)...]
    for u in us
        @test pdf(C, [0, u]) == 0
        @test pdf(C, [u, 0]) == 0
        @test pdf(C, [1, u]) == 0
        @test pdf(C, [u, 1]) == 0
    end
end