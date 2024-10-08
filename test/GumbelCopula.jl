@testitem "Extreme gumbel density test" begin
    using Distributions
    G = GumbelCopula(2, 244.3966206319112)
    u = [0.969034207932297, 0.9638122014293545]
    den = pdf(G, u)
    @test true

    G = GumbelCopula(2, 1544.3966206319112)
    u = [0.969034207932297, 0.9638122014293545]
    den = pdf(G, u)
    @test true
end

@testitem "Gumbel Rosenblatt" begin
    using StatsBase

    G = GumbelCopula(2, 244.3966206319112)
    u = rand(G, 10^5)

    U = rosenblatt(G, u)
    @test corkendall(U[1, :], U[1, :]) ≈ 0.0 atol = 0.01
end
