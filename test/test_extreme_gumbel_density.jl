@testitem "Extreme gumbel density test" begin
    using Distributions
    G = GumbelCopula(2, 244.3966206319112)
    u = [0.969034207932297,0.9638122014293545]
    den = pdf(G, u)
    @test true

    G = GumbelCopula(2, 344.3966206319112)
    u = [0.969034207932297,0.9638122014293545]
    den = pdf(G, u)
    @test true
end