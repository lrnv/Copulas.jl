
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(2, 1.2)) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(2,8)) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,20)) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,7)) end
@testitem "Generic" tags=[:GumbelCopula] setup=[M] begin M.check(GumbelCopula(4,100)) end

@testitem "Boundary test for bivariate Gumbel" begin
    using Distributions
    G = GumbelCopula(2, 2.5)
    @test pdf(G, [0.1,0.0]) == 0.0
    @test pdf(G, [0.0,0.1]) == 0.0
    @test pdf(G, [0.0,0.0]) == 0.0
end