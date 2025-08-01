@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,3)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(2,Inf)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(3,1-log(rand(M.rng)))) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(3,7)) end
@testitem "Generic" tags=[:JoeCopula] setup=[M] begin M.check(JoeCopula(4,1-log(rand(M.rng)))) end

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