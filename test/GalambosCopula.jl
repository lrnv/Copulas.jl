@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(0.0)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(0.3)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(120)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(20)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(210)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(4.3)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(80)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(Inf)) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(5+5*rand(M.rng))) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(rand(M.rng))) end
@testitem "Generic" tags=[:GalambosCopula] setup=[M] begin M.check(GalambosCopula(1+4*rand(M.rng))) end

@testitem "Extreme frank density test" begin
    rand(GalambosCopula(19.7), 1000)
    rand(GalambosCopula(210.0), 1000)
    @test true
end