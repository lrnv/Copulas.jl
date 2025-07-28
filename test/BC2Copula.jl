@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(1.0, 0.0)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(0.5, 0.3)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(0.5, 0.5)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(0.5516353577049822, 0.33689370624999193)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(0.7,0.3)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(1/2,1/2)) end
@testitem "Generic" tags=[:BC2Copula] setup=[M] begin M.check(BC2Copula(rand(M.rng), rand(M.rng))) end
