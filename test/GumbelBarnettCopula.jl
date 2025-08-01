@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(2,1.0)) end
@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,0.1)) end
@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,0.35)) end
@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(3,rand(M.rng)*0.38)) end
@testitem "Generic" tags=[:GumbelBarnettCopula] setup=[M] begin M.check(GumbelBarnettCopula(4,0.2)) end