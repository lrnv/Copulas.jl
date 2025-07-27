@testitem "Generic" tags=[:MOCopula] setup=[M] begin M.check(MOCopula(0.5960710257852946, 0.3313524247810329, 0.09653466861970061)) end
@testitem "Generic" tags=[:MOCopula] setup=[M] begin M.check(MOCopula(0.1,0.2,0.3)) end
@testitem "Generic" tags=[:MOCopula] setup=[M] begin M.check(MOCopula(0.5, 0.5, 0.5)) end
@testitem "Generic" tags=[:MOCopula] setup=[M] begin M.check(MOCopula(1.0, 1.0, 1.0)) end
@testitem "Generic" tags=[:MOCopula] setup=[M] begin M.check(MOCopula(rand(M.rng), rand(M.rng), rand(M.rng))) end
