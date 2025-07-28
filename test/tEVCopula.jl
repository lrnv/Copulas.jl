@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(10.0, 1.0)) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(3.0, 0.0)) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(4.0, 0.5)) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(5.0, -0.5)) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(5.466564460573727, -0.6566645244416698)) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(4+6*rand(M.rng), -0.9+1.9*rand(M.rng))) end
@testitem "Generic" tags=[:tEVCopula] setup=[M] begin M.check(tEVCopula(2.0, 0.5)) end
