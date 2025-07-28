@testitem "Generic" tags=[:AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula( [0.2, 0.4])) end
@testitem "Generic" tags=[:AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.0, 0.0])) end
@testitem "Generic" tags=[:AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.1, 0.2])) end
@testitem "Generic" tags=[:AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.1,0.2])) end
@testitem "Generic" tags=[:AsymMixedCopula] setup=[M] begin M.check(AsymMixedCopula([0.2, 0.4])) end
