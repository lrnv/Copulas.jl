@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.2, [0.3,0.6])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.5, [0.5, 0.2])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
