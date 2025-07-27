@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [0.8360692316060747, 0.68704221750134])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.2, [0.3,0.6])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.5, [0.5, 0.2])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(12.29006035397328, [0.7036713552821277, 0.7858058549340399])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(12.29006035397328, [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(2.8130363753722403, [0.3539590866764071, 0.15146985093210463])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(2.8130363753722403, [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1.0, [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(12.29006035397328, [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(2.8130363753722403, [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(1+4*rand(M.rng), [0.0, 0.0])) end
@testitem "Generic" tags=[:AsymLogCopula] setup=[M] begin M.check(AsymLogCopula(10+5*rand(M.rng), [0.0, 0.0])) end
