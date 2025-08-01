@testitem "Generic" tags=[:TCopula] setup=[M] begin M.check(TCopula(2, [1 0.7; 0.7 1])) end
@testitem "Generic" tags=[:TCopula] setup=[M] begin M.check(TCopula(4, [1 0.5; 0.5 1])) end
@testitem "Generic" tags=[:TCopula] setup=[M] begin M.check(TCopula(20,[1 -0.5; -0.5 1])) end