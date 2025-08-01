@testitem "Generic" tags=[:EmpiricalCopula] setup=[M] begin M.check(EmpiricalCopula(randn(2,10),pseudo_values=false)) end
@testitem "Generic" tags=[:EmpiricalCopula] setup=[M] begin M.check(EmpiricalCopula(randn(2,20),pseudo_values=false)) end
