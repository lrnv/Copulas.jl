@testitem "Generic" tags=[:EmpiricalCopula] setup=[M] begin M.check(EmpiricalCopula(randn(2,100),pseudo_values=false)) end
@testitem "Generic" tags=[:EmpiricalCopula] setup=[M] begin M.check(EmpiricalCopula(randn(2,200),pseudo_values=false)) end
