@testitem "Generic" setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,-rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,0.7)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,-1.0)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,-rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,0.6)) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(3,rand(M.rng))) end
@testitem "Generic" tags=[:AMHCopula] setup=[M] begin M.check(AMHCopula(4,-0.3)) end