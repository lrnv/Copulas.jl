@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.0)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.1)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.3437537135972244)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.7103550345192344)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(0.8)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(1.0)) end
@testitem "Generic" tags=[:CuadrasAugeCopula] setup=[M] begin M.check(CuadrasAugeCopula(rand(M.rng))) end
