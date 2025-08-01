@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(0.1, [0.2,0.6])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(0.6129496106778634, [0.820474440393214, 0.22304578643880224])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(11.647356700032505, [0.6195348270893413, 0.4197760589260566])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5.0, [0.8, 0.3])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(8.810168494949659, [0.5987759444612732, 0.5391280234619427])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(10+5*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(10+5*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5+4*rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(5+4*rand(M.rng), [rand(M.rng), rand(M.rng)])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(rand(M.rng), [1.0, 1.0])) end
@testitem "Generic" tags=[:AsymGalambosCopula] setup=[M] begin M.check(AsymGalambosCopula(rand(M.rng), [rand(M.rng), rand(M.rng)])) end
