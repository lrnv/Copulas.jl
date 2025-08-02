@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,1.0)) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(2,rand(M.rng))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(3,rand(M.rng))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,-log(rand(M.rng)))) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,0.05)) end
@testitem "Generic" tags=[:InvGaussianCopula] setup=[M] begin M.check(InvGaussianCopula(4,1.0)) end

