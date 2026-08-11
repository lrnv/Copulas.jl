using BenchmarkTools
using Copulas
using Distributions
using Random

# Tachometer runs the suite repeatedly for both revisions. One second per target
# is sufficient for stable minima while keeping the complete CI job affordable.
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

const SEED = 23
const SUITE = BenchmarkGroup()

SUITE["sampling"] = BenchmarkGroup()
SUITE["density"] = BenchmarkGroup()
SUITE["data"] = BenchmarkGroup()
SUITE["conditioning"] = BenchmarkGroup()
SUITE["fitting"] = BenchmarkGroup()

# Each model below represents a distinct implementation path rather than every
# possible family/dimension combination.
clayton = ClaytonCopula(5, 2.0)
gumbel = GumbelCopula(5, 2.0)

rho = 0.35
sigma = fill(rho, 10, 10)
for i in axes(sigma, 1)
    sigma[i, i] = 1.0
end
gaussian = GaussianCopula(sigma)

nested = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.5);
    leaves = [1, 2],
    children = [
        ClaytonCopula(2, 3.0),
        ClaytonCopula(2, 2.5),
    ],
)

archimax = ArchimaxCopula(
    2,
    Copulas.ClaytonGenerator(2.0),
    Copulas.GalambosTail(1.5),
)

SUITE["sampling"]["clayton_d5"] = @benchmarkable(
    rand(rng, $clayton, 10_000),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)
SUITE["sampling"]["gaussian_d10"] = @benchmarkable(
    rand(rng, $gaussian, 10_000),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)
SUITE["sampling"]["nested_d6"] = @benchmarkable(
    rand(rng, $nested, 100),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)
SUITE["sampling"]["archimax_d2"] = @benchmarkable(
    rand(rng, $archimax, 2_000),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)

gumbel_points = rand(Xoshiro(SEED + 1), 5, 10_000)
gaussian_points = rand(Xoshiro(SEED + 2), 10, 10_000)
nested_points = rand(Xoshiro(SEED + 3), 6, 2_000)

SUITE["density"]["gumbel_d5"] = @benchmarkable(
    logpdf($gumbel, points),
    setup = (points = $gumbel_points),
    evals = 1,
)
SUITE["density"]["gaussian_d10"] = @benchmarkable(
    logpdf($gaussian, points),
    setup = (points = $gaussian_points),
    evals = 1,
)
SUITE["density"]["nested_d6"] = @benchmarkable(
    logpdf($nested, points),
    setup = (points = $nested_points),
    evals = 1,
)

raw_data = randn(Xoshiro(SEED + 4), 5, 10_000)
checkerboard_data = randn(Xoshiro(SEED + 5), 3, 2_000)
checkerboard = CheckerboardCopula(checkerboard_data; m=20, pseudo_values=false)
checkerboard_points = rand(Xoshiro(SEED + 6), 3, 1_000)

SUITE["data"]["pseudos_5x10000"] = @benchmarkable(
    pseudos(data),
    setup = (data = $raw_data),
    evals = 1,
)
SUITE["data"]["checkerboard_cdf"] = @benchmarkable(
    cdf($checkerboard, points),
    setup = (points = $checkerboard_points),
    evals = 1,
)

rosenblatt_copula = GaussianCopula(5, 0.35)
rosenblatt_points = rand(Xoshiro(SEED + 7), 5, 2_000)
SUITE["conditioning"]["rosenblatt_gaussian_d5"] = @benchmarkable(
    rosenblatt($rosenblatt_copula, points),
    setup = (points = $rosenblatt_points),
    evals = 1,
)

gumbel_fit_data = rand(Xoshiro(SEED + 8), GumbelCopula(2, 2.0), 2_000)
gaussian_fit_data = rand(Xoshiro(SEED + 9), GaussianCopula(3, 0.35), 1_000)
sklar_model = SklarDist(
    ClaytonCopula(3, 2.0),
    (Normal(), LogNormal(0.0, 0.5), Gamma(2.0, 1.0)),
)
sklar_fit_data = rand(Xoshiro(SEED + 10), sklar_model, 1_000)

SUITE["fitting"]["gumbel_itau"] = @benchmarkable(
    fit(GumbelCopula, data; method=:itau),
    setup = (data = $gumbel_fit_data),
    evals = 1,
)
SUITE["fitting"]["gaussian_mle"] = @benchmarkable(
    fit(GaussianCopula, data; method=:mle),
    setup = (data = $gaussian_fit_data),
    evals = 1,
)
SUITE["fitting"]["sklar_ifm"] = @benchmarkable(
    fit(
        SklarDist{ClaytonCopula,Tuple{Normal,LogNormal,Gamma}},
        data;
        sklar_method=:ifm,
        copula_method=:mle,
    ),
    setup = (data = $sklar_fit_data),
    evals = 1,
)
