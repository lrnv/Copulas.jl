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
SUITE["cdf"] = BenchmarkGroup()
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
student = TCopula(4, [1.0 0.5; 0.5 1.0])
bb1 = BB1Copula(2, 1.2, 1.5)
galambos = GalambosCopula(2, 1.5)

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
SUITE["sampling"]["student_d2"] = @benchmarkable(
    rand(rng, $student, 10_000),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)
SUITE["sampling"]["galambos_d2"] = @benchmarkable(
    rand(rng, $galambos, 10_000),
    setup = (rng = Xoshiro(SEED)),
    evals = 1,
)

gumbel_points = rand(Xoshiro(SEED + 1), 5, 10_000)
gaussian_points = rand(Xoshiro(SEED + 2), 10, 10_000)
nested_points = rand(Xoshiro(SEED + 3), 6, 2_000)
pair_points = clamp.(rand(Xoshiro(SEED + 11), 2, 10_000), 1e-6, 1 - 1e-6)

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
SUITE["density"]["bb1_d2"] = @benchmarkable(
    logpdf($bb1, points),
    setup = (points = $pair_points),
    evals = 1,
)
SUITE["density"]["galambos_d2"] = @benchmarkable(
    logpdf($galambos, points),
    setup = (points = $pair_points),
    evals = 1,
)

SUITE["cdf"]["bb1_d2"] = @benchmarkable(
    cdf($bb1, points),
    setup = (points = $pair_points),
    evals = 1,
)
SUITE["cdf"]["galambos_d2"] = @benchmarkable(
    cdf($galambos, points),
    setup = (points = $pair_points),
    evals = 1,
)

raw_data = randn(Xoshiro(SEED + 4), 5, 10_000)
checkerboard_data = randn(Xoshiro(SEED + 5), 3, 2_000)
checkerboard = CheckerboardCopula(checkerboard_data; m=20, pseudo_values=false)
checkerboard_points = rand(Xoshiro(SEED + 6), 3, 1_000)
empirical_data = randn(Xoshiro(SEED + 12), 2, 2_000)
empirical = EmpiricalCopula(empirical_data; pseudo_values=false)
beta = BetaCopula(empirical_data)
empirical_points = rand(Xoshiro(SEED + 13), 2, 1_000)

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
SUITE["data"]["empirical_cdf"] = @benchmarkable(
    cdf($empirical, points),
    setup = (points = $empirical_points),
    evals = 1,
)
SUITE["data"]["beta_logpdf"] = @benchmarkable(
    logpdf($beta, points),
    setup = (points = $empirical_points),
    evals = 1,
)

rosenblatt_copula = GaussianCopula(5, 0.35)
rosenblatt_points = rand(Xoshiro(SEED + 7), 5, 2_000)
SUITE["conditioning"]["rosenblatt_gaussian_d5"] = @benchmarkable(
    rosenblatt($rosenblatt_copula, points),
    setup = (points = $rosenblatt_points),
    evals = 1,
)
SUITE["conditioning"]["inverse_rosenblatt_gaussian_d5"] = @benchmarkable(
    inverse_rosenblatt($rosenblatt_copula, points),
    setup = (points = $rosenblatt_points),
    evals = 1,
)

bb1_distortion = condition(bb1, 2, 0.4)
galambos_distortion = condition(galambos, 2, 0.4)
conditional_probabilities = collect(range(1e-5, 1 - 1e-5; length=1_000))

SUITE["conditioning"]["construct_bb1_d2_1000"] = @benchmarkable(
    [condition($bb1, 2, u) for u in probabilities],
    setup = (probabilities = $conditional_probabilities),
    evals = 1,
)
SUITE["conditioning"]["cdf_bb1_d2"] = @benchmarkable(
    cdf($bb1_distortion, probabilities),
    setup = (probabilities = $conditional_probabilities),
    evals = 1,
)
SUITE["conditioning"]["quantile_galambos_d2"] = @benchmarkable(
    quantile($galambos_distortion, probabilities),
    setup = (probabilities = $conditional_probabilities),
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
