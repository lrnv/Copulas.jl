using BenchmarkTools
using Copulas
using Distributions
using Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

const SEED = 23
const SUITE = BenchmarkGroup()

function bench_sampling(C, n)
    return @benchmarkable(
        rand(rng, $C, $n),
        setup=(rng = Xoshiro(SEED)),
        evals=1,
    )
end

function bench_logpdf(C, points)
    return @benchmarkable logpdf($C, $points) evals=1
end

function bench_cdf(C, points)
    return @benchmarkable cdf($C, $points) evals=1
end

function bench_rosenblatt(C, points)
    return @benchmarkable rosenblatt($C, $points) evals=1
end

function bench_inverse_rosenblatt(C, points)
    return @benchmarkable inverse_rosenblatt($C, $points) evals=1
end

function bench_condition(C, j, values)
    return @benchmarkable(
        [condition($C, $j, u) for u in $values],
        evals=1,
    )
end

function bench_conditional_cdf(C, j, base, points)
    D = condition(C, j, base)
    return @benchmarkable cdf($D, $points) evals=1
end

function bench_conditional_quantile(C, j, base, probabilities)
    D = condition(C, j, base)
    return @benchmarkable quantile($D, $probabilities) evals=1
end

function bench_pseudos(data)
    return @benchmarkable pseudos($data) evals=1
end

function bench_fitting(model, data; kwargs...)
    workload = () -> fit(model, data; kwargs...)
    return @benchmarkable $workload() evals=1
end

SUITE["sampling"] = BenchmarkGroup()
SUITE["density"] = BenchmarkGroup()
SUITE["cdf"] = BenchmarkGroup()
SUITE["data"] = BenchmarkGroup()
SUITE["conditioning"] = BenchmarkGroup()
SUITE["fitting"] = BenchmarkGroup()

# Representative models: each one exercises a distinct implementation path.
clayton = ClaytonCopula{5}(2.0)
gumbel = GumbelCopula{5}(2.0)

rho = 0.35
sigma = fill(rho, 10, 10)
for i in axes(sigma, 1)
    sigma[i, i] = 1.0
end
gaussian = GaussianCopula{10}(sigma)

nested = NestedArchimedeanCopula{6}(Copulas.ClaytonGenerator(1.5);
    leaves=[1, 2],
    children=[ClaytonCopula{2}(3.0), ClaytonCopula{2}(2.5)],
)
archimax = ArchimaxCopula{2}(Copulas.ClaytonGenerator(2.0), Copulas.GalambosTail(1.5))
student = TCopula{2}(4, [1.0 0.5; 0.5 1.0])
bb1 = BB1Copula{2}(1.2, 1.5)
galambos = GalambosCopula{2}(1.5)

SUITE["sampling"]["clayton_d5"] = bench_sampling(clayton, 10_000)
SUITE["sampling"]["gaussian_d10"] = bench_sampling(gaussian, 10_000)
SUITE["sampling"]["nested_d6"] = bench_sampling(nested, 100)
SUITE["sampling"]["archimax_d2"] = bench_sampling(archimax, 2_000)
SUITE["sampling"]["student_d2"] = bench_sampling(student, 10_000)
SUITE["sampling"]["galambos_d2"] = bench_sampling(galambos, 10_000)

gumbel_points = rand(Xoshiro(SEED + 1), 5, 10_000)
gaussian_points = rand(Xoshiro(SEED + 2), 10, 10_000)
nested_points = rand(Xoshiro(SEED + 3), 6, 2_000)
pair_points = clamp.(rand(Xoshiro(SEED + 11), 2, 10_000), 1e-6, 1 - 1e-6)

SUITE["density"]["gumbel_d5"] = bench_logpdf(gumbel, gumbel_points)
SUITE["density"]["gaussian_d10"] = bench_logpdf(gaussian, gaussian_points)
SUITE["density"]["nested_d6"] = bench_logpdf(nested, nested_points)
SUITE["density"]["bb1_d2"] = bench_logpdf(bb1, pair_points)
SUITE["density"]["galambos_d2"] = bench_logpdf(galambos, pair_points)

SUITE["cdf"]["bb1_d2"] = bench_cdf(bb1, pair_points)
SUITE["cdf"]["galambos_d2"] = bench_cdf(galambos, pair_points)

raw_data = randn(Xoshiro(SEED + 4), 5, 10_000)
checkerboard_data = randn(Xoshiro(SEED + 5), 3, 2_000)
checkerboard = CheckerboardCopula{3}(checkerboard_data; m=20, pseudo_values=false)
checkerboard_points = rand(Xoshiro(SEED + 6), 3, 1_000)
empirical_data = randn(Xoshiro(SEED + 12), 2, 2_000)
empirical = EmpiricalCopula{2}(empirical_data; pseudo_values=false)
beta = BetaCopula{2}(empirical_data)
empirical_points = rand(Xoshiro(SEED + 13), 2, 1_000)

SUITE["data"]["pseudos_5x10000"] = bench_pseudos(raw_data)
SUITE["data"]["checkerboard_cdf"] = bench_cdf(checkerboard, checkerboard_points)
SUITE["data"]["empirical_cdf"] = bench_cdf(empirical, empirical_points)
SUITE["data"]["beta_logpdf"] = bench_logpdf(beta, empirical_points)

rosenblatt_copula = GaussianCopula{5}(0.35)
rosenblatt_points = rand(Xoshiro(SEED + 7), 5, 2_000)
SUITE["conditioning"]["rosenblatt_gaussian_d5"] =
    bench_rosenblatt(rosenblatt_copula, rosenblatt_points)
SUITE["conditioning"]["inverse_rosenblatt_gaussian_d5"] =
    bench_inverse_rosenblatt(rosenblatt_copula, rosenblatt_points)

conditional_probabilities = collect(range(1e-5, 1 - 1e-5; length=1_000))
SUITE["conditioning"]["construct_bb1_d2_1000"] =
    bench_condition(bb1, 2, conditional_probabilities)
SUITE["conditioning"]["cdf_bb1_d2"] =
    bench_conditional_cdf(bb1, 2, 0.4, conditional_probabilities)
SUITE["conditioning"]["quantile_galambos_d2"] =
    bench_conditional_quantile(galambos, 2, 0.4, conditional_probabilities)

gumbel_fit_data = rand(Xoshiro(SEED + 8), GumbelCopula{2}(2.0), 2_000)
gaussian_fit_data = rand(Xoshiro(SEED + 9), GaussianCopula{3}(0.35), 1_000)
sklar_model = SklarDist(
    ClaytonCopula{3}(2.0),
    (Normal(), LogNormal(0.0, 0.5), Gamma(2.0, 1.0)),
)
sklar_fit_data = rand(Xoshiro(SEED + 10), sklar_model, 1_000)

SUITE["fitting"]["gumbel_itau"] = bench_fitting(GumbelCopula, gumbel_fit_data; method=:itau)
SUITE["fitting"]["gaussian_mle"] = bench_fitting(GaussianCopula, gaussian_fit_data; method=:mle)
SUITE["fitting"]["sklar_ifm"] = bench_fitting(
    SklarDist{ClaytonCopula,Tuple{Normal,LogNormal,Gamma}},
    sklar_fit_data;
    sklar_method=:ifm,
    copula_method=:mle,
)
