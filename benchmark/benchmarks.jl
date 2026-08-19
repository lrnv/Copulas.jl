using BenchmarkTools
using Copulas
using Distributions
using Random

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

const SEED = 23
const SUITE = BenchmarkGroup()

bench_name(C) = lowercase(replace(string(nameof(typeof(C))), "Copula" => "")) * "_d$(length(C))"
function bench_group!(suite, operation)
    haskey(suite, operation) || (suite[operation] = BenchmarkGroup())
    return suite[operation]
end

function bench_sampling!(suite, C, n)
    bench_group!(suite, "sampling")[bench_name(C)] = @benchmarkable(
        rand(rng, $C, $n),
        setup=(rng = Xoshiro(SEED)),
        evals=1,
    )
end

function bench_logpdf!(suite, C, points)
    bench_group!(suite, "logpdf")[bench_name(C)] =
        @benchmarkable logpdf($C, $points) evals=1
end

function bench_cdf!(suite, C, points)
    bench_group!(suite, "cdf")[bench_name(C)] =
        @benchmarkable cdf($C, $points) evals=1
end

function bench_rosenblatt!(suite, C, points)
    bench_group!(suite, "rosenblatt")[bench_name(C)] =
        @benchmarkable rosenblatt($C, $points) evals=1
end

function bench_inverse_rosenblatt!(suite, C, points)
    bench_group!(suite, "inverse_rosenblatt")[bench_name(C)] =
        @benchmarkable inverse_rosenblatt($C, $points) evals=1
end

function bench_condition!(suite, C, j, values)
    bench_group!(suite, "condition")[bench_name(C)] = @benchmarkable(
        [condition($C, $j, u) for u in $values],
        evals=1,
    )
end

function bench_conditional_cdf!(suite, C, j, base, points)
    D = condition(C, j, base)
    bench_group!(suite, "conditional_cdf")[bench_name(C)] =
        @benchmarkable cdf($D, $points) evals=1
end

function bench_conditional_quantile!(suite, C, j, base, probabilities)
    D = condition(C, j, base)
    bench_group!(suite, "conditional_quantile")[bench_name(C)] =
        @benchmarkable quantile($D, $probabilities) evals=1
end

function bench_pseudos!(suite, data)
    name = "matrix_$(size(data, 1))x$(size(data, 2))"
    bench_group!(suite, "pseudos")[name] = @benchmarkable pseudos($data) evals=1
end

function bench_fitting!(suite, name, model, data; kwargs...)
    workload = () -> fit(model, data; kwargs...)
    bench_group!(suite, "fitting")[name] = @benchmarkable $workload() evals=1
end

# Representative models: each one exercises a distinct implementation path.
clayton = ClaytonCopula(5, 2.0)
gumbel = GumbelCopula(5, 2.0)

rho = 0.35
sigma = fill(rho, 10, 10)
for i in axes(sigma, 1)
    sigma[i, i] = 1.0
end
gaussian = GaussianCopula(sigma)

nested = NestedArchimedeanCopula(Copulas.ClaytonGenerator(1.5);
    leaves=[1, 2],
    children=[ClaytonCopula(2, 3.0), ClaytonCopula(2, 2.5)],
)
archimax = ArchimaxCopula(2, Copulas.ClaytonGenerator(2.0), Copulas.GalambosTail(1.5))
student = TCopula(4, [1.0 0.5; 0.5 1.0])
bb1 = BB1Copula(2, 1.2, 1.5)
galambos = GalambosCopula(2, 1.5)

bench_sampling!(SUITE, clayton, 10_000)
bench_sampling!(SUITE, gaussian, 10_000)
bench_sampling!(SUITE, nested, 100)
bench_sampling!(SUITE, archimax, 2_000)
bench_sampling!(SUITE, student, 10_000)
bench_sampling!(SUITE, galambos, 10_000)

gumbel_points = rand(Xoshiro(SEED + 1), 5, 10_000)
gaussian_points = rand(Xoshiro(SEED + 2), 10, 10_000)
nested_points = rand(Xoshiro(SEED + 3), 6, 2_000)
pair_points = clamp.(rand(Xoshiro(SEED + 11), 2, 10_000), 1e-6, 1 - 1e-6)

bench_logpdf!(SUITE, gumbel, gumbel_points)
bench_logpdf!(SUITE, gaussian, gaussian_points)
bench_logpdf!(SUITE, nested, nested_points)
bench_logpdf!(SUITE, bb1, pair_points)
bench_logpdf!(SUITE, galambos, pair_points)

bench_cdf!(SUITE, bb1, pair_points)
bench_cdf!(SUITE, galambos, pair_points)

raw_data = randn(Xoshiro(SEED + 4), 5, 10_000)
checkerboard_data = randn(Xoshiro(SEED + 5), 3, 2_000)
checkerboard = CheckerboardCopula(checkerboard_data; m=20, pseudo_values=false)
checkerboard_points = rand(Xoshiro(SEED + 6), 3, 1_000)
empirical_data = randn(Xoshiro(SEED + 12), 2, 2_000)
empirical = EmpiricalCopula(empirical_data; pseudo_values=false)
beta = BetaCopula(empirical_data)
empirical_points = rand(Xoshiro(SEED + 13), 2, 1_000)

bench_pseudos!(SUITE, raw_data)
bench_cdf!(SUITE, checkerboard, checkerboard_points)
bench_cdf!(SUITE, empirical, empirical_points)
bench_logpdf!(SUITE, beta, empirical_points)

rosenblatt_copula = GaussianCopula(5, 0.35)
rosenblatt_points = rand(Xoshiro(SEED + 7), 5, 2_000)
bench_rosenblatt!(SUITE, rosenblatt_copula, rosenblatt_points)
bench_inverse_rosenblatt!(SUITE, rosenblatt_copula, rosenblatt_points)

conditional_probabilities = collect(range(1e-5, 1 - 1e-5; length=1_000))
bench_condition!(SUITE, bb1, 2, conditional_probabilities)
bench_conditional_cdf!(SUITE, bb1, 2, 0.4, conditional_probabilities)
bench_conditional_quantile!(SUITE, galambos, 2, 0.4, conditional_probabilities)

gumbel_fit_data = rand(Xoshiro(SEED + 8), GumbelCopula(2, 2.0), 2_000)
gaussian_fit_data = rand(Xoshiro(SEED + 9), GaussianCopula(3, 0.35), 1_000)
sklar_model = SklarDist(
    ClaytonCopula(3, 2.0),
    (Normal(), LogNormal(0.0, 0.5), Gamma(2.0, 1.0)),
)
sklar_fit_data = rand(Xoshiro(SEED + 10), sklar_model, 1_000)

bench_fitting!(SUITE, "gumbel_itau", GumbelCopula, gumbel_fit_data; method=:itau)
bench_fitting!(SUITE, "gaussian_mle", GaussianCopula, gaussian_fit_data; method=:mle)
bench_fitting!(
    SUITE,
    "sklar_ifm",
    SklarDist{ClaytonCopula,Tuple{Normal,LogNormal,Gamma}},
    sklar_fit_data;
    sklar_method=:ifm,
    copula_method=:mle,
)
