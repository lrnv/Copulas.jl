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

# Rectangle probabilities: `measure` over many boxes, and the interval
# conditioning primitive on the same boxes (with no point coordinate it must
# dispatch to `measure`, so the two entries should stay level; the mixed
# entry differentiates the box volume in one coordinate and has no `measure`
# counterpart).
function bench_measure(C, boxes)
    return @benchmarkable(
        [Copulas.measure($C, lo, hi) for (lo, hi) in $boxes],
        evals=1,
    )
end

function bench_box_volume(C, boxes)
    bs = ntuple(identity, length(C))
    return @benchmarkable(
        [Copulas._box_partial_cdf($C, (), (), $bs, (), (), lo, hi) for (lo, hi) in $boxes],
        evals=1,
    )
end

function bench_box_partial(C, boxes)
    d = length(C)
    bs = ntuple(k -> k + 1, d - 1)
    return @benchmarkable(
        [Copulas._box_partial_cdf($C, (), (1,), $bs, (), (0.4,), Base.tail(lo), Base.tail(hi))
         for (lo, hi) in $boxes],
        evals=1,
    )
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
SUITE["measure"] = BenchmarkGroup()
SUITE["fitting"] = BenchmarkGroup()
SUITE["inference"] = BenchmarkGroup()

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
SUITE["cdf"]["student_d2"] = bench_cdf(student, pair_points[:, 1:10])

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
SUITE["conditioning"]["cdf_student_d2"] =
    bench_conditional_cdf(student, 2, 0.4, conditional_probabilities)
SUITE["conditioning"]["quantile_student_d2"] =
    bench_conditional_quantile(student, 2, 0.4, conditional_probabilities)

function random_boxes(rng, d, n)
    return [
        (ntuple(_ -> rand(rng) / 2, d), ntuple(_ -> 0.5 + rand(rng) / 2, d))
        for _ in 1:n
    ]
end
clayton3 = ClaytonCopula{3}(2.0)
gumbel3 = GumbelCopula{3}(1.6)
boxes2 = random_boxes(Xoshiro(SEED + 14), 2, 1_000)
boxes3 = random_boxes(Xoshiro(SEED + 15), 3, 1_000)
boxes5 = random_boxes(Xoshiro(SEED + 16), 5, 200)
SUITE["measure"]["clayton_d2"] = bench_measure(ClaytonCopula{2}(2.0), boxes2)
SUITE["measure"]["clayton_d3"] = bench_measure(clayton3, boxes3)
SUITE["measure"]["gumbel_d3"] = bench_measure(gumbel3, boxes3)
SUITE["measure"]["clayton_d5"] = bench_measure(clayton, boxes5)
SUITE["measure"]["box_volume_clayton_d3"] = bench_box_volume(clayton3, boxes3)
SUITE["measure"]["box_volume_clayton_d5"] = bench_box_volume(clayton, boxes5)
SUITE["measure"]["box_partial_clayton_d3"] = bench_box_partial(clayton3, boxes3)
SUITE["measure"]["box_partial_gumbel_d3"] = bench_box_partial(gumbel3, boxes3)

# Dimension-specialized fitting benchmarks (#443).
# Keep derived measures disabled below so these workloads isolate the
# fitting/optimization hot path itself.
clayton_fit_d2 =    rand(Xoshiro(SEED + 15), ClaytonCopula{2}(2.0), 1_000)
clayton_fit_d5 =    rand(Xoshiro(SEED + 16), ClaytonCopula{5}(2.0), 1_000)
gumbel_mle_fit_d2 = rand(Xoshiro(SEED + 17), GumbelCopula{2}(2.0), 1_000)
gumbel_mle_fit_d5 = rand(Xoshiro(SEED + 18), GumbelCopula{5}(2.0), 1_000)
bb1_fit_d2 =        rand(Xoshiro(SEED + 19), BB1Copula{2}(1.2, 1.5), 1_000)
bb1_fit_d5 =        rand(Xoshiro(SEED + 20), BB1Copula{5}(1.2, 1.5), 1_000)
gumbel_fit_data =   rand(Xoshiro(SEED + 8),  GumbelCopula{2}(2.0), 2_000)
gaussian_fit_data = rand(Xoshiro(SEED + 9),  GaussianCopula{3}(0.35), 1_000)
sklar_fit_data =    rand(Xoshiro(SEED + 10), SklarDist(
    ClaytonCopula{3}(2.0),
    (Normal(), LogNormal(0.0, 0.5), Gamma(2.0, 1.0)),
), 1_000)


SUITE["fitting"]["clayton_mle_d2"] = bench_fitting(
    ClaytonCopula,
    clayton_fit_d2;
    method=:mle,
)

SUITE["fitting"]["clayton_mle_d5"] = bench_fitting(
    ClaytonCopula,
    clayton_fit_d5;
    method=:mle,
)

SUITE["fitting"]["gumbel_mle_d2"] = bench_fitting(
    GumbelCopula,
    gumbel_mle_fit_d2;
    method=:mle,
)

SUITE["fitting"]["gumbel_mle_d5"] = bench_fitting(
    GumbelCopula,
    gumbel_mle_fit_d5;
    method=:mle,
)

SUITE["fitting"]["bb1_mle_d2"] = bench_fitting(
    BB1Copula,
    bb1_fit_d2;
    method=:mle,
)

SUITE["fitting"]["bb1_mle_d5"] = bench_fitting(
    BB1Copula,
    bb1_fit_d5;
    method=:mle,
)
SUITE["fitting"]["gumbel_itau"] = bench_fitting(
    GumbelCopula,
    gumbel_fit_data;
    method=:itau,
)
SUITE["fitting"]["gaussian_mle"] = bench_fitting(
    GaussianCopula, 
    gaussian_fit_data; 
    method=:mle
)
SUITE["fitting"]["gaussian_itau"] = bench_fitting(
    GaussianCopula,
    gaussian_fit_data;
    method=:itau
)
student_fit_data = rand(Xoshiro(SEED + 14), student, 2_000)
SUITE["fitting"]["student_rank_matching"] =
    bench_fitting(TCopula{2}, student_fit_data; method=:itau_irho,)
SUITE["fitting"]["student_itau"] =
    bench_fitting(TCopula{2}, student_fit_data; method=:itau,)
SUITE["fitting"]["sklar_ifm"] = bench_fitting(
    SklarDist{ClaytonCopula,Tuple{Normal,LogNormal,Gamma}},
    sklar_fit_data;
    sklar_method=:ifm,
    copula_method=:mle,
)


clayton_mle_model = fit(CopulaModel, ClaytonCopula, clayton_fit_d2;
                        method=:mle,)
gumbel_itau_model = fit(CopulaModel, GumbelCopula, gumbel_mle_fit_d2;
                        method=:itau,)
SUITE["inference"]["clayton_mle_hessian_d2"] =
    @benchmarkable infer($clayton_mle_model; method=:hessian) evals=1
SUITE["inference"]["gumbel_itau_godambe_d2"] =
    @benchmarkable infer($gumbel_itau_model; method=:godambe) evals=1
