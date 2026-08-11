using Copulas
using DelimitedFiles
using Distributions
using Random

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA = joinpath(ROOT, "data")
mkpath(DATA)

rng = Xoshiro(23)

# Rows are observations, matching the on-disk and R conventions. Julia runners
# transpose them once, before timing, into Copulas.jl's d x n convention.
uniform_points = rand(rng, 10_000, 10) .* 0.998 .+ 0.001
raw_data = randn(rng, 10_000, 5)
gumbel_fit = permutedims(rand(rng, GumbelCopula(2, 2.0), 2_000))
gaussian_fit = permutedims(rand(rng, GaussianCopula(3, 0.35), 1_000))

writedlm(joinpath(DATA, "uniform_points.csv"), uniform_points, ',')
writedlm(joinpath(DATA, "raw_data.csv"), raw_data, ',')
writedlm(joinpath(DATA, "gumbel_fit.csv"), gumbel_fit, ',')
writedlm(joinpath(DATA, "gaussian_fit.csv"), gaussian_fit, ',')

println("Wrote deterministic cross-language fixtures to $DATA")
