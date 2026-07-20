# Tests for NestedArchimedeanCopula: the nested-Archimedean density and its
# lower-tail partial-observation likelihood as an EMERGENT capability of the standard
# condition + subsetdims framework (Yang & Li, arXiv:2605.23134).
#
# Coverage:
#   1. Flat dispatch — a leaves-only declaration returns the native
#      ArchimedeanCopula and gives a bit-for-bit identical logpdf.
#   2. Uncensored density vs an INDEPENDENT reference: the nested CDF assembled
#      directly from the generators, mixed-partial-differentiated by nested
#      ForwardDiff. This shares no code path with the Faà di Bruno recursion.
#   3. Uncensored density vs an EXTERNAL reference (acopula log-likelihoods,
#      committed in test/data/nested/).
#   4. Lower-tail partial observation via the gist recipe
#      `logpdf(subsetdims(X,O),x_O) + logcdf_or_cdf(condition(X,O,x_O),x_C)`,
#      checked against the SAME independent ForwardDiff references (mixed partial
#      over the observed dims), incl. the bivariate Clayton closed form, plus a
#      fast-path type probe (condition returns NestedDistortion) and the
#      multi-unobserved generic ForwardDiff fallback.
#   4b. SklarDist lower-tail likelihood via condition + subsetdims on the data scale,
#       incl. the contrast with Distributions.censored.
#   5. Heterogeneous (mixed-family) and arbitrary-depth nesting build & are finite.
#   6. Constructor errors (bad tiling) and a fit/smoke usage of logpdf.

using Test, Copulas, Distributions, ForwardDiff, DelimitedFiles, Random
import StatsBase
import Copulas: Generator, ϕ, ϕ⁻¹, ϕ⁻¹⁽¹⁾, ϕ⁽ᵏ⁾
import Copulas: ClaytonGenerator, GumbelGenerator, FrankGenerator, JoeGenerator
import Copulas: NestedDistortion, subsetdims, condition, _censored_copula_logpdf

# Seeded RNG, matching runtests' `StableRNG(123)` when StableRNGs is on the path
# (the package test environment); falls back to a seeded Xoshiro so this file
# also runs standalone via `--project=.`. The value is invariant either way:
# every draw feeds BOTH sides of each equality.
const _NEST_RNG = try
    @eval import StableRNGs
    StableRNGs.StableRNG(123)
catch
    Random.Xoshiro(123)
end

# ---------------------------------------------------------------------------
# Independent reference: nested-Archimedean CDF assembled straight from the
# generators, mixed-partial over the observed dims by nested ForwardDiff. No
# Faà di Bruno / partial-Bell code is touched.
# ---------------------------------------------------------------------------
struct RefSpec
    G::Any
    leaves::Vector{Tuple{BigFloat,Bool}}   # (value, censored)
    children::Vector{RefSpec}
end
RefSpec(G, leaves) = RefSpec(G, leaves, RefSpec[])

function ref_vars(s::RefSpec)
    vars = Tuple{BigFloat,Bool}[]
    walk(n) = (for lf in n.leaves; push!(vars, lf); end; for c in n.children; walk(c); end)
    walk(s); return vars
end

# CDF in depth-first variable order; generic in eltype(x) for ForwardDiff Duals.
function ref_cdf(s::RefSpec, x::AbstractVector)
    idx = Ref(0)
    function eval_node(node)::eltype(x)
        acc = zero(eltype(x))
        for _ in node.leaves
            idx[] += 1; acc += ϕ⁻¹(node.G, x[idx[]])
        end
        for ch in node.children
            acc += ϕ⁻¹(node.G, eval_node(ch))
        end
        return ϕ(node.G, acc)
    end
    return eval_node(s)
end

# Nested mixed partial of f over index set `obs`, at u (BigFloat).
function mixed_partial(f, u::Vector{BigFloat}, obs::Vector{Int})
    function rec(k::Int, uu::AbstractVector)
        k > length(obs) && return f(uu)
        i = obs[k]
        return ForwardDiff.derivative(uu[i]) do t
            R = promote_type(typeof(t), eltype(uu))
            v = R[j == i ? t : uu[j] for j in eachindex(uu)]
            rec(k + 1, v)
        end
    end
    return rec(1, copy(u))
end

function ref_logpdf(s::RefSpec)
    vars = ref_vars(s)
    u = BigFloat[v[1] for v in vars]
    obs = [i for (i, v) in enumerate(vars) if !v[2]]
    f = x -> ref_cdf(s, x)
    isempty(obs) && return log(abs(f(u)))
    return log(abs(mixed_partial(f, u, obs)))
end

# Express lower-tail partial observation through the STANDARD condition + subsetdims API
# (the "gist recipe"). `δ[i] == true` ⇒ coordinate `i` is unobserved/lower-tail.
#   logL_O = logpdf(subsetdims(C,O), u_O) + logcdf_or_cdf(condition(C,O,u_O), u_C)
# with the degenerate masks handled explicitly (no observed ⇒ log cdf; no
# no lower-tail coordinates ⇒ plain logpdf).
logcdf_or_cdf(D, u::Real) = logcdf(D, u)
logcdf_or_cdf(S, u::AbstractVector) = log(cdf(S, u))
function gist_censored(C, u, δ)
    obs = Tuple(i for i in eachindex(δ) if !δ[i])
    cen = [i for i in eachindex(δ) if δ[i]]
    isempty(cen) && return logpdf(C, u)
    isempty(obs) && return log(cdf(C, u))
    u_obs = length(obs) == 1 ? u[obs[1]] : [u[i] for i in obs]
    u_cen = length(cen) == 1 ? u[cen[1]] : [u[i] for i in cen]
    return logpdf(subsetdims(C, obs), u_obs) +
           logcdf_or_cdf(condition(C, obs, u_obs), u_cen)
end
# Data-scale analogue on a SklarDist (subsetdims + condition push-forwards).
function gist_sklar(S, x, δ)
    obs = Tuple(i for i in eachindex(δ) if !δ[i])
    cen = [i for i in eachindex(δ) if δ[i]]
    isempty(cen) && return logpdf(S, x)
    isempty(obs) && return log(cdf(S, x))
    x_obs = length(obs) == 1 ? x[obs[1]] : [x[i] for i in obs]
    x_cen = length(cen) == 1 ? x[cen[1]] : [x[i] for i in cen]
    return logpdf(subsetdims(S, obs), x_obs) +
           logcdf_or_cdf(condition(S, obs, x_obs), x_cen)
end

const _ACOPULA_CASES = [
    ("clayton_d10_2level", ClaytonGenerator, [5, 5],        1.5, 3.0),
    ("clayton_d20_2level", ClaytonGenerator, [5, 5, 5, 5],  2.0, 4.0),
    ("gumbel_d10",         GumbelGenerator,  [5, 5],        2.0, 5.0),
    ("frank_d10",          FrankGenerator,   [5, 5],        2.0, 5.0),
]

function representative_rows(n::Int, k::Int)
    k >= n && return collect(1:n)
    return unique!(round.(Int, range(1, n; length = k)))
end

function acopula_maxerr(datadir, name, GT, sectors, θroot, θsector; nrows = 12)
    U  = readdlm(joinpath(datadir, name * "_U.csv"), ',')
    ll = vec(readdlm(joinpath(datadir, name * "_acopula_ll.csv"), ','))
    C = NestedArchimedeanCopula(GT(θroot);
            children = [ArchimedeanCopula(s, GT(θsector)) for s in sectors])
    maxerr = 0.0
    for i in representative_rows(size(U, 1), nrows)
        ours = Float64(logpdf(C, big.(U[i, :])))
        maxerr = max(maxerr, abs(ours - ll[i]))
    end
    return maxerr
end

@testset "NestedArchimedeanCopula" begin
    # Local seeded RNG so this file is self-contained standalone AND under
    # runtests.jl (where a `const rng = StableRNG(123)` also exists); every draw
    # feeds both sides of each equality, so the value is invariant.
    rng = _NEST_RNG

    # -----------------------------------------------------------------------
    # 1. Flat dispatch → native ArchimedeanCopula, bit-for-bit logpdf.
    # -----------------------------------------------------------------------
    @testset "flat declaration dispatches to native (bit-for-bit)" begin
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0); leaves = [1, 2, 3])
        @test C isa ArchimedeanCopula{3}
        @test !(C isa NestedArchimedeanCopula)
        native = ClaytonCopula(3, 2.0)
        for _ in 1:5
            u = rand(rng, 3) .* 0.6 .+ 0.2
            @test logpdf(C, u) === logpdf(native, u)
        end
        Cg = NestedArchimedeanCopula(GumbelGenerator(2.5); leaves = [1, 2, 3, 4])
        @test Cg isa ArchimedeanCopula{4}
        ng = GumbelCopula(4, 2.5)
        for _ in 1:5
            u = rand(rng, 4) .* 0.6 .+ 0.2
            @test logpdf(Cg, u) === logpdf(ng, u)
        end
    end

    # -----------------------------------------------------------------------
    # 2. Uncensored density vs the independent ForwardDiff reference.
    # -----------------------------------------------------------------------
    @testset "uncensored density vs independent ForwardDiff reference" begin
        # Same-family Clayton: root(1.5) over two Clayton(3.0) panels (dims 1:2, 3:4).
        C = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                children = [ClaytonCopula(2, 3.0), ClaytonCopula(2, 3.0)])
        for u0 in ([0.25, 0.40, 0.65, 0.80], [0.72, 0.31, 0.58, 0.44])
            u = big.(u0)
            spec = RefSpec(ClaytonGenerator(big(1.5)),
                       Tuple{BigFloat,Bool}[],
                       [RefSpec(ClaytonGenerator(big(3.0)), [(u[1], false), (u[2], false)]),
                        RefSpec(ClaytonGenerator(big(3.0)), [(u[3], false), (u[4], false)])])
            @test logpdf(C, u) ≈ ref_logpdf(spec) atol = 1e-10
        end
        # Heterogeneous: Clayton root over a Gumbel panel + a Frank panel.
        H = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                children = [GumbelCopula(2, 2.0), FrankCopula(2, 3.0)])
        for u0 in ([0.23, 0.47, 0.71, 0.59], [0.76, 0.35, 0.42, 0.68])
            u = big.(u0)
            spec = RefSpec(ClaytonGenerator(big(1.5)),
                       Tuple{BigFloat,Bool}[],
                       [RefSpec(GumbelGenerator(big(2.0)), [(u[1], false), (u[2], false)]),
                        RefSpec(FrankGenerator(big(3.0)),  [(u[3], false), (u[4], false)])])
            @test logpdf(H, u) ≈ ref_logpdf(spec) atol = 1e-10
        end
    end

    # -----------------------------------------------------------------------
    # 3. Uncensored density vs external acopula reference log-likelihoods.
    #    Files in test/data/nested/ : 2-level nesting, equal-size sectors with
    #    a single sector parameter; compared at Float64 tolerance.
    # -----------------------------------------------------------------------
    @testset "uncensored density vs external acopula reference" begin
        datadir = joinpath(@__DIR__, "data", "nested")
        for case in _ACOPULA_CASES
            @test acopula_maxerr(datadir, case...; nrows = 12) < 1e-9
        end
    end

    # -----------------------------------------------------------------------
    # 3b. Edge-composition method (the overloadable `composition_taylor` hook).
    #     The DEFAULT (testsets above) is the direct jet; here we check the two
    #     OTHER shipped paths agree with it per-edge: the implicit App. A.4
    #     solver and the Clayton/Clayton closed-form override. Placed next to the
    #     bit-identity guard of testset 3.
    # -----------------------------------------------------------------------
    @testset "edge-composition method (hook)" begin
        direct   = Copulas.composition_taylor_direct
        implicit = Copulas.composition_taylor_implicit

        # (b) implicit == direct across same-family AND a cross-family edge, at
        #     several depths, with nestable params (inner ≥ outer dependence).
        edges = [
            ("Clayton/Clayton", ClaytonGenerator(2.0), ClaytonGenerator(5.0), 0.3),
            ("Gumbel/Gumbel",   GumbelGenerator(2.0),  GumbelGenerator(4.0),  0.5),
            ("Frank/Frank",     FrankGenerator(2.0),   FrankGenerator(5.0),   0.5),
            ("Joe/Joe",         JoeGenerator(1.5),     JoeGenerator(3.0),     0.4),
            ("Gumbel/Clayton",  GumbelGenerator(2.0),  ClaytonGenerator(5.0), 0.4),  # cross-family
        ]
        for (_, Go, Gi, t0) in edges, d in (2, 4, 6, 8)
            @test maximum(abs.(implicit(Go, Gi, t0, d) .- direct(Go, Gi, t0, d))) < 1e-10
        end

        # (c) Clayton/Clayton closed-form override == BOTH general methods.
        Go, Gi = ClaytonGenerator(2.0), ClaytonGenerator(5.0)
        ov = Copulas.composition_taylor(Go, Gi, 0.3, 6)
        @test maximum(abs.(ov .- direct(Go, Gi, 0.3, 6)))   < 1e-10
        @test maximum(abs.(ov .- implicit(Go, Gi, 0.3, 6))) < 1e-10
        # BigFloat precision flows through the override (and the implicit path).
        @test eltype(Copulas.composition_taylor(ClaytonGenerator(2.0), ClaytonGenerator(5.0), big"0.3", 6)) === BigFloat
        @test eltype(implicit(ClaytonGenerator(2.0), ClaytonGenerator(5.0), big"0.3", 6)) === BigFloat
    end

    # -----------------------------------------------------------------------
    # 4. Lower-tail partial observation via condition + subsetdims (gist recipe).
    #    Reproduces the SAME independent references the old bespoke API checked,
    #    now through the STANDARD API — proving the specialised condition /
    #    subsetdims path is equivalent (the denominator c_O cancels exactly).
    # -----------------------------------------------------------------------
    @testset "lower-tail partial observation via condition + subsetdims (gist recipe)" begin
        # (a) Bivariate Clayton(3), dim 2 lower-tail: gist == closed form ∂C/∂u₁.
        θ = 3.0
        u1 = cdf(Exponential(1.0), 0.5)
        u2 = cdf(Exponential(1.0), 1.0)
        Cbiv = NestedArchimedeanCopula(ClaytonGenerator(2.0);   # outer irrelevant: single panel
                   children = [ClaytonCopula(2, 3.0)])
        dC_du1 = u1^(-θ - 1) * (u1^(-θ) + u2^(-θ) - 1)^(-(1 / θ + 1))
        @test gist_censored(Cbiv, [u1, u2], [false, true]) ≈ log(dC_du1) atol = 1e-9
        # Explicit decomposition: subsetdims p==1 ⇒ Uniform (base term 0), so the
        # whole value is logcdf(condition) ⇒ den cancels.
        @test logpdf(subsetdims(Cbiv, (1,)), u1) +
              logcdf(condition(Cbiv, (1,), u1), u2) ≈ log(dC_du1) atol = 1e-9
        @test logpdf(subsetdims(Cbiv, (1,)), u1) == 0.0      # Uniform marginal
        # Fast-path probe: condition returns our specialised NestedDistortion.
        @test condition(Cbiv, (1,), u1) isa NestedDistortion

        # (b) Flat ArchimedeanCopula via the SAME gist recipe, against the
        #     ϕ⁽ᵏ⁾ + ϕ⁻¹′ closed form (upstream ArchimedeanDistortion / subsetdims).
        Cf = ClaytonCopula(4, 2.0)
        uf = [0.5, 0.8, 0.3, 0.6]; δf = [false, true, false, true]
        G = Cf.G; ssum = sum(ϕ⁻¹(G, uf[i]) for i in 1:4); k = count(!, δf)
        ref_cop = log(abs(ϕ⁽ᵏ⁾(G, k, ssum))) +
                  sum(log(abs(ϕ⁻¹⁽¹⁾(G, uf[i]))) for i in 1:4 if !δf[i])
        @test gist_censored(Cf, uf, δf) ≈ ref_cop atol = 1e-10

        # (c) Nested, multi-unobserved (2 lower-tail): root Clayton(2)-leaf over
        #     Clayton(5) + Gumbel(3), one lower-tail leaf in each sector.
        #     Multi-coordinate (|unobserved| = 2): now routed
        #     through our Faà di Bruno tree walk via the `_partial_cdf` override
        #     (no ForwardDiff). The RefSpec reference is an INDEPENDENT
        #     BigFloat-ForwardDiff CDF mixed partial that shares no code with our
        #     kernel, so matching it to 1e-10 (the old generic path was asserted
        #     only at the loose 1e-7) is evidence the multi-coordinate CDF is exact.
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                leaves = [1],
                children = [ClaytonCopula(2, 5.0), GumbelCopula(2, 3.0)])
        u = [0.40, 0.30, 0.70, 0.55, 0.80]
        δ = [false, false, true, true, false]
        spec = RefSpec(ClaytonGenerator(big(2.0)),
                   [(big(u[1]), false)],
                   [RefSpec(ClaytonGenerator(big(5.0)), [(big(u[2]), false), (big(u[3]), true)]),
                    RefSpec(GumbelGenerator(big(3.0)),  [(big(u[4]), true),  (big(u[5]), false)])])
        @test gist_censored(C, u, δ) ≈ Float64(ref_logpdf(spec)) atol = 1e-10

        # (c') Single lower-tail nested ⇒ the FAST NestedDistortion path. Observe
        #      all of {1,2,3,4}, leave dim 5 lower-tail.
        δ1 = [false, false, false, false, true]
        spec1 = RefSpec(ClaytonGenerator(big(2.0)),
                    [(big(u[1]), false)],
                    [RefSpec(ClaytonGenerator(big(5.0)), [(big(u[2]), false), (big(u[3]), false)]),
                     RefSpec(GumbelGenerator(big(3.0)),  [(big(u[4]), false), (big(u[5]), true)])])
        @test condition(C, (1, 2, 3, 4), [u[1], u[2], u[3], u[4]]) isa NestedDistortion
        @test gist_censored(C, u, δ1) ≈ Float64(ref_logpdf(spec1)) atol = 1e-9

        # (d) Degenerate masks: all observed == plain logpdf; all lower-tail == log cdf.
        C2 = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                 children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
        u2v = big.([0.30, 0.55, 0.70, 0.40, 0.62, 0.80])
        @test gist_censored(C2, u2v, falses(6)) == logpdf(C2, u2v)
        @test gist_censored(C2, u2v, trues(6)) ≈ log(cdf(C2, u2v)) atol = 1e-30
    end

    # -----------------------------------------------------------------------
    # 4b. SklarDist lower-tail likelihood via condition + subsetdims (data scale).
    # -----------------------------------------------------------------------
    @testset "SklarDist lower-tail via condition + subsetdims" begin
        # (a) Bivariate Clayton(3), dim 2 lower-tail — closed form, data scale.
        θ = 3.0
        m = (Exponential(1.0), Exponential(1.0))
        S = SklarDist(ClaytonCopula(2, θ), m)
        x1 = 0.7; c2 = 1.3
        u1 = cdf(m[1], x1); u2 = cdf(m[2], c2)
        dCdu1 = u1^(-θ - 1) * (u1^(-θ) + u2^(-θ) - 1)^(-1 / θ - 1)
        expected = log(dCdu1) + logpdf(m[1], x1)
        @test gist_sklar(S, [x1, c2], [false, true]) ≈ expected atol = 1e-10

        # (b) Omitted mask == the plain joint density.
        @test gist_sklar(S, [x1, c2], [false, false]) ≈ logpdf(S, [x1, c2]) atol = 1e-12

        # (c) Finite where the Distributions.censored route is -Inf — the
        #     motivating contrast for the lower-tail recipe.
        Sc = SklarDist(ClaytonCopula(2, θ), (m[1], censored(m[2], upper = c2)))
        @test logpdf(Sc, [x1, c2]) == -Inf
        @test isfinite(gist_sklar(S, [x1, c2], [false, true]))

        # (d) All lower-tail == log cdf of the joint model.
        @test gist_sklar(S, [x1, c2], [true, true]) ≈ log(cdf(S, [x1, c2])) atol = 1e-10

        # (e) Nested copula on the data scale, multi-coordinate: the gist recipe is
        #     finite and matches the observed-marginal densities + the nested
        #     mixed partial at the PIT point (independent reference).
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                children = [ClaytonCopula(3, 5.0), GumbelCopula(3, 3.0)])
        margins = ntuple(_ -> Exponential(1.0), 6)
        Sn = SklarDist(C, margins)
        x = [0.7, 0.3, 0.9, 0.5, 0.4, 1.1]
        δ = [false, true, false, false, true, false]
        u = [cdf(margins[i], x[i]) for i in 1:6]
        margin_ll = sum(logpdf(margins[i], x[i]) for i in 1:6 if !δ[i])
        # copula-scale mixed partial over the observed coords (data-scale gist
        # should equal margin densities + the copula-scale gist at the PIT point).
        cop_ll = gist_censored(C, u, δ)
        # Tightened 1e-7 → 1e-10: both sides route the multi-coordinate conditional
        # CDF through our kernel now, so the data-scale/copula-scale split agrees
        # to machine precision. (Self-consistency via gist_censored — NOT an
        # independent reference; the independent proof is testset 4c below.)
        @test gist_sklar(Sn, x, δ) ≈ margin_ll + cop_ll atol = 1e-10
        @test isfinite(gist_sklar(Sn, x, δ))
    end

    # -----------------------------------------------------------------------
    # 4c. Multi-coordinate conditional CDF routes through OUR Faà di Bruno kernel
    #     (not ForwardDiff). The standard API (condition + subsetdims) now CALLS
    #     `_censored_copula_logpdf` for any number of lower-tail dims, via the
    #     `_partial_cdf(::NestedArchimedeanCopula, …)` override. Fixed literals.
    # -----------------------------------------------------------------------
    @testset "multi-coordinate conditional CDF routes through our kernel" begin
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                leaves = [1],
                children = [ClaytonCopula(2, 5.0), GumbelCopula(2, 3.0)])
        u = [0.40, 0.30, 0.70, 0.55, 0.80]
        δ = [false, false, true, true, false]   # lower-tail dims 3,4 (|unobserved| = 2)

        # (i) DIRECT-KERNEL EQUALITY. The gist via the standard API equals a DIRECT
        #     `_censored_copula_logpdf` call to machine precision — the load-bearing
        #     proof that `condition()`/`_cdf` now CALLS our kernel. (Pre-override the
        #     LHS was ForwardDiff over the closed-form CDF; it happened to agree on
        #     this benign case to ~1e-15 too, so the *independent BigFloat reference*
        #     is what proves exactness, while this equality proves the code PATH.)
        api  = gist_censored(C, u, δ)
        kern = _censored_copula_logpdf(C, u, δ, Float64)
        @test api ≈ kern atol = 1e-10
        # exactness witness against the BigFloat kernel (no Float64 roundoff in ref).
        @test api ≈ Float64(_censored_copula_logpdf(C, big.(u), δ, BigFloat)) atol = 1e-9

        # (ii) CDF CONTRACT on the real multi-coordinate ConditionalCopula. Build it
        #      via condition(); cdf must stay in [0,1] and be non-decreasing on a
        #      fixed interior increasing v-grid. Catches sign/assembly errors that a
        #      single-point equality misses. Interior grid + tolerance-relaxed
        #      monotone check avoid boundary/quantile flakiness.
        obs = Tuple(i for i in 1:5 if !δ[i])
        CC = condition(C, obs, [u[i] for i in obs])
        vgrid = [[0.2, 0.2], [0.4, 0.3], [0.6, 0.55], [0.8, 0.85], [0.95, 0.97]]
        vals = [cdf(CC, v) for v in vgrid]
        @test all(0.0 .<= vals .<= 1.0)
        @test all(vals[k + 1] >= vals[k] - 1e-12 for k in 1:length(vals) - 1)

        # (iii) BigFloat-ROUTABILITY + boundary robustness (the DEFENSIBLE adversarial
        #       win). Standard-API conditioning differentiates a closed-form CDF in
        #       Float64 (ForwardDiff); at high differentiation order for a fast-tail
        #       generator BOTH Float64 paths (ForwardDiff AND our Float64 kernel) lose
        #       precision and eventually NaN — so we do NOT claim the Float64 kernel
        #       beats Float64 ForwardDiff. The universal advantage is that the kernel
        #       computes the SAME quantity exactly in BigFloat, which the Float64-locked
        #       standard API cannot reach. Verify: (a) a moderately deep multi-coordinate
        #       point is finite and BigFloat-exact; (b) at a deeper point the Float64
        #       ForwardDiff conditional CDF is non-finite (NaN) yet the BigFloat kernel
        #       is finite and matches.
        Cj = NestedArchimedeanCopula(JoeGenerator(1.2);
                children = [JoeCopula(4, 15.0), JoeCopula(4, 15.0)])
        δj = [false, false, false, false, false, false, false, true]  # observe 7 (order 7)
        # (a) moderate point: finite under the Float64 kernel AND BigFloat-exact.
        um = fill(0.99, 8)
        gm = gist_censored(Cj, um, δj)
        @test isfinite(gm)
        @test gm ≈ Float64(_censored_copula_logpdf(Cj, big.(um), δj, BigFloat)) atol = 1e-8
        # (b) deep tail: upstream's Float64 ForwardDiff conditional CDF NaNs, but the
        #     BigFloat kernel is exact. is/js are the override's upstream slot names:
        #     is = lower-tail dim (8), js = observed dims (1..7).
        ud = fill(0.999999, 8)
        fd_f64 = Copulas._partial_cdf(Cj, (8,), (1, 2, 3, 4, 5, 6, 7),
                                      (ud[8],), (ud[1], ud[2], ud[3], ud[4], ud[5], ud[6], ud[7]))
        # NOTE: this `_partial_cdf` call goes through OUR override (Cj is nested), so
        # in Float64 it now follows the kernel — which ALSO NaNs at this depth. The
        # point being asserted is that the Float64 path (whichever) is non-finite here
        # while BigFloat recovers the exact value, NOT that one Float64 path beats the
        # other.
        @test !isfinite(fd_f64)
        big_log = _censored_copula_logpdf(Cj, big.(ud),
                      [false, false, false, false, false, false, false, true], BigFloat)
        @test isfinite(big_log)
    end

    # -----------------------------------------------------------------------
    # 5. Arbitrary-depth nesting builds, is finite, and matches the reference.
    # -----------------------------------------------------------------------
    @testset "arbitrary-depth nesting" begin
        # Inner nested copula: Joe(3) over a Joe(4) panel (dims 2:3 once placed).
        joesub = NestedArchimedeanCopula(JoeGenerator(3.0);
                     children = [JoeCopula(2, 4.0)])
        C = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                leaves = [1], children = [joesub])
        @test C isa NestedArchimedeanCopula{3}
        u = big.([0.2, 0.6, 0.7])
        spec = RefSpec(ClaytonGenerator(big(1.5)),
                   [(u[1], false)],
                   [RefSpec(JoeGenerator(big(3.0)), Tuple{BigFloat,Bool}[],
                        [RefSpec(JoeGenerator(big(4.0)), [(u[2], false), (u[3], false)])])])
        @test isfinite(logpdf(C, u))
        @test logpdf(C, u) ≈ ref_logpdf(spec) atol = 1e-9
    end

    # -----------------------------------------------------------------------
    # 6. Constructor validation and a fit/smoke usage.
    # -----------------------------------------------------------------------
    @testset "constructor validation & smoke" begin
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [1, 1])
        # Overlapping dims must error.
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [1], children = [ClaytonCopula(2, 5.0) => [1, 2]])
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(2, 5.0) => [1]])
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            children = [ClaytonCopula(2, 5.0) => [2, 3]])
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [0], children = [ClaytonCopula(2, 5.0)])
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [-1], children = [ClaytonCopula(2, 5.0)])
        # Auto-placement must not silently overlap with a root leaf.
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [2], children = [ClaytonCopula(2, 5.0)])
        # But it may fill a free contiguous block before a later root leaf.
        placed = NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [3], children = [ClaytonCopula(2, 5.0)])
        @test placed.children[1][2] == [1, 2]
        # Legacy positional form still works and tiles 1:4.
        old = NestedArchimedeanCopula(ClaytonGenerator(2.0),
                  [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
        @test old isa NestedArchimedeanCopula{4}

        # Smoke: a small log-likelihood over sampled-ish data is finite, and a
        # higher root dependence gives a different (here larger) value at a
        # strongly-clustered point.
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])
        pts = [big.([0.3, 0.32, 0.6, 0.62]), big.([0.5, 0.52, 0.2, 0.22])]
        ll = sum(logpdf(C, p) for p in pts)
        @test isfinite(ll)

        # Mixed CDF boundaries marginalise coordinates at one and vanish when
        # any coordinate is zero. Density support checks accept numeric input
        # types without attempting to convert -Inf to an integer.
        u = [0.3, 0.4, 0.6, 0.7]
        @test cdf(C, [u[1], u[2], 1.0, 1.0]) ≈ cdf(ClaytonCopula(2, 5.0), u[1:2])
        @test iszero(cdf(C, [u[1], 0.0, u[3], u[4]]))
        @test logpdf(C, [0, 1, 1, 1]) == -Inf
        @test logpdf(C, [u[1], 1.0, u[3], u[4]]) == -Inf
        @test logpdf(C, [u[1], -0.1, u[3], u[4]]) == -Inf
        @test logpdf(C, [u[1], NaN, u[3], u[4]]) == -Inf
    end

    # -----------------------------------------------------------------------
    # 7. Global implicit override gives correct nested densities (end-to-end).
    #    Redefining the GENERIC `composition_taylor(::Generator,::Generator,…)`
    #    method switches every edge to the implicit App. A.4 solver. We re-run
    #    the 4 external-acopula CSV cases of testset 3 through the implicit path
    #    and assert the SAME `maxerr < 1e-9` — proving the implicit solver
    #    reproduces correct nested logpdfs at the full density level, not just
    #    per-edge.
    #
    #    A same-signature redefinition is a SESSION-GLOBAL override (it emits a
    #    benign "Method overwritten" warning) — so this testset runs LAST, after
    #    the default-direct byte-identity guard (testset 3) has already asserted.
    # -----------------------------------------------------------------------
    @testset "global implicit override gives correct nested densities" begin
        # Repoint the global default to the implicit solver (benign overwrite warning).
        Copulas.composition_taylor(o::Copulas.Generator, i::Copulas.Generator, t₀, d) =
            Copulas.composition_taylor_implicit(o, i, t₀, d)

        datadir = joinpath(@__DIR__, "data", "nested")
        for case in _ACOPULA_CASES
            @test acopula_maxerr(datadir, case...; nrows = 6) < 1e-9
        end

        # Restore the shipped default-direct generic method so the override does
        # not leak into any later test in the same session.
        Copulas.composition_taylor(o::Copulas.Generator, i::Copulas.Generator, t₀, d) =
            Copulas.composition_taylor_direct(o, i, t₀, d)
    end

    @testset "subsetdims respects requested coordinate order (reordering)" begin
        # Regression: SubsetCopula relabelled by sorted dims, ignoring the
        # requested order, so subsetdims(C, perm) returned the wrong marginal for
        # a non-exchangeable nested structure. The marginal CDF of a REORDERED
        # subset must equal the joint CDF saturated at the requested coordinates.
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                children = [ClaytonCopula(3, 5.0), ClaytonCopula(2, 6.0)])   # d=5
        for dims in [(4, 1, 2), (2, 4, 1), (1, 3), (3, 1), (5, 2, 3, 1)]
            u = [0.2 + 0.1k for k in 1:length(dims)]
            v = ones(5); for (k, j) in enumerate(dims); v[j] = u[k]; end
            @test cdf(subsetdims(C, Tuple(dims)), u) ≈ cdf(C, v) atol = 1e-10
        end

        # Density uses the same requested-coordinate order. Compare it with an
        # independent mixed derivative of the saturated full CDF.
        dims = (4, 1, 2)
        u = big.([0.35, 0.55, 0.65])
        function reordered_marginal_cdf(x)
            v = ones(eltype(x), 5)
            for (k, j) in enumerate(dims)
                v[j] = x[k]
            end
            return cdf(C, v)
        end
        ref_density = mixed_partial(reordered_marginal_cdf, u, collect(1:3))
        @test logpdf(subsetdims(C, dims), u) ≈ log(abs(ref_density)) atol = 1e-9

        # Conditioning the reordered subset must differentiate the original
        # global coordinate selected by the first requested position (dim 4).
        δ = [false, true, true]
        ref_partial = mixed_partial(reordered_marginal_cdf, u, [1])
        @test gist_censored(subsetdims(C, dims), u, δ) ≈ log(abs(ref_partial)) atol = 1e-9

        # The reordered marginal remains sampleable through inverse Rosenblatt.
        sr = rand(Random.MersenneTwister(41), subsetdims(C, dims), 3)
        @test size(sr) == (3, 3)
        @test all(0 .<= sr .<= 1) && all(isfinite, sr)
        # Order genuinely matters: a within-panel-spanning reorder differs.
        @test !isapprox(cdf(subsetdims(C, (4, 1, 2)), [0.3, 0.5, 0.6]),
                        cdf(subsetdims(C, (2, 4, 1)), [0.3, 0.5, 0.6]); atol = 1e-6)

        S = subsetdims(C, (1, 2, 4, 5))
        @test all(length(child[1]) == length(child[2]) for child in S.children)

        # Deep child subtrees with one surviving coordinate collapse to a leaf
        # under the nearest retained ancestor.
        inner = NestedArchimedeanCopula(GumbelGenerator(2.0);
                    leaves = [1], children = [GumbelCopula(2, 4.0)])
        deep = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                    leaves = [1], children = [inner])
        collapsed = subsetdims(deep, (1, 3))
        @test collapsed isa ArchimedeanCopula{2}
        @test cdf(collapsed, [0.4, 0.7]) ≈ cdf(ClaytonCopula(2, 1.5), [0.4, 0.7])
    end

    @testset "rand works (inverse-Rosenblatt sampler)" begin
        # rand(C) StackOverflowed before the dedicated _rand!; here it must return
        # a valid point and a valid sample matrix, all in [0,1] and finite.
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                children = [ClaytonCopula(2, 5.0), ClaytonCopula(2, 6.0)])   # d=4
        rng = Random.MersenneTwister(7)
        s1 = rand(rng, C)
        @test length(s1) == 4
        @test all(0 .<= s1 .<= 1) && all(isfinite, s1)
        S = rand(rng, C, 100)
        @test size(S) == (4, 100)
        @test all(0 .<= S .<= 1) && all(isfinite, S)
        # Marginals are (approximately) uniform: mean of each coordinate ≈ 0.5.
        @test all(abs.(vec(sum(S, dims = 2)) ./ 100 .- 0.5) .< 0.1)
    end

    @testset "fit: generator-parameter recovery (fixed tree MLE)" begin
        # Sample from a known nested Clayton tree, fit from a deliberately wrong
        # same-shape template, and check the recovered θ are close to the truth.
        # The tolerances below are deliberately loose: this is a smoke/regression
        # check that MLE moves toward the known parameters and preserves the tree.
        Ctrue = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                    children = [ClaytonCopula(2, 6.0), ClaytonCopula(2, 8.0)])
        U = rand(Random.MersenneTwister(20240601), Ctrue, 1000)

        Cstart = NestedArchimedeanCopula(ClaytonGenerator(1.0);
                     children = [ClaytonCopula(2, 3.0), ClaytonCopula(2, 3.0)])

        M = Distributions.fit(Copulas.CopulaModel, Cstart, U)
        Chat = M.result
        @test Chat isa NestedArchimedeanCopula
        @test M.converged
        # Same tree shape preserved.
        @test length(Chat) == 4
        @test length(Chat.children) == 2
        # Generators stayed in the Clayton family (no collapse / type change).
        @test Chat.G isa ClaytonGenerator
        @test Chat.children[1][1].G isa ClaytonGenerator
        @test Chat.children[2][1].G isa ClaytonGenerator
        # Parameter recovery (loose MLE tolerances on 1000 samples).
        @test Chat.G.θ ≈ 2.0 atol = 0.6
        @test Chat.children[1][1].G.θ ≈ 6.0 atol = 1.2
        @test Chat.children[2][1].G.θ ≈ 8.0 atol = 1.2
        # Fitted log-likelihood beats the (wrong) starting point.
        @test Distributions.loglikelihood(Chat, U) > Distributions.loglikelihood(Cstart, U)

        # coef / dof report the real free-parameter count (3: root + 2 panels),
        # so AIC/BIC are correct (the generic path would give dof=0 → wrong AIC).
        @test length(StatsBase.coef(M)) == 3
        @test StatsBase.dof(M) == 3
        @test StatsBase.coef(M) ≈ [Chat.G.θ, Chat.children[1][1].G.θ, Chat.children[2][1].G.θ]
        @test StatsBase.aic(M) ≈ -2 * Distributions.loglikelihood(Chat, U) + 2 * 3

        # Quick instance shim returns just the fitted copula with the same fit.
        Cq = Distributions.fit(Cstart, U[:, 1:150])
        @test Cq isa NestedArchimedeanCopula

        # Bare-type fit is intentionally unsupported (tree not inferable).
        @test_throws Exception Copulas._example(NestedArchimedeanCopula, 4)
        # Only :mle is supported.
        @test_throws ArgumentError Distributions.fit(Copulas.CopulaModel, Cstart, U; method = :itau)
    end

    @testset "fit: parametrisation layer (nesting + custom reparam)" begin
        C = NestedArchimedeanCopula(ClaytonGenerator(1.5); leaves = [1],
                                    children = [ClaytonCopula(2, 4.0)])
        U = rand(Random.MersenneTwister(7), C, 300)
        rootθ(M)  = M.result.G.θ
        childθ(M) = M.result.children[1][1].G.θ

        # default parametrisation: 2 free parameters (root + child)
        Md = Distributions.fit(Copulas.CopulaModel, C, U)
        @test StatsBase.dof(Md) == 2

        # custom reparam encoding NESTING (no template): child θ = root θ + softplus(δ)
        # ≥ root θ, so every optimiser step is a valid nesting.
        sp(x) = log1p(exp(-abs(x))) + max(x, zero(x))
        nest = α -> (θr = exp(α[1]); θc = θr + sp(α[2]);
            NestedArchimedeanCopula(ClaytonGenerator(θr); leaves = [1],
                                    children = [ClaytonCopula(2, θc)]))
        Mn = Distributions.fit(Copulas.CopulaModel, nest, [0.0, 0.0], U)
        @test rootθ(Mn) ≤ childθ(Mn)                  # nesting enforced by the user's reparam
        @test StatsBase.dof(Mn) == 2

        # custom reparam SHARING one θ across root and child → 1 free parameter
        recon = α -> (θ = exp(α[1]);
            NestedArchimedeanCopula(ClaytonGenerator(θ); leaves = [1],
                                    children = [ClaytonCopula(2, θ)]))
        Ms = Distributions.fit(Copulas.CopulaModel, recon, [log(2.0)], U)
        @test StatsBase.dof(Ms) == 1                  # shared ⇒ fewer dof than #generators
        @test rootθ(Ms) ≈ childθ(Ms)                  # the shared parameter

        # quick_fit returns just the copula; the dimension comes from the reparam
        @test Distributions.fit(Copulas.CopulaModel, recon, [log(2.0)], U; quick_fit = true).result isa NestedArchimedeanCopula
    end
end
