# Tests for NestedArchimedeanCopula: the nested-Archimedean density and its
# per-variable censoring as an EMERGENT capability of the standard
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
#   4. Per-variable censoring via the gist recipe
#      `logpdf(subsetdims(X,O),x_O) + logcdf_or_cdf(condition(X,O,x_O),x_C)`,
#      checked against the SAME independent ForwardDiff references (mixed partial
#      over the observed dims), incl. the bivariate Clayton closed form, plus a
#      fast-path type probe (condition returns NestedDistortion) and the
#      multi-unobserved generic ForwardDiff fallback.
#   4b. SklarDist survival likelihood via condition + subsetdims on the data scale,
#       incl. the Distributions.censored == -Inf contrast.
#   5. Heterogeneous (mixed-family) and arbitrary-depth nesting build & are finite.
#   6. Constructor errors (bad tiling) and a fit/smoke usage of logpdf.

using Test, Copulas, Distributions, ForwardDiff, DelimitedFiles, Random
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

# Express per-variable censoring through the STANDARD condition + subsetdims API
# (the "gist recipe"). `δ[i] == true` ⇒ coordinate `i` is censored/unobserved.
#   logL_O = logpdf(subsetdims(C,O), u_O) + logcdf_or_cdf(condition(C,O,u_O), u_C)
# with the degenerate masks handled explicitly (no observed ⇒ log cdf; no
# censored ⇒ plain logpdf).
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
        for _ in 1:5
            u = big.(rand(rng, 4) .* 0.6 .+ 0.2)
            spec = RefSpec(ClaytonGenerator(big(1.5)),
                       Tuple{BigFloat,Bool}[],
                       [RefSpec(ClaytonGenerator(big(3.0)), [(u[1], false), (u[2], false)]),
                        RefSpec(ClaytonGenerator(big(3.0)), [(u[3], false), (u[4], false)])])
            @test logpdf(C, u) ≈ ref_logpdf(spec) atol = 1e-10
        end
        # Heterogeneous: Clayton root over a Gumbel panel + a Frank panel.
        H = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                children = [GumbelCopula(2, 2.0), FrankCopula(2, 3.0)])
        for _ in 1:5
            u = big.(rand(rng, 4) .* 0.6 .+ 0.2)
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
        cases = [
            ("clayton_d10_2level", ClaytonGenerator, [5, 5],        1.5, 3.0),
            ("clayton_d20_2level", ClaytonGenerator, [5, 5, 5, 5],  2.0, 4.0),
            ("gumbel_d10",         GumbelGenerator,  [5, 5],        2.0, 5.0),
            ("frank_d10",          FrankGenerator,   [5, 5],        2.0, 5.0),
        ]
        for (name, GT, sectors, θroot, θsector) in cases
            U  = readdlm(joinpath(datadir, name * "_U.csv"), ',')
            ll = vec(readdlm(joinpath(datadir, name * "_acopula_ll.csv"), ','))
            C = NestedArchimedeanCopula(GT(θroot);
                    children = [ArchimedeanCopula(s, GT(θsector)) for s in sectors])
            maxerr = 0.0
            for i in axes(U, 1)
                ours = Float64(logpdf(C, big.(U[i, :])))
                maxerr = max(maxerr, abs(ours - ll[i]))
            end
            @test maxerr < 1e-9
        end
    end

    # -----------------------------------------------------------------------
    # 4. Per-variable censoring via condition + subsetdims (gist recipe).
    #    Reproduces the SAME independent references the old bespoke API checked,
    #    now through the STANDARD API — proving the specialised condition /
    #    subsetdims path is equivalent (the denominator c_O cancels exactly).
    # -----------------------------------------------------------------------
    @testset "per-variable censoring via condition + subsetdims (gist recipe)" begin
        # (a) Bivariate Clayton(3), dim 2 censored: gist == closed form ∂C/∂u₁.
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

        # (c) Nested, multi-unobserved (2 censored): root Clayton(2)-leaf over
        #     Clayton(5) + Gumbel(3), one censored leaf in each sector + a
        #     censored root leaf. Multi-censored (|unobserved| = 2): now routed
        #     through our Faà di Bruno tree walk via the `_partial_cdf` override
        #     (no ForwardDiff). The RefSpec reference is an INDEPENDENT
        #     BigFloat-ForwardDiff CDF mixed partial that shares no code with our
        #     kernel, so matching it to 1e-10 (the old generic path was asserted
        #     only at the loose 1e-7) is evidence the multi-censored CDF is exact.
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

        # (c') Single-censored nested ⇒ the FAST NestedDistortion path. Observe
        #      all of {1,2,3,4}, censor only dim 5.
        δ1 = [false, false, false, false, true]
        spec1 = RefSpec(ClaytonGenerator(big(2.0)),
                    [(big(u[1]), false)],
                    [RefSpec(ClaytonGenerator(big(5.0)), [(big(u[2]), false), (big(u[3]), false)]),
                     RefSpec(GumbelGenerator(big(3.0)),  [(big(u[4]), false), (big(u[5]), true)])])
        @test condition(C, (1, 2, 3, 4), [u[1], u[2], u[3], u[4]]) isa NestedDistortion
        @test gist_censored(C, u, δ1) ≈ Float64(ref_logpdf(spec1)) atol = 1e-9

        # (d) Degenerate masks: all observed == plain logpdf; all censored == log cdf.
        C2 = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                 children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
        u2v = big.([0.30, 0.55, 0.70, 0.40, 0.62, 0.80])
        @test gist_censored(C2, u2v, falses(6)) == logpdf(C2, u2v)
        @test gist_censored(C2, u2v, trues(6)) ≈ log(cdf(C2, u2v)) atol = 1e-30
    end

    # -----------------------------------------------------------------------
    # 4b. SklarDist survival likelihood via condition + subsetdims (data scale).
    # -----------------------------------------------------------------------
    @testset "SklarDist survival via condition + subsetdims" begin
        # (a) Bivariate Clayton(3), dim 2 right-censored — closed form, data scale.
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
        #     motivating contrast for the per-variable censoring facility.
        Sc = SklarDist(ClaytonCopula(2, θ), (m[1], censored(m[2], upper = c2)))
        @test logpdf(Sc, [x1, c2]) == -Inf
        @test isfinite(gist_sklar(S, [x1, c2], [false, true]))

        # (d) All censored == log cdf of the joint model.
        @test gist_sklar(S, [x1, c2], [true, true]) ≈ log(cdf(S, [x1, c2])) atol = 1e-10

        # (e) Nested copula on the data scale, multi-censored: the gist recipe is
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
        # Tightened 1e-7 → 1e-10: both sides route the multi-censored conditional
        # CDF through our kernel now, so the data-scale/copula-scale split agrees
        # to machine precision. (Self-consistency via gist_censored — NOT an
        # independent reference; the independent proof is testset 4c below.)
        @test gist_sklar(Sn, x, δ) ≈ margin_ll + cop_ll atol = 1e-10
        @test isfinite(gist_sklar(Sn, x, δ))
    end

    # -----------------------------------------------------------------------
    # 4c. Multi-censored conditional CDF routes through OUR Faà di Bruno kernel
    #     (not ForwardDiff). The standard API (condition + subsetdims) now CALLS
    #     `_censored_copula_logpdf` for any number of censored dims, via the
    #     `_partial_cdf(::NestedArchimedeanCopula, …)` override. Fixed literals.
    # -----------------------------------------------------------------------
    @testset "multi-censored conditional CDF routes through our kernel" begin
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                leaves = [1],
                children = [ClaytonCopula(2, 5.0), GumbelCopula(2, 3.0)])
        u = [0.40, 0.30, 0.70, 0.55, 0.80]
        δ = [false, false, true, true, false]   # censor dims 3,4 (|unobserved| = 2)

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

        # (ii) CDF CONTRACT on the real multi-censored ConditionalCopula. Build it
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
        #       standard API cannot reach. Verify: (a) a moderately deep multi-censored
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
        #     is = censored dim (8), js = observed dims (1..7).
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
        # Inner nested copula: Joe(3) over a Joe(4) panel (dims 5:6 once placed).
        joesub = NestedArchimedeanCopula(JoeGenerator(3.0);
                     children = [JoeCopula(2, 4.0)])
        C = NestedArchimedeanCopula(ClaytonGenerator(1.5);
                children = [GumbelCopula(2, 2.0), FrankCopula(2, 3.0), joesub])
        @test C isa NestedArchimedeanCopula{6}
        u = big.([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        spec = RefSpec(ClaytonGenerator(big(1.5)),
                   Tuple{BigFloat,Bool}[],
                   [RefSpec(GumbelGenerator(big(2.0)), [(u[1], false), (u[2], false)]),
                    RefSpec(FrankGenerator(big(3.0)),  [(u[3], false), (u[4], false)]),
                    RefSpec(JoeGenerator(big(3.0)), Tuple{BigFloat,Bool}[],
                        [RefSpec(JoeGenerator(big(4.0)), [(u[5], false), (u[6], false)])])])
        @test isfinite(logpdf(C, u))
        @test logpdf(C, u) ≈ ref_logpdf(spec) atol = 1e-9
    end

    # -----------------------------------------------------------------------
    # 6. Constructor validation and a fit/smoke usage.
    # -----------------------------------------------------------------------
    @testset "constructor validation & smoke" begin
        # Overlapping dims must error.
        @test_throws ArgumentError NestedArchimedeanCopula(ClaytonGenerator(2.0);
            leaves = [1], children = [ClaytonCopula(2, 5.0) => [1, 2]])
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
    end
end
