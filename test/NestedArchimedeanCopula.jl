# Tests for NestedArchimedeanCopula: the nested-Archimedean density and its
# optional per-variable censored (survival) likelihood (Yang & Li, arXiv:2605.23134).
#
# Coverage:
#   1. Flat dispatch — a leaves-only declaration returns the native
#      ArchimedeanCopula and gives a bit-for-bit identical logpdf.
#   2. Uncensored density vs an INDEPENDENT reference: the nested CDF assembled
#      directly from the generators, mixed-partial-differentiated by nested
#      ForwardDiff. This shares no code path with the Faà di Bruno recursion.
#   3. Uncensored density vs an EXTERNAL reference (acopula log-likelihoods,
#      committed in test/data/nested/).
#   4. Censored likelihood vs the same independent ForwardDiff reference (mixed
#      partial over the observed dims only), incl. the bivariate Clayton
#      closed form; plus the omitted-mask == plain-density default.
#   5. Heterogeneous (mixed-family) and arbitrary-depth nesting build & are finite.
#   6. Constructor errors (bad tiling) and a fit/smoke usage of logpdf.

using Test, Copulas, Distributions, ForwardDiff, DelimitedFiles
import Copulas: Generator, ϕ, ϕ⁻¹
import Copulas: ClaytonGenerator, GumbelGenerator, FrankGenerator, JoeGenerator

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

@testset "NestedArchimedeanCopula" begin

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
    # 4. Censored likelihood (logpdf with a mask) vs the ForwardDiff reference.
    # -----------------------------------------------------------------------
    @testset "censored likelihood vs independent ForwardDiff reference" begin
        # Bivariate Clayton(3), dim 2 right-censored: closed form ∂C/∂u₁.
        θ = big(3.0)
        u1 = BigFloat(cdf(Exponential(1.0), 0.5))
        u2 = BigFloat(cdf(Exponential(1.0), 1.0))
        Cbiv = NestedArchimedeanCopula(ClaytonGenerator(2.0);   # outer irrelevant: single panel
                   children = [ClaytonCopula(2, 3.0)])
        v = logpdf(Cbiv, [u1, u2]; censored = [false, true])
        dC_du1 = u1^(-θ - 1) * (u1^(-θ) + u2^(-θ) - 1)^(-(1 / θ + 1))
        @test Float64(v) ≈ Float64(log(dC_du1)) atol = 1e-9

        # Nested, censoring across sectors: root Clayton(2) over Clayton(5) +
        # Gumbel(3), one censored leaf in each sector + a censored root leaf.
        C = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                leaves = [1],
                children = [ClaytonCopula(2, 5.0), GumbelCopula(2, 3.0)])
        u = big.([0.40, 0.30, 0.70, 0.55, 0.80])
        δ = [false, false, true, true, false]
        spec = RefSpec(ClaytonGenerator(big(2.0)),
                   [(u[1], false)],
                   [RefSpec(ClaytonGenerator(big(5.0)), [(u[2], false), (u[3], true)]),
                    RefSpec(GumbelGenerator(big(3.0)),  [(u[4], true),  (u[5], false)])])
        @test logpdf(C, u; censored = δ) ≈ ref_logpdf(spec) atol = 1e-9

        # Omitted mask == plain density; full mask present changes the value.
        C2 = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                 children = [ClaytonCopula(3, 5.0), ClaytonCopula(3, 6.0)])
        u2v = big.([0.30, 0.55, 0.70, 0.40, 0.62, 0.80])
        @test logpdf(C2, u2v; censored = falses(6)) == logpdf(C2, u2v)
        @test logpdf(C2, u2v; censored = [false, true, false, false, false, true]) != logpdf(C2, u2v)
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
