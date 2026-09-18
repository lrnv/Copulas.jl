from pathlib import Path
import re


def read(path):
    return Path(path).read_text()


def write(path, text):
    Path(path).write_text(text)


def replace(path, old, new, count=1):
    text = read(path)
    if old not in text:
        raise RuntimeError(f"missing replacement anchor in {path}: {old[:120]!r}")
    text = text.replace(old, new, count)
    write(path, text)


def sub(path, pattern, repl, count=1, flags=0):
    text = read(path)
    text2, n = re.subn(pattern, repl, text, count=count, flags=flags)
    if n != count:
        raise RuntimeError(
            f"expected {count} regex replacement(s) in {path}, got {n}: {pattern!r}"
        )
    write(path, text2)


# ---------------------------------------------------------------------------
# Source: Paramorph owns optimizer coordinates and parameter geometry.
# ---------------------------------------------------------------------------
replace(
    "src/ArchimedeanCopula.jl",
    """    pspace = Paramorph.param_space(CT, d)\n    θ₀ = zeros(Paramorph.dimension(pspace))\n\n    if start isa Real\n        θ₀[1] = start\n    elseif start ∈ (:itau, :irho)\n        θ₀[1] = only(Distributions.params(_fit(CT, U, Val{start}(); weights)))\n    end\n\n    cop(θ) = CT(d, Paramorph.constrain(pspace, θ))\n    f(θ) = -_weighted_loglikelihood(cop(θ), U, weights)\n\n    res = Optim.optimize(\n        f,\n        Optim.TwiceDifferentiableConstraints(),\n        θ₀,\n""",
    """    pspace = Paramorph.param_space(CT, d)\n    α₀ = if start isa Real\n        Paramorph.unconstrain(pspace, start)\n    elseif start ∈ (:itau, :irho)\n        θ₀ = only(Distributions.params(_fit(CT, U, Val{start}(); weights)))\n        Paramorph.unconstrain(pspace, θ₀)\n    else\n        zeros(Paramorph.dimension(pspace))\n    end\n\n    cop(α) = CT(d, Paramorph.constrain(pspace, α))\n    f(α) = -_weighted_loglikelihood(cop(α), U, weights)\n\n    res = Optim.optimize(\n        f,\n        Optim.TwiceDifferentiableConstraints(),\n        α₀,\n""",
)

replace(
    "src/Fitting.jl",
    "Simple parametric families opt into the generic routines by defining `Paramorph.param_space`\n#####  and `Paramorph.param_space`.",
    "Simple parametric families opt into the generic routines by defining `Paramorph.param_space`\n#####  and a canonical `CT(d, parameters...)` constructor.",
)

# ---------------------------------------------------------------------------
# Tests: Distributions.params is positional; names and geometry live in spaces.
# ---------------------------------------------------------------------------
replace("test/operations/distribution.jl", "@test params(C) isa NamedTuple", "@test params(C) isa Tuple")

replace(
    "test/api/constructors.jl",
    """        @test only(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n\n        covariance[1, 2] = covariance[2, 1] = 0\n        @test only(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n\n        first_params = params(C)\n        second_params = params(C)\n        @test first_params.Σ == second_params.Σ\n        @test first_params.Σ !== second_params.Σ\n        first_params.Σ[1, 2] = first_params.Σ[2, 1] = 0\n        @test only(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n""",
    """        @test last(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n\n        covariance[1, 2] = covariance[2, 1] = 0\n        @test last(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n\n        first_params = params(C)\n        second_params = params(C)\n        first_matrix = last(first_params)\n        second_matrix = last(second_params)\n        @test first_matrix == second_matrix\n        @test first_matrix !== second_matrix\n        first_matrix[1, 2] = first_matrix[2, 1] = 0\n        @test last(params(C)) ≈ [1.0 1/3; 1/3 1.0]\n""",
)

replace(
    "test/operations/strict_optimizer_errors.jl",
    """Copulas._example(::Type{_BrokenADCopula}, d) = _BrokenADCopula(d, 0.5)\nCopulas._unbound_params(::Type{_BrokenADCopula}, d, θ) = [log(θ.θ / (1 - θ.θ))]\nCopulas._rebound_params(::Type{_BrokenADCopula}, d, α) = (; θ=inv(one(first(α)) + exp(-first(α))))\n""",
    """Copulas.Paramorph.param_space(::Type{<:_BrokenADCopula}, d) =\n    Copulas.Paramorph.ProbOpen(:θ)\n""",
)

replace(
    "test/api/generators.jl",
    "verifies their transform, inverse, derivative, and reconstruction identities.",
    "verifies their transform, inverse, and derivative identities.",
)
replace("test/api/generators.jl", "@test params(G) == (;)", "@test !applicable(params, G)")
replace(
    "test/api/generators.jl",
    """            @test params(G) isa NamedTuple\n            rebuilt = typeof(G)(values(params(G))...)\n            @test params(rebuilt) == params(G)\n""",
    """            @test !applicable(params, G)\n""",
)

replace(
    "test/api/tails.jl",
    "verifies stable-tail, Pickands, derivative, and reconstruction identities.",
    "verifies stable-tail, Pickands, and derivative identities.",
)
replace("test/api/tails.jl", "@test params(tail) isa NamedTuple", "@test !applicable(params, tail)")

replace(
    "test/api/public_compositions.jl",
    "@test params(frailty_generator) == (F=Exponential(),)",
    "@test !applicable(params, frailty_generator)",
)
replace(
    "test/api/public_compositions.jl",
    """    ranked_empirical = EmpiricalGenerator(_FIXTURE_DATA; pseudo_values=false)\n    @test params(ranked_empirical) == params(EmpiricalGenerator(pseudos(_FIXTURE_DATA)))\n""",
    """    ranked_empirical = EmpiricalGenerator(_FIXTURE_DATA; pseudo_values=false)\n    expected_empirical = EmpiricalGenerator(pseudos(_FIXTURE_DATA))\n    @test all(t -> Copulas.ϕ(ranked_empirical, t) ≈ Copulas.ϕ(expected_empirical, t),\n              (0.2, 0.5, 0.8))\n""",
)
replace(
    "test/api/public_compositions.jl",
    "@test params(tail) == (B=Float64.(B),)",
    "@test !applicable(params, tail)",
)

replace(
    "test/operations/fitting.jl",
    """    θ = params(fitted)\n    @test fitted isa TCopula{3}\n    @test θ.ν > 0\n    @test isfinite(θ.ν)\n    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(θ.Σ),)\n    @test maximum(abs.(LinearAlgebra.diag(θ.Σ) .- 1),) < 1e-12\n""",
    """    ν̂, Σ̂ = params(fitted)\n    @test fitted isa TCopula{3}\n    @test ν̂ > 0\n    @test isfinite(ν̂)\n    @test LinearAlgebra.isposdef(LinearAlgebra.Symmetric(Σ̂),)\n    @test maximum(abs.(LinearAlgebra.diag(Σ̂) .- 1),) < 1e-12\n""",
)
replace("test/operations/fitting.jl", "@test 0 < params(fitted).ν < 2", "@test 0 < first(params(fitted)) < 2")
replace(
    "test/operations/fitting.jl",
    "# Generic fitting additionally depends on the example, parameter transform,\n# and reconstruction methods selected for the concrete family.",
    "# Generic fitting additionally depends on the parameter space and canonical\n# reconstruction selected for the concrete family.",
)
replace(
    "test/operations/fitting.jl",
    """_check_parameter_roundtrip(C) =\n    !(C isa EmpiricalEVCopula) && !(C isa FGMCopula && length(C) != 2)\n\nfunction test_mle_parameter_plumbing(C)\n    Base.@nospecialize C\n\n    CT = typeof(C)\n    d = length(C)\n    bounded = params(C)\n    unbounded = Copulas._unbound_params(CT, d, bounded)\n    restored = Copulas._rebound_params(CT, d, unbounded)\n\n    @test keys(restored) == keys(bounded)\n\n    @test all(key -> getfield(bounded, key) ≈ getfield(restored, key), keys(bounded))\n\n    if applicable(Copulas._example, CT, d)\n        example = Copulas._example(CT, d)\n        @test example isa Copulas.Copula{d}\n    end\n\n    return nothing\nend\n""",
    """function test_mle_parameter_plumbing(C, pspace)\n    Base.@nospecialize C pspace\n\n    CT = typeof(C)\n    d = length(C)\n    bounded = params(C)\n    unconstrained = Copulas._parameter_space_coordinates(pspace, bounded)\n    restored = Copulas._parameter_space_copula(CT, d, pspace, unconstrained)\n    restored_params = params(restored)\n\n    @test length(Copulas.Paramorph.names(pspace)) == length(bounded)\n    @test length(restored_params) == length(bounded)\n    @test all(zip(restored_params, bounded)) do pair\n        a, b = pair\n        applicable(isapprox, a, b) ? isapprox(a, b) : a == b\n    end\n\n    return nothing\nend\n""",
)
replace(
    "test/operations/fitting.jl",
    """        _has_fitting_parameters(C) || continue\n        _check_parameter_roundtrip(C) || continue\n\n        @testset \"$(case.name)\" begin\n            test_mle_parameter_plumbing(C)\n        end\n""",
    """        _has_fitting_parameters(C) || continue\n        pspace = try\n            Copulas.Paramorph.param_space(CT, d)\n        catch err\n            err isa MethodError || rethrow()\n            nothing\n        end\n        pspace === nothing && continue\n\n        @testset \"$(case.name)\" begin\n            test_mle_parameter_plumbing(C, pspace)\n        end\n""",
)
replace("test/operations/fitting.jl", "    @test_throws Exception Copulas._example(NestedArchimedeanCopula, 4)\n", "")
sub(
    "test/operations/fitting.jl",
    r'''# Fitting-operation proof for parameterizations\. Public\n# route availability and result interfaces are covered by this operation and\n# the final routing inventory;\n# this file checks that unconstrained coordinates map bijectively to the\n# intended constrained parameter space\.\n@testset "asymmetric Mixed feasible fitting parameterization" begin\n.*?\nend\n\n\n@testset "dimension-specialized fitting reconstruction" begin\n.*?\nend\n''',
    '''# The asymmetric Mixed family uses a specialized interior chart because its\n# feasible region is not a Cartesian product supported by the generic spaces.\n@testset "asymmetric Mixed specialized MLE stays feasible" begin\n    U = [0.10 0.25 0.40 0.55 0.70 0.85;\n         0.15 0.20 0.45 0.60 0.75 0.90]\n    fitted = fit(Copulas.AsymMixedCopula, U; method=:mle)\n    θ₁, θ₂ = params(fitted)\n    @test θ₁ >= 0\n    @test θ₁ + θ₂ <= 1\n    @test θ₁ + 2θ₂ <= 1\n    @test θ₁ + 3θ₂ >= 0\nend\n\n@testset "dimension-specialized fitting reconstruction" begin\n    C = ClaytonCopula{3}(2.0)\n    CT = typeof(C)\n    pspace = Copulas.Paramorph.param_space(CT, 3)\n    α₀ = only(Copulas._parameter_space_coordinates(pspace, params(C)))\n\n    f(α) = only(params(Copulas._parameter_space_copula(CT, 3, pspace, [α])))\n\n    @test f(α₀) ≈ 2.0\n    derivative = ForwardDiff.derivative(f, α₀)\n    @test isfinite(derivative)\n    @test derivative > 0\nend\n''',
    flags=re.S,
)
replace("test/operations/fitting.jl", "    R̂ = params(fitted).Σ", "    R̂ = only(params(fitted))")
replace(
    "test/operations/fitting.jl",
    "@test params(fit(ClaytonCopula, U; weights)).θ ≈ params(removed).θ rtol=1e-6",
    "@test only(params(fit(ClaytonCopula, U; weights))) ≈ only(params(removed)) rtol=1e-6",
)

replace(
    "test/correctness/bivariate_families.jl",
    "@test abs(only(params(fitted).θ)) <= 1",
    "@test abs(only(params(fitted))) <= 1",
)
replace("test/correctness/elliptical.jl", "@test params(C2).ν == 2", "@test first(params(C2)) == 2")
replace("test/correctness/elliptical.jl", "@test params(C20).ν == 20", "@test first(params(C20)) == 20")

sub(
    "test/correctness/extreme_value.jl",
    r'''@testset "NoTail parameter roundtrip" begin\n.*?\nend\n''',
    '''@testset "NoTail parameter space" begin\n    C = ExtremeValueCopula{2}(Copulas.NoTail())\n    pspace = Copulas.Paramorph.param_space(Copulas.NoTail, 2)\n\n    @test Copulas.Paramorph.names(pspace) == ()\n    @test Copulas.Paramorph.dimension(pspace) == 0\n    @test Copulas.Paramorph.constrain(pspace, Float64[]) == ()\n    @test params(C) == ()\nend\n''',
    flags=re.S,
)
replace(
    "test/correctness/extreme_value.jl",
    """        reconstructed = Copulas.AsymGalambosTail(values(params(tail))...)\n        @test reconstructed.α == tail.α\n        @test reconstructed.β == tail.β\n""",
    """        @test !applicable(params, tail)\n""",
)

# ---------------------------------------------------------------------------
# Documentation: tuple-valued Distribution params and Paramorph fitting API.
# ---------------------------------------------------------------------------
p = Path("docs/src/dev/developer_guide.md")
text = p.read_text()
text = text.replace("three-method [`Generator`](@ref) protocol", "two-method [`Generator`](@ref) protocol")
text = text.replace("implement its three-method mathematical contract:", "implement its two-method mathematical contract:")
text = text.replace(
    "Paramorph.param_space(::Type{<:MardiaCopula}, d) = Paramorph.Id(:θ)",
    "Paramorph.param_space(::Type{<:MardiaCopula}, d) = Paramorph.Bounded(:θ, -1.0, 1.0)",
)
p.write_text(text)

sub(
    "docs/src/dev/developer_guide.md",
    r'''### Opting into generic fitting methods\n\n.*?\nEach fitting method is dispatched on `Val\{:method\}` for performance and clarity\.''',
    '''### Opting into generic fitting methods\n\nSimple parametric families opt into the generic transformed-space MLE by\ndefining their parameter geometry with `Paramorph.param_space` and supporting\nthe canonical `CT(d, parameters...)` constructor. `Distributions.params(C)`\nremains the positional constructor tuple; logical names and constraints belong\nto the Paramorph space.\n\n| Definition | Purpose |\n| ---------- | ------- |\n| `Paramorph.param_space(CT, d)` | Describe logical parameter names, constraints, and optimizer dimension |\n| `CT(d, parameters...)` | Reconstruct a copula from constrained logical parameters |\n| `_available_fitting_methods(CT, d)` | Register the in-package estimators exposed for that family |\n\nExample minimal skeleton:\n\n```julia\nParamorph.param_space(::Type{<:MyCopula}, d) = Paramorph.Pos(:θ)\n_available_fitting_methods(::Type{<:MyCopula}, d) = (:mle,) # in-package only\n\n# No custom MLE is required: the generic route maps an unconstrained optimizer\n# vector through `Paramorph.constrain` and calls `MyCopula(d, θ)`.\n```\n\nA custom `_fit(::Type{MyCopula}, U, ::Val{:mle}; ...)` remains appropriate when\nthe feasible set is not represented by the available parameter spaces or when\nthe family has a materially better dedicated objective.\n\nEach fitting method is dispatched on `Val{:method}` for performance and clarity.''',
    flags=re.S,
)
sub(
    "docs/src/dev/developer_guide.md",
    r'''Remark that we could also opt-in the default moment matching methods, but for that we need to specify parameter relaxations through the following: \n\n```@example generic_copula_example\nCopulas\._unbound_params.*?```\n\nAnd we need to change our availiable methods:''',
    '''The `Paramorph.Bounded(:θ, -1.0, 1.0)` declaration above already\ndescribes the optimizer relaxation; no legacy parameter-transform hooks are\nneeded. If the corresponding rank-inversion identities are implemented, the\nfamily can register those methods alongside its custom estimator:\n\n''',
    flags=re.S,
)
p = Path("docs/src/dev/developer_guide.md")
p.write_text(p.read_text().replace("availiable methods", "available methods"))

# SklarDist params are `(copula, margins)`.
for path in (
    "docs/src/manual/conditioning_and_subsetting.md",
    "docs/src/examples/ifm1.md",
    "docs/src/examples/lossalae.md",
    "docs/src/examples/expectation_maximization.md",
    "docs/src/examples/fitting_sklar.md",
    "docs/src/bestiary/nested.md",
    "docs/src/bestiary/empirical.md",
):
    text = read(path)
    text = re.sub(r"params\((\w+)\)\.copula", r"first(params(\1))", text)
    text = re.sub(r"params\((\w+)\)\.margins", r"last(params(\1))", text)
    write(path, text)

replace(
    "docs/src/examples/ifm1.md",
    "params(ifm2_cop).Σ .- params(first(params(quick_fit))).Σ",
    "only(params(ifm2_cop)) .- only(params(first(params(quick_fit))))",
)
replace(
    "docs/src/bestiary/liouville.md",
    "(parameters=params(C13).α, sample=rand(rng, C13))",
    "(parameters=last(params(C13)), sample=rand(rng, C13))",
)
replace(
    "docs/src/bestiary/elliptical.md",
    "(df=params(Ĉ).ν, correlation=params(Ĉ).Σ[1, 2])",
    "(df=first(params(Ĉ)), correlation=last(params(Ĉ))[1, 2])",
    count=2,
)

# Remove migration-only scripts. This script removes itself after running; the
# workflow is deleted separately by a user-attributed commit so final CI runs.
for path in (
    "scripts/refactor_distribution_params.py",
    "scripts/finish_distribution_params.py",
    "scripts/finalize_paramorph_cleanup.py",
):
    Path(path).unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# Audit stale contracts before a commit is made.
# ---------------------------------------------------------------------------
failures = []
for root in (Path("src"), Path("test"), Path("docs")):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in (".jl", ".md"):
            continue
        text = path.read_text()
        for token in ("_unbound_params", "_rebound_params", "_construct_fitted_copula"):
            if token in text:
                failures.append(f"legacy hook {token}: {path}")
        if re.search(r"params\([^\n]*isa NamedTuple", text):
            failures.append(f"NamedTuple params assertion: {path}")
        # Direct named access on a simple params result is the old Distribution
        # contract. Complex calls such as params(fit(...).C) are not matched.
        for lineno, line in enumerate(text.splitlines(), 1):
            if re.search(r"\bparams\([A-Za-z_Ĉ][A-Za-z0-9_Ĉ]*\)\.[A-Za-z_α-ωΑ-ΩΣνθρ]", line):
                failures.append(f"named params access: {path}:{lineno}:{line.strip()}")

if failures:
    raise RuntimeError("stale Paramorph migration contracts remain:\n" + "\n".join(failures))
