from pathlib import Path

p = Path('src/NestedArchimedeanCopula.jl')
s = p.read_text()

# Constructor documentation: structural validation only.
old = '''The constructor validates each parent -> child edge at the actual number of
leaves below the child.  Copulas.jl accepts only generator pairs for which an
analytical nesting certificate is implemented.  A certified pair with invalid
parameters raises `DomainError`; a pair for which no certificate is implemented
raises `ArgumentError` rather than silently constructing an unvalidated copula.
'''
new = '''The constructor validates only the tree structure and dimension placement. It
intentionally does **not** validate the mathematical nesting relation between
parent and child generators. Expert users may therefore construct trees outside
the built-in fitting geometry. Template fitting performs its own upfront
validation by constructing a nesting-aware Paramorph parameter space.
'''
assert old in s
s = s.replace(old, new, 1)

# Replace the old three-state certificate table (including BB rules) by the
# deliberately small table of fitting geometries.
start = s.index('# ---- Nesting validity')
end = s.index('# ---- Unified keyword constructor', start)
geometry = r'''# ---- Nesting geometry for template fitting ---------------------------------
#
# Public construction is deliberately permissive: these rules are NOT constructor
# validation. They describe only the parameter geometries currently available to
# template fitting. The supported non-trivial rules are restricted to the standard
# one-parameter generators; extending this table to multi-parameter/BB families is
# intentionally left open for contributions.

_nested_fit_rule(::Generator, ::Generator) = nothing
_nested_fit_rule(::IndependentGenerator, ::Generator) = :free
_nested_fit_rule(::AMHGenerator, ::AMHGenerator) = :greater
_nested_fit_rule(::ClaytonGenerator, ::ClaytonGenerator) = :greater
_nested_fit_rule(::FrankGenerator, ::FrankGenerator) = :greater
_nested_fit_rule(::GumbelGenerator, ::GumbelGenerator) = :greater
_nested_fit_rule(::GumbelBarnettGenerator, ::GumbelBarnettGenerator) = :lower
_nested_fit_rule(::InvGaussianGenerator, ::InvGaussianGenerator) = :greater
_nested_fit_rule(::JoeGenerator, ::JoeGenerator) = :greater
_nested_fit_rule(::AMHGenerator, ::ClaytonGenerator) = :amh_clayton

_nested_child(ch::Tuple) = ch[1]
_nested_child(ch::NestedArchimedeanCopula) = ch

function _unsupported_nested_fit_rule(parent::Generator, child::Generator)
    edge = "$(nameof(typeof(parent))) -> $(nameof(typeof(child)))"
    throw(ArgumentError(
        "template fitting has no Paramorph nesting geometry for $edge. " *
        "Built-in nesting geometries currently cover the standard one-parameter " *
        "generators (plus an independence parent and AMH -> Clayton); " *
        "multi-parameter/BB nesting rules are open for contributions. " *
        "NestedArchimedeanCopula construction itself remains permissive.",
    ))
end

'''
s = s[:start] + geometry + s[end:]

# Constructor no longer validates generator nesting.
s = s.replace('''    kids2 = Any[_place_dims(kids[i], kiddims[i]) for i in eachindex(kids)]
    _validate_nested_edges(G, kids2)
''', '''    kids2 = Any[_place_dims(kids[i], kiddims[i]) for i in eachindex(kids)]
''', 1)

# Rewrite the fixed-tree fitting geometry around Paramorph.DependentProduct.
start = s.index('function _push_nested_spaces!')
end = s.index('function _push_nested_parameter_values!', start)
space_code = r'''# Intrinsic scalar bounds used when a one-parameter generator participates in a
# nesting geometry. `parent_role=true` narrows families whose finite-dimensional
# standalone domain contains negative dependence but whose supported nesting rule
# is currently known only on the non-negative branch.
function _nested_scalar_bounds(G::AMHGenerator, dloc::Int, parent_role::Bool)
    lower = parent_role ? 0.0 : clamp(_find_critical_value_amh(dloc), -1, 1)
    return lower, 1.0
end
_nested_scalar_bounds(::ClaytonGenerator, dloc::Int, parent_role::Bool) =
    (parent_role ? 0.0 : -inv(dloc - 1), nothing)
_nested_scalar_bounds(::FrankGenerator, dloc::Int, parent_role::Bool) =
    ((parent_role || dloc > 2) ? 0.0 : nothing, nothing)
_nested_scalar_bounds(::GumbelGenerator, ::Int, ::Bool) = (1.0, nothing)
function _nested_scalar_bounds(::GumbelBarnettGenerator, dloc::Int, ::Bool)
    return 0.0, clamp(_find_critical_value_gumbelbarnett(dloc), 0, 1)
end
_nested_scalar_bounds(::InvGaussianGenerator, ::Int, ::Bool) = (0.0, nothing)
_nested_scalar_bounds(::JoeGenerator, ::Int, ::Bool) = (1.0, nothing)

function _nested_interval_space(name::Symbol, lower, upper)
    if lower === nothing && upper === nothing
        return Paramorph.Id(name)
    elseif upper === nothing
        return Paramorph.LowerClosed(name, lower)
    elseif lower === nothing
        throw(ArgumentError("upper-only nested parameter domains are not implemented"))
    end
    lower < upper || throw(ArgumentError(
        "nested fitting parameter $name has an empty or degenerate intrinsic interval [$lower, $upper]",
    ))
    return Paramorph.Bounded(name, lower, upper)
end

function _nested_standard_space(G::Generator, dloc::Int, name::Symbol; parent_role::Bool)
    bounds = try
        _nested_scalar_bounds(G, dloc, parent_role)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "template fitting currently provides nesting geometries only for standard " *
            "one-parameter generators; $(nameof(typeof(G))) is open for contributions",
        ))
    end
    return _nested_interval_space(name, bounds...)
end

function _nested_single_parameter_name(G::Generator, dloc::Int, tag::String)
    nms = Paramorph.names(_generator_space(G, dloc))
    length(nms) == 1 || return nothing
    return Symbol(tag, "_", only(nms))
end

function _nested_child_space(parent::Generator, child::Generator, dloc::Int,
                             name, tag::String, parent_name; parent_role::Bool)
    rule = _nested_fit_rule(parent, child)
    rule === nothing && _unsupported_nested_fit_rule(parent, child)

    if rule === :free
        # An actual IndependentGenerator has no parameter of its own and imposes
        # no cross-edge restriction. A flat child can therefore use any intrinsic
        # Paramorph space; a child that is itself a parent must have one of the
        # supported one-parameter nesting geometries for its outgoing edges.
        if parent_role
            name === nothing && _unsupported_nested_fit_rule(parent, child)
            return _nested_standard_space(child, dloc, name; parent_role=true)
        end
        return Paramorph.Prefixed(Symbol(tag), _generator_space(child, dloc))
    end

    name === nothing && _unsupported_nested_fit_rule(parent, child)
    parent_name === nothing && error("dependent nesting rule requires a parent parameter")
    lower, upper = _nested_scalar_bounds(child, dloc, parent_role)

    if rule === :greater
        return Paramorph.GreaterThan(name, parent_name; lower, upper)
    elseif rule === :lower
        return Paramorph.LowerThan(name, parent_name; lower, upper)
    elseif rule === :amh_clayton
        # For a non-independent AMH parent the classical sufficient nesting rule
        # is simply θ_child >= 1; because θ_parent < 1 this already implies the
        # cross-edge ordering, so no dynamic reference is needed here.
        lower = lower === nothing ? 1.0 : max(lower, 1.0)
        return _nested_interval_space(name, lower, upper)
    end
    error("unknown nested fitting rule $rule")
end

function _push_nested_fit_spaces!(spaces, C::NestedArchimedeanCopula, tag::String;
                                  parent=nothing, parent_name=nothing)
    dloc = _local_arity(C)
    current_name = _nested_single_parameter_name(C.G, dloc, tag)

    if parent === nothing
        if !(C.G isa IndependentGenerator)
            current_name === nothing && throw(ArgumentError(
                "template fitting a nested parent requires a supported one-parameter generator; " *
                "$(nameof(typeof(C.G))) is open for contributions",
            ))
            push!(spaces, _nested_standard_space(C.G, dloc, current_name; parent_role=true))
        end
    else
        push!(spaces, _nested_child_space(
            parent, C.G, dloc, current_name, tag, parent_name; parent_role=true))
    end

    for (i, ch) in enumerate(C.children)
        childtag = "$(tag)[$i]"
        if ch isa Tuple
            cc, ds = ch
            cd = max(length(ds), 2)
            cname = _nested_single_parameter_name(cc.G, cd, childtag)
            push!(spaces, _nested_child_space(
                C.G, cc.G, cd, cname, childtag, current_name; parent_role=false))
        else
            _push_nested_fit_spaces!(spaces, ch, childtag;
                                     parent=C.G, parent_name=current_name)
        end
    end
    return spaces
end

function Paramorph.param_space(C::NestedArchimedeanCopula)
    spaces = Paramorph.AbstractParameterSpace[]
    _push_nested_fit_spaces!(spaces, C, "G")
    return Paramorph.DependentProduct(Tuple(spaces))
end

'''
s = s[:start] + space_code + s[end:]

# Update fitting commentary and remove the old objective-time certificate. The
# default template map is valid by construction through its DependentProduct.
s = s.replace('''# NESTING VALIDITY: the DEFAULT parametrisation does not enforce the cross-node
# "inner at least as dependent as outer" condition (the constructor leaves it to
# the caller) — an unconstrained α only keeps each generator valid in its own
# family, so a fitted optimum CAN have inner θ < outer θ. To constrain it, pass a
# custom `reparam`/`init` that encodes the constraint (see fit()).
''', '''# NESTING VALIDITY: the DEFAULT template parametrisation uses a
# `Paramorph.DependentProduct`. Supported parent-child inequalities are therefore
# enforced by the coordinate map itself, while the public constructor remains
# intentionally permissive. Custom `reparam`/`init` maps remain user-defined and
# are responsible for the validity of the trees they produce.
''', 1)

cert_start = s.index('# Every constructible tree is a certified nesting')
fit_start = s.index('function _fit_nested(recon, α₀::AbstractVector, U; weights=nothing)', cert_start)
cert_end = fit_start
s = s[:cert_start] + '''# The template fitting chart is nesting-valid by construction; custom runtime
# parametrisations are an expert escape hatch and remain caller-responsible.
''' + s[cert_end:]
old_loss = '''    loss(α) = begin
        C = recon(α)
        _nested_certified(C) || return convert(eltype(α), Inf)
        return -_weighted_loglikelihood(C, U, weights)
    end
'''
assert old_loss in s
s = s.replace(old_loss, '''    loss(α) = -_weighted_loglikelihood(recon(α), U, weights)
''', 1)

s = s.replace('''  * **template** `C0`: a template instance whose tree shape (leaf layout, children
    blocks) and per-node generator families are kept fixed; only the scalar θ of
    each node is re-optimised, each inside its own family domain.
''', '''  * **template** `C0`: a template instance whose tree shape (leaf layout, children
    blocks) and per-node generator families are kept fixed. For the supported
    standard one-parameter nesting rules, Paramorph jointly constrains parent and
    child parameters so every finite optimiser coordinate maps to a valid nesting.
''', 1)

p.write_text(s)

# The old fitting-validation shim is obsolete: geometry now performs the job.
cp = Path('src/Copulas.jl')
cs = cp.read_text()
needle = '    include("NestedArchimedeanFitValidation.jl")\n'
assert needle in cs
cp.write_text(cs.replace(needle, '', 1))
Path('src/NestedArchimedeanFitValidation.jl').unlink()

# Correctness tests: construction is permissive; fitting geometry is restrictive.
tp = Path('test/correctness/nested_archimedean.jl')
ts = tp.read_text()
ts = ts.replace('''Copulas._nested_status(o::ImplicitTestGenerator, i::ImplicitTestGenerator, d::Int) =
    Copulas._nested_status(o.inner, i.inner, d)
''', '', 1)
start = ts.index('    @testset "nesting validity certificates" begin')
end = ts.index('    # -----------------------------------------------------------------------\n    # 3. Uncensored density', start)
new_test = '''    @testset "permissive construction and fitting geometry" begin
        P = Copulas.Paramorph

        # Construction validates structure only: even a mathematically invalid
        # Clayton ordering can be represented deliberately.
        bad = NestedArchimedeanCopula(ClaytonGenerator(5.0);
                children = [ClaytonCopula{2}(2.0)])
        @test bad isa NestedArchimedeanCopula{2}

        # The template fitting chart carries the ordering. The invalid template
        # therefore fails when mapped into that chart, before optimisation.
        pbad = P.param_space(bad)
        @test_throws DomainError P.unconstrain(pbad, Copulas._nested_parameter_values(bad))

        good = NestedArchimedeanCopula(ClaytonGenerator(2.0);
                children = [ClaytonCopula{2}(5.0)])
        pgood = P.param_space(good)
        α = P.unconstrain(pgood, Copulas._nested_parameter_values(good))
        rebuilt = Copulas._nested_from_coordinates(good, pgood, α)
        @test rebuilt.G.θ <= Copulas._nested_child(only(rebuilt.children)).G.θ

        # Unsupported multi-parameter nesting is still constructible, but the
        # fitting geometry is deliberately absent and invites contributions.
        bb = NestedArchimedeanCopula(Copulas.BB1Generator(1.0, 1.0);
                children = [ClaytonCopula{2}(2.0)])
        err = try
            P.param_space(bb)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("open for contributions", sprint(showerror, err))
    end

'''
ts = ts[:start] + new_test + ts[end:]
tp.write_text(ts)

# Replace the former proposal-time certificate tests with geometric guarantees.
tp = Path('test/operations/nested_fitting_validation.jl')
tp.write_text('''@testset "nested template fitting uses dependent Paramorph geometry" begin
    P = Copulas.Paramorph
    C0 = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(2.0);
        children=[ClaytonCopula{2}(5.0)],
    )
    p = P.param_space(C0)
    @test p isa P.DependentProduct
    α0 = P.unconstrain(p, Copulas._nested_parameter_values(C0))
    @test all(isfinite, α0)

    # Arbitrary unconstrained coordinates always reconstruct inside the supported
    # ordering, so no objective-time certificate/Inf barrier is required.
    for α in (zeros(2), [-4.0, -4.0], [3.0, -2.0], [-2.0, 3.0])
        candidate = Copulas._nested_from_coordinates(C0, p, α)
        parent = candidate.G.θ
        child = Copulas._nested_child(only(candidate.children)).G.θ
        @test parent >= 0
        @test child >= parent
    end

    # An invalid template remains constructible but is rejected before fitting.
    bad = NestedArchimedeanCopula(
        Copulas.ClaytonGenerator(5.0);
        children=[ClaytonCopula{2}(2.0)],
    )
    U = rand(StableRNG(49_000), C0, 20)
    @test_throws DomainError fit(CopulaModel, bad, U)

    fitted = fit(C0, U)
    @test fitted.G.θ <= Copulas._nested_child(only(fitted.children)).G.θ
end
''')

# Extend the coefficient regression with the actual ordering guarantee.
tp = Path('test/operations/parameter_coefficients.jl')
ts = tp.read_text()
needle = '''    pnested = P.param_space(Cnested)
    @test P.dimension(pnested) == 3
    @test StatsBase.dof(Cnested) == 3
'''
if needle in ts:
    repl = needle + '''    @test pnested isa P.DependentProduct
    αnested = P.unconstrain(pnested, Copulas._nested_parameter_values(Cnested))
    Cnested0 = Copulas._nested_from_coordinates(Cnested, pnested, zeros(3))
    @test all(Copulas._nested_child(ch).G.θ >= Cnested0.G.θ for ch in Cnested0.children)
    @test P.unconstrain(pnested, P.constrain(pnested, αnested)) ≈ αnested
'''
    ts = ts.replace(needle, repl, 1)
tp.write_text(ts)

# Documentation: constructor permissive, fitting geometry deliberately limited.
dp = Path('docs/src/bestiary/nested.md')
ds = dp.read_text()
old = '''The constructor validates every parent-child nesting edge and distinguishes a
mathematically certified invalid edge from a pair for which Copulas.jl does not
have an analytical certificate.
'''
if old in ds:
    ds = ds.replace(old, '''The constructor validates the tree structure and dimension placement only. It does
not reject a tree because of its parent-child generator parameters. This is
deliberate: expert users can represent and inspect arbitrary nested structures.

Template fitting is stricter. It constructs a nesting-aware
`Paramorph.DependentProduct` before optimisation, so supported parent-child
inequalities are enforced by the parameter geometry itself. The built-in fitting
rules currently cover the standard one-parameter AMH, Clayton, Frank, Gumbel,
Gumbel-Barnett, inverse-Gaussian and Joe homogeneous nestings, an independent
parent, and AMH → Clayton. Multi-parameter/BB fitting geometries are not yet
implemented; contributions are welcome.
''', 1)
else:
    marker = '# Fitting\n'
    if marker in ds:
        ds = ds.replace(marker, '''## Nesting validity and fitting geometry

Construction is intentionally permissive with respect to mathematical nesting.
Template fitting, however, builds a nesting-aware `Paramorph.DependentProduct`
and rejects unsupported or out-of-region templates before optimisation. Built-in
fitting geometry currently covers standard one-parameter generator rules; BB and
other multi-parameter rules are open for contributions.

''' + marker, 1)
dp.write_text(ds)
