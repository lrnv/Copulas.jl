from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def write_if_changed(path: Path, text: str) -> None:
    old = path.read_text()
    if text != old:
        path.write_text(text)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"expected migration block not found in {path}: {old[:80]!r}")
    text2 = text.replace(old, new, 1)
    path.write_text(text2)


def migrate_simple_paramorph_fields(text: str) -> str:
    """Translate the 0.0.2 `field::transform` shorthand inside @paramorph structs.

    The 0.0.3 DSL makes the storage type explicit. Complex structs whose fields
    were previously installed through schema overrides are handled separately
    below and are intentionally not guessed here.
    """
    lines = text.splitlines(keepends=True)
    numeric = None
    out = []
    header = re.compile(r"^Paramorph\.@paramorph\s+(\w+)\s+struct\b")
    scalar = ("asℝ", "asℝ₊", "nonnegative(", "closed_lower(", "bounded_interval(")
    matrix = ("correlation_matrix(", "positive_definite_matrix(", "variogram_matrix(")

    for line in lines:
        m = header.match(line)
        if m:
            numeric = m.group(1)
            out.append(line)
            continue

        if numeric is not None:
            stripped = line.strip()
            if stripped == "end":
                numeric = None
                out.append(line)
                continue
            if "~" not in line and "::" in line:
                indent = line[: len(line) - len(line.lstrip())]
                body = line.strip()
                name, annotation = body.split("::", 1)
                if annotation.startswith(scalar):
                    line = f"{indent}{name}::{numeric} ~ {annotation}\n"
                elif annotation.startswith(matrix):
                    line = f"{indent}{name}::Matrix{{{numeric}}} ~ {annotation}\n"
        out.append(line)
    return "".join(out)


# 1. Mechanical conversion of the ordinary field-level declarations.
for path in SRC.rglob("*.jl"):
    write_if_changed(path, migrate_simple_paramorph_fields(path.read_text()))

# 2. Geometry-dependent structs. These are explicit because the whole point of
#    the spike is to test that the 0.0.3 DSL can absorb the former Paramorph
#    override methods rather than silently retaining them.

replace_once(
    SRC / "MiscellaneousCopulas/FGMCopula.jl",
    '''Paramorph.@paramorph T struct FGMCopula{d,T<:Real} <: Copula{d}\n    θ::Vector{T}\nend\n\nParamorph.parameter_fields_override(::Type{<:FGMCopula}) = (:θ,)\nfunction _fgm_schema(d::Integer)\n    d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))\n    if d == 2\n        interval = Paramorph.bounded_interval(-1.0, 1.0)\n        return Paramorph.TransformVariables.as((\n            θ=Paramorph.TransformVariables.as(Vector, interval, 1),\n        ))\n    end\n    subsets = [Tuple(S) for k in 2:d for S in Combinatorics.combinations(1:d, k)]\n    corners = collect(Iterators.product(ntuple(_ -> (-1.0, 1.0), d)...))\n    corner_rows = reduce(vcat, [\n        reshape([-prod(corner[i] for i in subset) for subset in subsets], 1, :)\n        for corner in corners\n    ])\n    q = length(subsets)\n    identity = Matrix{Float64}(LinearAlgebra.I, q, q)\n    A = [corner_rows; identity; -identity]\n    return Paramorph.TransformVariables.as((θ=Paramorph.polytope(A, ones(size(A, 1))),))\nend\nParamorph.schema_override(::Type{<:FGMCopula{d}}, ::NamedTuple) where {d} =\n    _fgm_schema(d)\n''',
    '''function _fgm_geometry(d::Integer, ::Type{T}) where {T<:Real}\n    d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))\n    if d == 2\n        interval = Paramorph.bounded_interval(-one(T), one(T))\n        return Paramorph.TransformVariables.as(Vector, interval, 1)\n    end\n    subsets = [Tuple(S) for k in 2:d for S in Combinatorics.combinations(1:d, k)]\n    corners = collect(Iterators.product(ntuple(_ -> (-one(T), one(T)), d)...))\n    corner_rows = reduce(vcat, [\n        reshape([-prod(corner[i] for i in subset) for subset in subsets], 1, :)\n        for corner in corners\n    ])\n    q = length(subsets)\n    identity = Matrix{T}(LinearAlgebra.I, q, q)\n    A = [corner_rows; identity; -identity]\n    return Paramorph.polytope(A, ones(T, size(A, 1)))\nend\n\nParamorph.@paramorph T struct FGMCopula{d,T<:Real} <: Copula{d}\n    θ::Vector{T} ~ _fgm_geometry(d, T)\nend\n''',
)

replace_once(
    SRC / "Tail/TawnTail.jl",
    '''Paramorph.@paramorph T struct TawnTail{T<:Real} <: Tail\n    d::Int\n    dep::Vector{T}\n    weights::Vector{Vector{T}}\nend\n''',
    '''Paramorph.@paramorph T struct TawnTail{T<:Real} <: Tail\n    d::Int\n    dep::Vector{T} ~ Paramorph.TransformVariables.as(\n        Vector, Paramorph.closed_lower(one(T)), 2^d - d - 1,\n    )\n    weights::Vector{Vector{T}} ~ Paramorph.repeat_transform(\n        Paramorph.TransformVariables.UnitSimplex(2^(d - 1)), d,\n    )\nend\n''',
)
text = (SRC / "Tail/TawnTail.jl").read_text()
text = re.sub(
    r'''\nParamorph\.parameter_fields_override\(::Type\{<:TawnTail\}\) = \(:dep, :weights\)\nfunction _tawn_schema.*?Paramorph\.schema_override\(tail::TawnTail\{T\}, ::NamedTuple\) where \{T\} =\n    _tawn_schema\(tail\.d, T\)\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "Tail/TawnTail.jl", text)

replace_once(
    SRC / "Tail/AsymGalambosTail.jl",
    '''Paramorph.@paramorph T struct AsymGalambosTail{T<:Real} <: BivariatePickandsTail\n    d::Int\n    dep::Vector{T}\n    weights::Vector{Vector{T}}\nend\n''',
    '''Paramorph.@paramorph T struct AsymGalambosTail{T<:Real} <: BivariatePickandsTail\n    d::Int\n    dep::Vector{T} ~ Paramorph.TransformVariables.as(\n        Vector, Paramorph.nonnegative(), 2^d - d - 1,\n    )\n    weights::Vector{Vector{T}} ~ Paramorph.repeat_transform(\n        Paramorph.TransformVariables.UnitSimplex(2^(d - 1)), d,\n    )\nend\n''',
)
text = (SRC / "Tail/AsymGalambosTail.jl").read_text()
text = re.sub(
    r'''\nParamorph\.parameter_fields_override\(::Type\{<:AsymGalambosTail\}\) = \(:dep, :weights\)\nfunction _asymgalambos_schema.*?Paramorph\.schema_override\(tail::AsymGalambosTail, ::NamedTuple\) =\n    _asymgalambos_schema\(tail\.d\)\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "Tail/AsymGalambosTail.jl", text)

replace_once(
    SRC / "Tail/AsymMixedTail.jl",
    '''Paramorph.@paramorph T struct AsymMixedTail{T<:Real} <: BivariatePickandsTail\n    θ₁::T\n    θ₂::T\nend\n''',
    '''Paramorph.@paramorph T struct AsymMixedTail{T<:Real} <: BivariatePickandsTail\n    θ₁::T\n    θ₂::T\n    @geometry ((θ₁, θ₂) ~ Paramorph.asymmetric_mixed())\nend\n''',
)
text = (SRC / "Tail/AsymMixedTail.jl").read_text()
text = re.sub(
    r'''\nParamorph\.parameter_fields_override\(::Type\{<:AsymMixedTail\}\) = \(:θ₁, :θ₂\)\nParamorph\.schema_override\(::Type\{<:AsymMixedTail\}, ::NamedTuple\) =\n    Paramorph\.asymmetric_mixed\(\)\n''',
    "\n",
    text,
)
write_if_changed(SRC / "Tail/AsymMixedTail.jl", text)

replace_once(
    SRC / "Tail/BC2Tail.jl",
    '''Paramorph.@paramorph T struct BC2Tail{T<:Real} <: DiscreteSpectralPickandsTail\n    d::Int\n    a::Vector{T}\nend\n''',
    '''Paramorph.@paramorph T struct BC2Tail{T<:Real} <: DiscreteSpectralPickandsTail\n    d::Int\n    a::Vector{T} ~ Paramorph.TransformVariables.as(\n        Vector, Paramorph.bounded_interval(zero(T), one(T)), d,\n    )\nend\n''',
)
text = (SRC / "Tail/BC2Tail.jl").read_text()
text = re.sub(
    r'''\nParamorph\.parameter_fields_override\(::Type\{<:BC2Tail\}\) = \(:a,\)\nfunction _bc2_schema.*?Paramorph\.schema_override\(tail::BC2Tail\{T\}, ::NamedTuple\) where \{T\} =\n    _bc2_schema\(tail\.d, T\)\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "Tail/BC2Tail.jl", text)

# Marshall-Olkin needs one extra direct-construction invariant in addition to
# the ordinary nonnegative-vector chart. Keep that invariant in an opaque
# transform rather than in a Paramorph method extension.
replace_once(
    SRC / "Tail/MOTail.jl",
    '''Paramorph.@paramorph T struct MOTail{T<:Real} <: DiscreteSpectralPickandsTail\n    d::Int\n    λ::Vector{T}\nend\n''',
    '''function _mo_geometry(d::Integer, ::Type{T}) where {T<:Real}\n    d >= 2 || throw(ArgumentError("Marshall-Olkin dimension must be at least two"))\n    base = Paramorph.TransformVariables.as(\n        Vector, Paramorph.nonnegative(), 2^d - 1,\n    )\n    forward = identity\n    function backward(λ)\n        subsets = _nonempty_subsets(d)\n        length(λ) == length(subsets) || throw(DimensionMismatch(\n            "expected $(length(subsets)) shock intensities for dimension $d",\n        ))\n        totals = zeros(eltype(λ), d)\n        @inbounds for (k, S) in enumerate(subsets), i in S\n            totals[i] += λ[k]\n        end\n        all(>(zero(eltype(totals))), totals) || throw(DomainError(\n            λ, "every Marshall-Olkin margin must have positive total shock rate",\n        ))\n        return λ\n    end\n    return Paramorph.joint_transform(base, forward, backward)\nend\n\nParamorph.@paramorph T struct MOTail{T<:Real} <: DiscreteSpectralPickandsTail\n    d::Int\n    λ::Vector{T} ~ _mo_geometry(d, T)\nend\n''',
)
text = (SRC / "Tail/MOTail.jl").read_text()
text = re.sub(
    r'''\nParamorph\.parameter_fields_override\(::Type\{<:MOTail\}\) = \(:λ,\)\nfunction _mo_schema.*?Paramorph\.schema_override\(tail::MOTail, ::NamedTuple\) = _mo_schema\(tail\.d\)\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "Tail/MOTail.jl", text)

# 3. Context-sensitive scalar chart: the family owns the mathematical
#    consequence of parent dimension; the parent only forwards `dimension`.
for rel, old, new in [
    (
        "Generator/ClaytonGenerator.jl",
        "θ::T ~ closed_lower(get(context, :lower, -one(T)))",
        "θ::T ~ closed_lower(haskey(context, :dimension) ? -inv(T(context.dimension - 1)) : -one(T))",
    ),
    (
        "Generator/AMHGenerator.jl",
        "θ::T ~ bounded_interval(\n        get(context, :lower, -one(T)), one(T),\n    )",
        "θ::T ~ bounded_interval(\n        haskey(context, :dimension) ? T(clamp(_find_critical_value_amh(context.dimension), -1, 1)) : -one(T),\n        one(T),\n    )",
    ),
    (
        "Generator/GumbelBarnettGenerator.jl",
        "θ::T ~ bounded_interval(\n        zero(T), get(context, :upper, one(T)),\n    )",
        "θ::T ~ bounded_interval(\n        zero(T),\n        haskey(context, :dimension) ? T(clamp(_find_critical_value_gumbelbarnett(context.dimension), 0, 1)) : one(T),\n    )",
    ),
]:
    replace_once(SRC / rel, old, new)

# Frank is unconstrained only in dimension two; in higher dimensions the
# generator must stay on the nonnegative branch.
replace_once(
    SRC / "Generator/FrankGenerator.jl",
    "θ::T ~ asℝ",
    "θ::T ~ (get(context, :dimension, 2) == 2 ? asℝ : nonnegative())",
)
text = (SRC / "Generator/FrankGenerator.jl").read_text()
text = re.sub(
    r'''\nfunction Paramorph\.schema_override\(\n    ::Type\{<:FrankGenerator\}, context::NamedTuple,\n\)\n    get\(context, :dimension, 2\) == 2 && return nothing\n    return Paramorph\.TransformVariables\.as\(\(θ=Paramorph\.nonnegative\(\),\)\)\nend\n''',
    "\n",
    text,
)
write_if_changed(SRC / "Generator/FrankGenerator.jl", text)

# 4. Dimension guards that were formerly schema_override methods are ordinary
#    geometry expressions now.
replace_once(
    SRC / "EllipticalCopulas/GaussianCopula.jl",
    "Σ::Matrix{T} ~ correlation_matrix(d)",
    "Σ::Matrix{T} ~ correlation_matrix(d >= 2 ? d : throw(ArgumentError(\"a public copula requires dimension d ≥ 2; got d=$d\")))",
)
text = (SRC / "EllipticalCopulas/GaussianCopula.jl").read_text()
text = re.sub(
    r'''\nfunction Paramorph\.schema_override\(::Type\{<:GaussianCopula\{d\}\}, ::NamedTuple\) where \{d\}.*?return Paramorph\.schema_override\(T, context\)\nend\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "EllipticalCopulas/GaussianCopula.jl", text)

replace_once(
    SRC / "EllipticalCopulas/TCopula.jl",
    "Σ::Matrix{T} ~ correlation_matrix(d)",
    "Σ::Matrix{T} ~ correlation_matrix(d >= 2 ? d : throw(ArgumentError(\"a public copula requires dimension d ≥ 2; got d=$d\")))",
)
text = (SRC / "EllipticalCopulas/TCopula.jl").read_text()
text = re.sub(
    r'''\nfunction Paramorph\.schema_override\(::Type\{<:TCopula\{d\}\}, ::NamedTuple\) where \{d\}.*?return Paramorph\.schema_override\(T, context\)\nend\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "EllipticalCopulas/TCopula.jl", text)

replace_once(
    SRC / "MiscellaneousCopulas/PlackettCopula.jl",
    "θ::P ~ nonnegative()",
    "θ::P ~ (d == 2 ? nonnegative() : throw(DimensionMismatch(\"PlackettCopula is only defined in dimension 2\")))",
)
text = (SRC / "MiscellaneousCopulas/PlackettCopula.jl").read_text()
text = re.sub(
    r'''\nfunction Paramorph\.schema_override\(::Type\{<:PlackettCopula\{d\}\}, ::NamedTuple\) where \{d\}.*?end\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "MiscellaneousCopulas/PlackettCopula.jl", text)

replace_once(
    SRC / "MiscellaneousCopulas/RafteryCopula.jl",
    "θ::P ~ bounded_interval(zero(P), one(P))",
    "θ::P ~ (d >= 2 ? bounded_interval(zero(P), one(P)) : throw(ArgumentError(\"a public copula requires dimension d ≥ 2; got d=$d\")))",
)
text = (SRC / "MiscellaneousCopulas/RafteryCopula.jl").read_text()
text = re.sub(
    r'''\nfunction Paramorph\.schema_override\(::Type\{<:RafteryCopula\{d\}\}, ::NamedTuple\) where \{d\}.*?end\n''',
    "\n",
    text,
    flags=re.S,
)
write_if_changed(SRC / "MiscellaneousCopulas/RafteryCopula.jl", text)

# 5. Point the spike at the unreleased Paramorph 0.0.3 branch.
project = ROOT / "Project.toml"
text = project.read_text().replace('Paramorph = "0.0.2"', 'Paramorph = "0.0.3"')
if "[sources]" not in text:
    text = text.replace(
        '\n[weakdeps]\n',
        '\n[sources]\nParamorph = {url = "https://github.com/lrnv/Paramorph.jl", rev = "paramorph-0.0.3-dsl"}\n\n[weakdeps]\n',
        1,
    )
project.write_text(text)

docs_project = ROOT / "docs/Project.toml"
text = docs_project.read_text().replace('Paramorph = "0.0.2"', 'Paramorph = "0.0.3"')
if "[sources]" not in text:
    text += '\n[sources]\nParamorph = {url = "https://github.com/lrnv/Paramorph.jl", rev = "paramorph-0.0.3-dsl"}\n'
docs_project.write_text(text)

# Remove the obsolete 0.0.2 parser comment.
copulas = SRC / "Copulas.jl"
text = copulas.read_text().replace(
    '''    # Paramorph 0.0.2 recognizes transformation identifiers syntactically in\n    # `@paramorph` fields. Keep private local aliases so the package itself can\n    # remain imported rather than brought wholesale into this namespace.\n''',
    '''    # Local aliases keep geometry declarations compact without importing the\n    # whole Paramorph namespace into Copulas.jl.\n''',
)
copulas.write_text(text)

print("Paramorph 0.0.3 mechanical migration applied")
