from pathlib import Path
import re
import subprocess

ROOT = Path('.')

def read(path):
    return Path(path).read_text()

def write(path, text):
    Path(path).write_text(text)

def replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f"missing anchor: {label}")
    return text.replace(old, new, 1)

def remove_params_methods(path):
    text = read(path)
    # Ordinary function definitions used by a few tails.
    while True:
        m = re.search(r'(?m)^function Distributions\.params\([^\n]*\)\n', text)
        if not m:
            break
        end = re.search(r'(?m)^end\s*$', text[m.end():])
        if not end:
            raise RuntimeError(f"unterminated params method in {path}")
        text = text[:m.start()] + text[m.end() + end.end():]
    # `= begin ... end` definitions.
    while True:
        m = re.search(r'(?m)^Distributions\.params\([^\n]*\)\s*=\s*begin\s*$', text)
        if not m:
            break
        end = re.search(r'(?m)^end\s*$', text[m.end():])
        if not end:
            raise RuntimeError(f"unterminated params begin block in {path}")
        text = text[:m.start()] + text[m.end() + end.end():]
    # One-line definitions.
    text = re.sub(r'(?m)^Distributions\.params\([^\n]*\)\s*=.*\n?', '', text)
    write(path, text)

# ---------------------------------------------------------------------------
# params belongs to Distribution. Paramorph owns parameter names / geometry.
# ---------------------------------------------------------------------------
p = Path('src/Copula.jl')
text = read(p)
anchor = 'Distributions.partype(C::Copula) = eltype(C)\n'
insert = '''Distributions.partype(C::Copula) = eltype(C)\n\nParamorph.param_space(C::Copula) = Paramorph.param_space(typeof(C), length(C))\n\nfunction Distributions.params(C::Copula)\n    p = Paramorph.param_space(C)\n    return map(Paramorph.names(p)) do name\n        value = getproperty(C, name)\n        return value isa AbstractArray ? copy(value) : value\n    end\nend\n'''
text = replace_once(text, anchor, insert, 'generic Copula params')
write(p, text)

# Generators and tails are not Distributions. Remove their params methods.
for folder in ('src/Generator', 'src/Tail'):
    for path in Path(folder).glob('*.jl'):
        remove_params_methods(path)

# Generator extension docs: only the mathematical generator contract is public.
p = Path('src/Generator.jl')
text = read(p)
text = text.replace('- `Distributions.params(G)`, returning a `NamedTuple` of its public parameters.\n', '')
text = text.replace('These methods are sufficient to construct', 'These methods are sufficient to construct')
text = text.replace('[`FrailtyGenerator`](@ref), `Distributions.params`.', '[`FrailtyGenerator`](@ref), `Paramorph.param_space`.')
text = re.sub(r'(?m)^_parameter_dof\(x::Generator\) = .*\n', '', text)
write(p, text)

p = Path('src/Tail.jl')
text = read(p)
text = re.sub(r'(?m)^_parameter_dof\(x::Tail\) = .*\n', '', text)
write(p, text)

# Archimedean and EV copulas read their logical parameters from their components.
p = Path('src/ArchimedeanCopula.jl')
text = read(p)
text = text.replace(
    'Distributions.params(C::ArchimedeanCopula) = Distributions.params(C.G)\n',
    '''function Distributions.params(C::ArchimedeanCopula)\n    p = Paramorph.param_space(C)\n    return map(Base.Fix1(getproperty, C.G), Paramorph.names(p))\nend\n''',
)
text = text.replace('and the three core methods `ϕ`, `max_monotony`, and\n`Distributions.params`:',
                    'and the two core methods `ϕ` and `max_monotony`:')
text = re.sub(r'(?m)^Distributions\.params\(::MyGenerator\) = \(;\)\n', '', text)
write(p, text)

p = Path('src/ExtremeValueCopula.jl')
text = read(p)
text = text.replace(
    'Distributions.params(C::ExtremeValueCopula) = Distributions.params(C.tail)\n',
    '''function Distributions.params(C::ExtremeValueCopula)\n    p = Paramorph.param_space(C)\n    return map(Base.Fix1(getproperty, C.tail), Paramorph.names(p))\nend\n''',
)
write(p, text)

# Archimax composes two component spaces, but params is still an ordinary tuple.
p = Path('src/ArchimaxCopula.jl')
text = read(p)
text = re.sub(
    r'(?ms)^Distributions\.params\(C::ArchimaxCopula\) = begin\n.*?^end\n',
    '''function Distributions.params(C::ArchimaxCopula)\n    d = length(C)\n    gp = Paramorph.param_space(typeof(C.gen), d)\n    tp = Paramorph.param_space(typeof(C.tail), d)\n    gvals = map(Base.Fix1(getproperty, C.gen), Paramorph.names(gp))\n    tvals = map(Base.Fix1(getproperty, C.tail), Paramorph.names(tp))\n    return (gvals..., tvals...)\nend\n''',
    text,
    count=1,
)
write(p, text)

# ---------------------------------------------------------------------------
# Field names should match the mathematical parameter names.
# ---------------------------------------------------------------------------
p = Path('src/EllipticalCopulas/TCopula.jl')
text = read(p)
text = re.sub(r'\bdf\b', 'ν', text)
text = re.sub(r'(?m)^Distributions\.params\(C::TCopula\) = .*\n', '', text)
write(p, text)

# Direct-field copulas are covered by generic params.
for path, typename in (
    ('src/EllipticalCopulas/GaussianCopula.jl', 'GaussianCopula'),
    ('src/MiscellaneousCopulas/PlackettCopula.jl', 'PlackettCopula'),
    ('src/MiscellaneousCopulas/RafteryCopula.jl', 'RafteryCopula'),
):
    p = Path(path)
    text = read(p)
    text = re.sub(rf'(?m)^Distributions\.params\([^\n]*::{typename}\) = .*\n', '', text)
    write(p, text)

# Structural distributions keep explicit tuple-valued params methods.
replacements = {
    'src/SklarDist.jl': [
        ('Distributions.params(S::SklarDist) = (copula=S.C, margins=S.m)',
         'Distributions.params(S::SklarDist) = (S.C, S.m)'),
    ],
    'src/MiscellaneousCopulas/FGMCopula.jl': [
        ('Distributions.params(C::FGMCopula) = (θ = collect(C.θ),)',
         'Distributions.params(C::FGMCopula) = (collect(C.θ),)'),
    ],
    'src/MiscellaneousCopulas/BetaCopula.jl': [
        ('Distributions.params(C::BetaCopula) = (ranks=C.ranks,)',
         'Distributions.params(C::BetaCopula) = (C.ranks,)'),
    ],
    'src/MiscellaneousCopulas/EmpiricalCopula.jl': [
        ('Distributions.params(C::EmpiricalCopula) = (u=C.u,)',
         'Distributions.params(C::EmpiricalCopula) = (C.u,)'),
    ],
    'src/LiouvilleCopula.jl': [
        ('Distributions.params(C::LiouvilleCopula) = (; G = C.G, α = C.α)',
         'Distributions.params(C::LiouvilleCopula) = (C.G, C.α)'),
        ('Distributions.params(C::LiouvilleCopula) = (; G=C.G, α=C.α)',
         'Distributions.params(C::LiouvilleCopula) = (C.G, C.α)'),
    ],
    'src/NestedArchimedeanCopula.jl': [
        ('Distributions.params(C::NestedArchimedeanCopula) =\n    (G=C.G, leaves=C.leafdims, children=C.children)',
         'Distributions.params(C::NestedArchimedeanCopula) =\n    (C.G, C.leafdims, C.children)'),
    ],
}
for path, pairs in replacements.items():
    p = Path(path)
    text = read(p)
    for old, new in pairs:
        text = text.replace(old, new)
    write(p, text)

# Parameter-free copulas get empty spaces, so generic params returns ().
for path, typename in (
    ('src/MiscellaneousCopulas/IndependentCopula.jl', 'IndependentCopula'),
    ('src/MiscellaneousCopulas/MCopula.jl', 'MCopula'),
    ('src/MiscellaneousCopulas/WCopula.jl', 'WCopula'),
):
    p = Path(path)
    text = read(p)
    text = re.sub(rf'(?m)^Distributions\.params\([^\n]*::{typename}[^\n]*\) = .*\n', '', text)
    marker = f'Paramorph.param_space(::Type{{<:{typename}}}, d) = ()\n'
    if marker not in text:
        # Insert after the first struct block / convenient location: before fitting marker when available.
        idx = text.find('# Fitting')
        if idx < 0:
            idx = len(text)
        text = text[:idx] + marker + '\n' + text[idx:]
    write(p, text)

# Parameter-free tails participate in EV/Archimax spaces without params methods.
for path, typename in (
    ('src/Tail/NoTail.jl', 'NoTail'),
    ('src/Tail/MTail.jl', 'MTail'),
):
    p = Path(path)
    text = read(p)
    marker = f'Paramorph.param_space(::Type{{{typename}}}, d) = ()\n'
    if marker not in text:
        # insert after struct declaration
        m = re.search(rf'(?m)^struct {typename}[^\n]* end\n', text)
        if not m:
            raise RuntimeError(f'cannot place empty space for {typename}')
        text = text[:m.end()] + marker + text[m.end():]
    write(p, text)

# AsymMixed has a deliberately local optimization chart, so params stays a
# distribution-level specialization on the containing EV copula.
p = Path('src/Tail/AsymMixedTail.jl')
text = read(p)
marker = 'Distributions.params(C::ExtremeValueCopula{D,<:AsymMixedTail}) where {D} =\n    (C.tail.θ₁, C.tail.θ₂)\n'
if marker not in text:
    insert_at = text.find('function _fit(')
    text = text[:insert_at] + marker + '\n' + text[insert_at:]
write(p, text)

# Marshall-Olkin: use the actual mathematical λ₁₂ name and keep a copula-level
# specialization because multivariate storage is a structural λ vector.
p = Path('src/Tail/MOTail.jl')
text = read(p)
text = text.replace('Paramorph.Pos(:λ₃)', 'Paramorph.Pos(:λ₁₂)')
marker = '''function Distributions.params(C::ExtremeValueCopula{d,<:MOTail}) where {d}\n    d == 2 && return _mo_bivariate_rates(C.tail)\n    return (copy(C.tail.λ),)\nend\n'''
if marker not in text:
    insert_at = text.find('function Paramorph.param_space')
    text = text[:insert_at] + marker + '\n' + text[insert_at:]
write(p, text)

# Hüsler-Reiss has two mathematically distinct representations in one tail
# type. Keep explicit distribution-level extraction rather than inventing a
# fake common parameter name.
p = Path('src/Tail/HuslerReissTail.jl')
text = read(p)
marker = '''Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:Real}}) where {D} =\n    (C.tail.parameter,)\nDistributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:AbstractMatrix}}) where {D} =\n    (copy(C.tail.parameter),)\n'''
if marker not in text:
    insert_at = text.find('Paramorph.param_space(')
    text = text[:insert_at] + marker + text[insert_at:]
write(p, text)

# extremal-t matrix form has a genuine Correlation space. Keep current storage
# for now; distribution-level methods expose tuples and scalar fitting geometry.
p = Path('src/Tail/tEVTail.jl')
text = read(p)
marker = '''Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{<:Any,<:Real}}) where {D} =\n    (C.tail.ν, C.tail.parameter)\nDistributions.params(C::ExtremeValueCopula{D,<:tEVTail{<:Any,<:AbstractMatrix}}) where {D} =\n    (C.tail.ν, copy(C.tail.parameter))\n'''
if marker not in text:
    insert_at = text.find('function Paramorph.param_space')
    text = text[:insert_at] + marker + text[insert_at:]
# Add matrix geometry; it is exactly positive df + correlation matrix.
scalar_block = '''function Paramorph.param_space(::Type{<:tEVTail{<:Any,<:Real}}, d)\n    lower = -inv(d - 1)\n    return (\n        Paramorph.Pos(:ν),\n        Paramorph.BoundedOpen(:ρ, lower, 1.0),\n    )\nend\n'''
if scalar_block in text and 'Correlation(:R, d)' not in text:
    text = text.replace(scalar_block, scalar_block + '''Paramorph.param_space(::Type{<:tEVTail{<:Any,<:AbstractMatrix}}, d) =\n    (Paramorph.Pos(:ν), Paramorph.Correlation(:R, d))\n''')
write(p, text)

# ---------------------------------------------------------------------------
# Fitting/inference consume tuple-valued Distributions.params.
# ---------------------------------------------------------------------------
p = Path('src/Fitting.jl')
text = read(p)
text = text.replace('Return the mathematical parameters of a copula or Sklar distribution as a\n`NamedTuple`, in canonical constructor order. For an ordinary parametric\ncopula, splatting `values(params(C))` into its documented typed constructor\nreconstructs the same model. Structural and empirical models document any\ndifferent reconstruction form explicitly.\n\nParameter names and values are public; concrete field names, storage-only type\nparameters and caches are not. A new in-package family must specialize this\nmethod before it can use generic fitting and display machinery. Missing\nspecializations therefore use Julia\'s normal `MethodError` rather than a\npackage-defined fallback exception.',
'''Return the mathematical parameters of a copula or Sklar distribution as a\n`Tuple`, in canonical constructor order, following the `Distributions.jl`\nconvention. Parameter names and constraints are supplied independently by\n`Paramorph.param_space`; `params` contains values only. For an ordinary\nparametric copula, splatting `params(C)` into its documented typed constructor\nreconstructs the same model.''')

old = '''function _parameter_space_value(p, θ::NamedTuple)\n    nms = Paramorph.names(p)\n    vals = ntuple(i -> getproperty(θ, nms[i]), length(nms))\n    isempty(vals) && return ()\n    return length(vals) == 1 ? vals[1] : vals\nend\n_parameter_space_coordinates(p, θ::NamedTuple) =\n    Paramorph.unconstrain(p, _parameter_space_value(p, θ))\n'''
new = '''_parameter_space_coordinates(p::Tuple, θ::Tuple) = Paramorph.unconstrain(p, θ)\n_parameter_space_coordinates(p, θ::Tuple) = Paramorph.unconstrain(p, only(θ))\n'''
if old not in text:
    raise RuntimeError('old parameter-space NamedTuple bridge not found')
text = text.replace(old, new, 1)

# Replace NamedTuple-key flattening with Paramorph names + tuple values.
start = text.index('function _flatten_params(params_nt::NamedTuple)')
end = text.index('\nend\n', start) + len('\nend\n')
new_flat = '''function _flatten_params(p, params::Tuple)\n    nm = String[]\n    θ = Any[]\n    nms = Paramorph.names(p)\n    length(nms) == length(params) || throw(DimensionMismatch(\n        "parameter-space names and distribution parameters have different lengths"))\n    for (name, value) in zip(nms, params)\n        _append_parameter!(nm, θ, value, String(name))\n    end\n    values = isempty(θ) ? Float64[] : collect(promote(float.(θ)...))\n    return nm, values\nend\n'''
text = text[:start] + new_flat + text[end:]

# Replace natural-parameter extraction so copula names come from Paramorph.
start = text.index('function _natural_parameters(D)')
end = text.index('\nend\n', start) + len('\nend\n')
new_natural = '''function _natural_parameters(D)\n    nm = String[]\n    θ = Any[]\n    if D isa SklarDist\n        if !(hasmethod(StatsBase.dof, Tuple{typeof(D.C)}) && iszero(StatsBase.dof(D.C)))\n            cp = Paramorph.param_space(D.C)\n            cn, cv = _flatten_params(cp, Distributions.params(D.C))\n            append!(nm, ("copula_" * name for name in cn))\n            append!(θ, cv)\n        end\n        for (i, margin) in pairs(D.m)\n            if applicable(Paramorph.param_space, margin)\n                mp = Paramorph.param_space(margin)\n                mn, mv = _flatten_params(mp, Distributions.params(margin))\n                append!(nm, ("margin_$(i)_" * name for name in mn))\n                append!(θ, mv)\n            else\n                _append_parameter!(nm, θ, Distributions.params(margin), "margin_$(i)")\n            end\n        end\n    elseif D isa Copula && applicable(Paramorph.param_space, D)\n        return _flatten_params(Paramorph.param_space(D), Distributions.params(D))\n    elseif !(hasmethod(StatsBase.dof, Tuple{typeof(D)}) && iszero(StatsBase.dof(D)))\n        _append_parameter!(nm, θ, Distributions.params(D), "")\n    end\n    values = isempty(θ) ? Float64[] : collect(promote(float.(θ)...))\n    return nm, values\nend\n'''
text = text[:start] + new_natural + text[end:]
write(p, text)

p = Path('src/Inference.jl')
text = read(p)
text = text.replace('θ::NamedTuple', 'θ::Tuple')
text = text.replace('parameters isa NamedTuple', 'parameters isa Tuple')
text = text.replace('_flatten_params(Distributions.params(\n            _parameter_space_copula(CT, d, pspace, αv)))[2]',
                    '_flatten_params(pspace, Distributions.params(\n            _parameter_space_copula(CT, d, pspace, αv)))[2]')
write(p, text)

# Nested generator fitting can read fields directly from Paramorph names.
p = Path('src/NestedArchimedeanCopula.jl')
text = read(p)
old = '''_generator_coordinates(G::Generator, dloc) =\n    _parameter_space_coordinates(_generator_space(G, dloc), Distributions.params(G))\n'''
new = '''function _generator_coordinates(G::Generator, dloc)\n    p = _generator_space(G, dloc)\n    values = map(Base.Fix1(getproperty, G), Paramorph.names(p))\n    return p isa Tuple ? Paramorph.unconstrain(p, values) :\n           Paramorph.unconstrain(p, only(values))\nend\n'''
if old in text:
    text = text.replace(old, new, 1)
write(p, text)

# Sklar dof and generic parameter counting no longer rely on NamedTuple.
p = Path('src/SklarDist.jl')
text = read(p)
text = text.replace('_parameter_dof(x::NamedTuple) = sum(_parameter_dof, values(x); init=0)',
                    '_parameter_dof(x::Tuple) = sum(_parameter_dof, x; init=0)')
write(p, text)

# ---------------------------------------------------------------------------
# Tests/docs using old NamedTuple params syntax.
# ---------------------------------------------------------------------------
for path in Path('test').rglob('*.jl'):
    text = read(path)
    text = text.replace('params(C).Σ', 'only(params(C))')
    text = text.replace('params(D) isa NamedTuple', 'params(D) isa Tuple')
    # BC2 statistical test uses named access.
    text = text.replace('a, b = params(C).a, params(C).b', 'a, b = params(C)')
    write(path, text)

p = Path('docs/src/dev/developer_guide.md')
text = read(p)
old = '''function Distributions.params(C::MyCopula) \n    # It will be assumed that `MyCopula{d}(params(C)...)` reproduces `C`.\n    # Keep `MyCopula(d, ...)` as a thin forwarder to this canonical constructor.\n    # The return value should be a NamedTuple. \n    return (θ = C.θ,) # Return a named tuple containing the parameters.\nend\n'''
new = '''Paramorph.param_space(::Type{<:MyCopula}, d) = Paramorph.Prob(:θ)\n# The generic `Distributions.params(::Copula)` returns `(C.θ,)`. Names and\n# constraints live in the Paramorph space, while `params` follows the\n# Distributions.jl tuple convention.\n'''
text = text.replace(old, new)
text = text.replace('`typeof(C)(values(params(C))...)`', '`typeof(C)(params(C)...)`')
write(p, text)

# Audit obvious violations. Remaining Copula/Sklar params are allowed; no
# Generator/Tail method may extend Distributions.params anymore.
for folder in ('src/Generator', 'src/Tail'):
    for path in Path(folder).glob('*.jl'):
        if 'Distributions.params(' in read(path):
            raise RuntimeError(f'non-Distribution params method remains in {path}')

# Keep the refactor itself clean.
subprocess.run(['git', 'diff', '--check'], check=True)

# Print remaining definitions/usages for review rather than hiding them.
subprocess.run(['git', 'grep', '-n', 'Distributions.params', '--', 'src', 'ext', 'test', 'docs'], check=False)
