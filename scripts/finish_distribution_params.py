from pathlib import Path
import re
import subprocess


def r(path): return Path(path).read_text()
def w(path, text): Path(path).write_text(text)

# ---------------------------------------------------------------------------
# Generator.jl: no Distributions.params on non-distributions.
# ---------------------------------------------------------------------------
p = Path('src/Generator.jl')
text = r(p)
for line in (
    'Distributions.params(::MarkerGenerator) = (;)\n',
    'Distributions.params(G::𝒲) = (X=G.X, order=G.order)\n',
    'Distributions.params(G::TiltedGenerator) = (Distributions.params(G.G)..., sJ = G.sJ)\n',
    'Distributions.params(G::FrailtyGenerator) = (F=G.F,)\n',
):
    text = text.replace(line, '')
w(p, text)

# These structural generator wrappers still need Distribution-level params when
# they are contained in an Archimedean copula.
p = Path('src/ArchimedeanCopula.jl')
text = r(p)
anchor = '''function Distributions.params(C::ArchimedeanCopula)\n    p = Paramorph.param_space(C)\n    return map(Base.Fix1(getproperty, C.G), Paramorph.names(p))\nend\n'''
extra = anchor + '''Distributions.params(::ArchimedeanCopula{d,<:MarkerGenerator}) where {d} = ()\nDistributions.params(C::ArchimedeanCopula{d,<:𝒲}) where {d} = (C.G.X, C.G.order)\nDistributions.params(C::ArchimedeanCopula{d,<:TiltedGenerator}) where {d} =\n    (C.G.G, C.G.p, C.G.sJ)\nDistributions.params(C::ArchimedeanCopula{d,<:FrailtyGenerator}) where {d} = (C.G.F,)\n'''
if anchor not in text:
    raise RuntimeError('Archimedean params anchor missing')
text = text.replace(anchor, extra, 1)
w(p, text)

# Internal numeric-type discovery may inspect component fields, but does not
# turn Generator/Tail into Distributions just to do so.
p = Path('src/utils.jl')
text = r(p)
anchor = '_parameter_eltype(x::Distributions.Distribution) = float(Distributions.partype(x))\n'
insert = anchor + '_parameter_eltype(x::Generator) = _parameter_eltype(fieldvalues(x))\n_parameter_eltype(x::Tail) = _parameter_eltype(fieldvalues(x))\n'
if insert not in text:
    text = text.replace(anchor, insert, 1)
w(p, text)

# ---------------------------------------------------------------------------
# Mathematical field names for polymorphic EV tails.
# ---------------------------------------------------------------------------
p = Path('src/Tail/HuslerReissTail.jl')
text = r(p)
old = '''struct HuslerReissTail{P} <: OneParameterPickandsTail\n    parameter::P\n    function HuslerReissTail(Γ::AbstractMatrix)\n'''
new = '''struct HuslerReissTail{P} <: OneParameterPickandsTail\n    θ::Union{Nothing,P}\n    Γ::Union{Nothing,P}\n    function HuslerReissTail(Γ::AbstractMatrix)\n'''
if old not in text: raise RuntimeError('HR struct anchor missing')
text = text.replace(old, new, 1)
text = text.replace('return new{Matrix{Float64}}(G)', 'return new{Matrix{Float64}}(nothing, G)', 1)
text = text.replace('return new{typeof(θf)}(θf)', 'return new{typeof(θf)}(θf, nothing)', 1)
text = text.replace('iszero(tail.parameter)', 'iszero(something(tail.θ))')
text = text.replace('isinf(tail.parameter)', 'isinf(something(tail.θ))')
text = text.replace('all(iszero, tail.parameter)', 'all(iszero, something(tail.Γ))')
text = text.replace('d == size(tail.parameter, 1)', 'd == size(something(tail.Γ), 1)')
# Remove scalar specialization: generic ExtremeValue params now reads :θ.
text = re.sub(r'(?m)^Distributions\.params\(C::ExtremeValueCopula\{D,<:HuslerReissTail\{<:Real\}\}\) where \{D\} =\n\s*\(C\.tail\.parameter,\)\n', '', text)
text = text.replace(
    'Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:AbstractMatrix}}) where {D} =\n    (copy(C.tail.parameter),)',
    'Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:AbstractMatrix}}) where {D} =\n    (copy(something(C.tail.Γ)),)',
)
text = text.replace('_hr_theta(tail::HuslerReissTail{<:Real}) = tail.parameter',
                    '_hr_theta(tail::HuslerReissTail{<:Real}) = something(tail.θ)')
text = text.replace('_hr_theta(tail::HuslerReissTail{<:AbstractMatrix}) = 2 / sqrt(tail.parameter[1, 2])',
                    '_hr_theta(tail::HuslerReissTail{<:AbstractMatrix}) = 2 / sqrt(something(tail.Γ)[1, 2])')
text = text.replace('γ = abs2(2 / tail.parameter)', 'γ = abs2(2 / something(tail.θ))')
text = text.replace('_hr_variogram(tail::HuslerReissTail{<:AbstractMatrix}, ::Int) = tail.parameter',
                    '_hr_variogram(tail::HuslerReissTail{<:AbstractMatrix}, ::Int) = something(tail.Γ)')
if 'tail.parameter' in text or 'C.tail.parameter' in text:
    raise RuntimeError('HuslerReissTail still uses ambiguous parameter field')
w(p, text)

p = Path('src/Tail/tEVTail.jl')
text = r(p)
old = '''struct tEVTail{T,P} <: BivariatePickandsTail\n    ν::T\n    parameter::P\n    function tEVTail(ν::Real, ρ::Real)\n'''
new = '''struct tEVTail{T,P} <: BivariatePickandsTail\n    ν::T\n    ρ::Union{Nothing,P}\n    R::Union{Nothing,P}\n    function tEVTail(ν::Real, ρ::Real)\n'''
if old not in text: raise RuntimeError('tEV struct anchor missing')
text = text.replace(old, new, 1)
text = text.replace('return new{typeof(νT),typeof(ρT)}(νT, ρT)',
                    'return new{typeof(νT),typeof(ρT)}(νT, ρT, nothing)', 1)
text = text.replace('return new{typeof(νf),typeof(RF)}(νf, RF)',
                    'return new{typeof(νf),typeof(RF)}(νf, nothing, RF)', 1)
text = text.replace('isone(tail.parameter)', 'isone(something(tail.ρ))')
text = text.replace('all(isone, tail.parameter)', 'all(isone, something(tail.R))')
text = text.replace('d >= 2 && tail.parameter > -inv(d - 1)',
                    'd >= 2 && something(tail.ρ) > -inv(d - 1)')
text = text.replace('d == size(tail.parameter, 1)', 'd == size(something(tail.R), 1)')
# Both scalar and matrix forms now have fields matching Paramorph.names, so the
# generic EV params method covers them.
text = re.sub(r'(?m)^Distributions\.params\(C::ExtremeValueCopula\{D,<:tEVTail\{<:Any,<:Real\}\}\) where \{D\} =\n\s*\(C\.tail\.ν, C\.tail\.parameter\)\n', '', text)
text = re.sub(r'(?m)^Distributions\.params\(C::ExtremeValueCopula\{D,<:tEVTail\{<:Any,<:AbstractMatrix\}\}\) where \{D\} =\n\s*\(C\.tail\.ν, copy\(C\.tail\.parameter\)\)\n', '', text)
text = text.replace('_tev_rho(tail::tEVTail{<:Any,<:Real}) = tail.parameter',
                    '_tev_rho(tail::tEVTail{<:Any,<:Real}) = something(tail.ρ)')
text = text.replace('_tev_rho(tail::tEVTail{<:Any,<:AbstractMatrix}) = tail.parameter[1, 2]',
                    '_tev_rho(tail::tEVTail{<:Any,<:AbstractMatrix}) = something(tail.R)[1, 2]')
text = text.replace('ρ = tail.parameter', 'ρ = something(tail.ρ)')
text = text.replace('_tev_correlation(tail::tEVTail{<:Any,<:AbstractMatrix}, ::Int) = tail.parameter',
                    '_tev_correlation(tail::tEVTail{<:Any,<:AbstractMatrix}, ::Int) = something(tail.R)')
if 'tail.parameter' in text or 'C.tail.parameter' in text:
    raise RuntimeError('tEVTail still uses ambiguous parameter field')
w(p, text)

# ---------------------------------------------------------------------------
# Remaining source code must consume tuple params, never NamedTuple properties.
# ---------------------------------------------------------------------------
p = Path('src/EllipticalCopulas/TCopula.jl')
text = r(p).replace('Distributions.params(G).Σ', 'only(Distributions.params(G))')
w(p, text)

p = Path('src/ExtremeValueCopula.jl')
text = r(p)
text = text.replace('only(values(Distributions.params(_fit(CT, U, Val{start}(); weights))))',
                    'only(Distributions.params(_fit(CT, U, Val{start}(); weights)))')
text = text.replace('only(values(Distributions.params(_fit(CT, U, Val(:iupper)))))',
                    'only(Distributions.params(_fit(CT, U, Val(:iupper))))')
w(p, text)

# Structural/empirical copulas follow the Distributions tuple convention too.
for path, old, new in (
    ('src/MiscellaneousCopulas/BernsteinCopula.jl',
     'Distributions.params(C::BernsteinCopula) = (m=C.m, weights=C.weights)',
     'Distributions.params(C::BernsteinCopula) = (C.m, C.weights)'),
    ('src/MiscellaneousCopulas/CheckerboardCopula.jl',
     'Distributions.params(C::CheckerboardCopula) = (m=C.m, boxes=C.boxes)',
     'Distributions.params(C::CheckerboardCopula) = (C.m, C.boxes)'),
):
    p = Path(path); text = r(p).replace(old, new); w(p, text)

# Nested internals use generator fields + Paramorph names, not params(generator).
p = Path('src/NestedArchimedeanCopula.jl')
text = r(p)
text = text.replace('''_gen_param_eltype(G::Generator) =\n    mapreduce(typeof, promote_type, values(Distributions.params(G)); init = Bool)''',
                    '_gen_param_eltype(G::Generator) = _parameter_eltype(G)')
text = text.replace('(parent=Distributions.params(parent), child=Distributions.params(child), leaves=d)',
                    '(parent=parent, child=child, leaves=d)')
text = text.replace('Reconstruct via `_gentype(G)(values(nt)...)`: the Generator\n# type-call (Generator.jl) splats positional args in field order, which equals the\n# order of `Distributions.params`.',
                    'Reconstruct through the generator type-call using values ordered by its\n# `Paramorph.param_space`.')
start = text.index('function _nested_coef(C::NestedArchimedeanCopula, tag::String = "G")')
end = text.index('\nend\n', start) + len('\nend\n')
newcoef = '''function _nested_generator_coef(G::Generator, dloc::Int, tag::String)\n    p = _generator_space(G, dloc)\n    names = String[]\n    values = Float64[]\n    for name in Paramorph.names(p)\n        value = getproperty(G, name)\n        value isa Number || continue\n        push!(names, "$(tag).$(name)")\n        push!(values, float(value))\n    end\n    return names, values\nend\n\nfunction _nested_coef(C::NestedArchimedeanCopula, tag::String = "G")\n    names, values = _nested_generator_coef(C.G, _local_arity(C), tag)\n    for (i, child) in enumerate(C.children)\n        if child isa Tuple\n            copula, ds = child\n            child_names, child_values = _nested_generator_coef(\n                copula.G, max(length(ds), 2), "$(tag)[$(i)]")\n            append!(names, child_names)\n            append!(values, child_values)\n        else\n            child_names, child_values = _nested_coef(child, "$(tag)[$(i)]")\n            append!(names, child_names)\n            append!(values, child_values)\n        end\n    end\n    return names, values\nend\n'''
text = text[:start] + newcoef + text[end:]
w(p, text)

# Fitting header: defining Paramorph geometry is the generic opt-in.
p = Path('src/Fitting.jl')
text = r(p).replace('families opt into the generic routines by defining `Distributions.params`',
                    'families opt into the generic routines by defining `Paramorph.param_space`')
w(p, text)

# ---------------------------------------------------------------------------
# Documentation: Generator/Tail are mathematical components, not Distributions.
# ---------------------------------------------------------------------------
for path in (Path('docs/src/api/public.md'), Path('docs/src/bestiary/archimedean.md')):
    text = r(path)
    text = text.replace(', `Distributions.params`', '')
    text = text.replace('and `Distributions.params`', '')
    text = text.replace('these three mathematical operations', 'these two mathematical operations')
    w(path, text)

p = Path('docs/src/dev/developer_guide.md')
text = r(p)
# Remove one-line generator/tail params examples and replace API-table rows.
text = re.sub(r'(?m)^Distributions\.params\((?:G|tail)::[^\n]*\) = .*\n', '', text)
text = text.replace('| `Distributions.params(G)`           | Return parameters as a `NamedTuple`                                | ✅ Public   |\n',
                    '| `Paramorph.param_space(typeof(G), d)` | Describe fitted parameter names and geometry when fitting is desired | ✅ Public   |\n')
text = text.replace('| `Distributions.params(C)` | Return parameters as a `NamedTuple`                    | ✅              |',
                    '| `Distributions.params(C)` | Return constructor parameters as a `Tuple`              | ✅              |')
text = text.replace('Distributions.params(C::MyEllipticalCopula) = (Σ = C.Σ,)',
                    'Paramorph.param_space(::Type{<:MyEllipticalCopula}, d) = Paramorph.Correlation(:Σ, d)')
text = text.replace('Distributions.params(C::MardiaCopula) = (; θ = C.θ,)',
                    'Paramorph.param_space(::Type{<:MardiaCopula}, d) = Paramorph.Id(:θ)')
w(p, text)

# ---------------------------------------------------------------------------
# Tests: no params methods on Generator/Tail and no named access on tuple params.
# ---------------------------------------------------------------------------
for path in Path('test').rglob('*.jl'):
    text = r(path)
    # Remove test-only Generator/Tail params definitions (simple one-liners).
    text = re.sub(r'(?m)^Distributions\.params\([^\n]*::[^\n]*(?:Generator|Tail)[^\n]*\) = .*\n', '', text)
    # Test-only Copula methods remain legitimate distributions, but must return tuples.
    text = re.sub(r'Distributions\.params\((C::[^\n]*Copula[^\n]*)\) = \(; θ=([^\n]*)\)',
                  r'Distributions.params(\1) = (\2,)', text)
    text = re.sub(r'Distributions\.params\((C::[^\n]*Copula[^\n]*)\) = \(; θ = ([^\n]*)\)',
                  r'Distributions.params(\1) = (\2,)', text)
    text = text.replace('Distributions.params(fitted_copula).θ', 'only(Distributions.params(fitted_copula))')
    text = text.replace('Distributions.params(Cx).θ', 'only(Distributions.params(Cx))')
    text = text.replace('Distributions.params(tail).ν == ν', 'tail.ν == ν')
    text = text.replace('Distributions.params(tail).R ≈ R', 'something(tail.R) ≈ R')
    w(path, text)

# Explicitly fix common multi-parameter tuple assertions in EM tests.
p = Path('test/extensions/expectation_maximization.jl')
text = r(p)
text = text.replace('bb1_params.θ', 'bb1_params[1]').replace('bb1_params.δ', 'bb1_params[2]')
w(p, text)

# Sklar API now expects a tuple, and Generator/Tail dof is no longer a params contract.
p = Path('test/api/sklar.jl')
text = r(p)
text = text.replace('@test params(D) isa NamedTuple', '@test params(D) isa Tuple')
text = re.sub(r'(?m)^\s*@test Copulas\._parameter_dof\(Copulas\.ClaytonGenerator\([^\n]*\n', '', text)
text = re.sub(r'(?m)^\s*@test Copulas\._parameter_dof\(Copulas\.GalambosTail\([^\n]*\n', '', text)
w(p, text)

# Source-level audit. params may be defined for Copula / SklarDist / ordinary
# Distributions only; Generator/Tail dispatch is forbidden.
proc = subprocess.run(
    ['git', 'grep', '-nE', r'Distributions\.params\([^\n]*::[^\n]*(Generator|Tail)', '--', 'src', 'test', 'docs'],
    text=True, capture_output=True,
)
if proc.stdout.strip():
    raise RuntimeError('Generator/Tail params definitions remain:\n' + proc.stdout)

# Named property access on params(C) should also be gone from package sources.
proc = subprocess.run(
    ['git', 'grep', '-nE', r'(Distributions\.)?params\([^\)]*\)\.[A-Za-zΑ-ωθρνΣΓ]', '--', 'src', 'ext', 'test'],
    text=True, capture_output=True,
)
if proc.stdout.strip():
    raise RuntimeError('named params access remains:\n' + proc.stdout)

subprocess.run(['git', 'diff', '--check'], check=True)
subprocess.run(['git', 'grep', '-n', 'Distributions.params', '--', 'src', 'ext', 'test', 'docs'], check=False)
