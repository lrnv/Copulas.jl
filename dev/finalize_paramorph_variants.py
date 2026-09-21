from pathlib import Path

root = Path("src")

# Extreme-value natural parameters consume the semantic bridge rather than
# Paramorph directly.
p = root / "ExtremeValueCopula.jl"
text = p.read_text()
old = '''function Distributions.params(C::ExtremeValueCopula)
    if Paramorph.is_paramorph_type(typeof(C.tail))
        return Tuple(values(Paramorph.parameter_values(C.tail)))
    end
    C.tail isa DiscreteSpectralCapableTail &&
        return (copy(_spectral_tail(C.tail).B),)
    return map(fieldnames(typeof(C.tail))) do name
        value = getfield(C.tail, name)
        value isa AbstractArray ? copy(value) : value
    end
end
'''
new = '''function Distributions.params(C::ExtremeValueCopula)
    declared = _declared_parameter_values(C.tail)
    if declared !== nothing
        return Tuple(map(values(declared)) do value
            value isa AbstractArray ? copy(value) : value
        end)
    end
    C.tail isa DiscreteSpectralCapableTail &&
        return (copy(_spectral_tail(C.tail).B),)
    return map(fieldnames(typeof(C.tail))) do name
        value = getfield(C.tail, name)
        value isa AbstractArray ? copy(value) : value
    end
end
'''
assert old in text, "ExtremeValue params block changed"
p.write_text(text.replace(old, new, 1))

# Huesler-Reiss: keep the domain object unchanged, but express each optimizer
# geometry through a private @paramorph declaration.
p = root / "Tail/HuslerReissTail.jl"
text = p.read_text()
needle = '''end
@inline _hr_is_independent(tail::HuslerReissTail{<:Real}) = iszero(something(tail.θ))
'''
insert = '''end

Paramorph.@paramorph T struct _HuslerReissScalarGeometry{T<:Real}
    θ::T ~ Paramorph.nonnegative()
end

Paramorph.@paramorph T struct _HuslerReissMatrixGeometry{T<:Real}
    d::Int
    Γ::Matrix{T} ~ Paramorph.variogram_matrix(d)
end

@inline _hr_is_independent(tail::HuslerReissTail{<:Real}) = iszero(something(tail.θ))
'''
assert needle in text, "HuslerReiss insertion point changed"
text = text.replace(needle, insert, 1)
old = '''Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:AbstractMatrix}}) where {D} =
    (copy(something(C.tail.Γ)),)
'''
new = '''Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:Real}}) where {D} =
    (something(C.tail.θ),)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{<:AbstractMatrix}}) where {D} =
    (copy(something(C.tail.Γ)),)
'''
assert old in text
text = text.replace(old, new, 1)
p.write_text(text)

# Extremal-t: same split, with dimension supplied through context for the
# exchangeable correlation bound and through an auxiliary field for matrices.
p = root / "Tail/tEVTail.jl"
text = p.read_text()
needle = '''end
@inline limit_kind(tail::tEVTail{<:Any,<:Real}, ::Val) =
'''
insert = '''end

Paramorph.@paramorph T struct _tEVScalarGeometry{T<:Real}
    ν::T ~ Paramorph.TransformVariables.asℝ₊
    ρ::T ~ Paramorph.bounded_interval(
        -inv(get(context, :dimension, 2) - 1), one(T); left_closed=false,
    )
end

Paramorph.@paramorph T struct _tEVMatrixGeometry{T<:Real}
    d::Int
    ν::T ~ Paramorph.TransformVariables.asℝ₊
    R::Matrix{T} ~ Paramorph.correlation_matrix(d)
end

@inline limit_kind(tail::tEVTail{<:Any,<:Real}, ::Val) =
'''
assert needle in text, "tEV insertion point changed"
text = text.replace(needle, insert, 1)
needle = '''_is_valid_in_dim(tail::tEVTail{<:Any,<:AbstractMatrix}, d::Int) =
    d == size(something(tail.R), 1)

    
_tail_constructor_parameter_names'''
replacement = '''_is_valid_in_dim(tail::tEVTail{<:Any,<:AbstractMatrix}, d::Int) =
    d == size(something(tail.R), 1)

Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{<:Any,<:Real}}) where {D} =
    (C.tail.ν, something(C.tail.ρ))
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{<:Any,<:AbstractMatrix}}) where {D} =
    (C.tail.ν, copy(something(C.tail.R)))

_tail_constructor_parameter_names'''
assert needle in text, "tEV parameter insertion point changed"
text = text.replace(needle, replacement, 1)
p.write_text(text)

# Bridge: introduce a generic tail prototype hook and make HR/tEV companions
# ordinary @paramorph clients. Remove their hand-written schemas completely.
p = root / "ParamorphFitting.jl"
text = p.read_text()
old = '''function _parameter_prototype(CT::Type{<:ExtremeValueCopula}, ::Val{d}) where {d}
    TT = _concrete_paramorph_type(tailof(CT))
    auxiliary = (; d=d)
    tail = _component_prototype(TT, (; dimension=d); auxiliary)
    return ExtremeValueCopula{d}(tail)
end
'''
new = '''function _tail_prototype(TT::Type, ::Val{d}) where {d}
    concrete = _concrete_paramorph_type(TT)
    return _component_prototype(
        concrete, (; dimension=d); auxiliary=(; d=d),
    )
end

function _tail_prototype(::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    geometry = _component_prototype(_HuslerReissScalarGeometry{Float64})
    return HuslerReissTail(geometry.θ)
end

function _tail_prototype(::Type{<:tEVTail}, ::Val{d}) where {d}
    geometry = _component_prototype(
        _tEVScalarGeometry{Float64}, (; dimension=d),
    )
    return tEVTail(geometry.ν, geometry.ρ)
end

function _parameter_prototype(CT::Type{<:ExtremeValueCopula}, vd::Val{d}) where {d}
    return ExtremeValueCopula{d}(_tail_prototype(tailof(CT), vd))
end
'''
assert old in text, "EV prototype bridge changed"
text = text.replace(old, new, 1)
old = '''    tail = _component_prototype(_concrete_paramorph_type(TT), (; dimension=d); auxiliary=(; d=d))
'''
new = '''    tail = _tail_prototype(TT, vd)
'''
assert old in text, "Archimax tail prototype line changed"
text = text.replace(old, new, 1)

start = text.index('''# Two legacy leaf representations cannot yet be written as a single @paramorph''')
# The legacy block runs to EOF in the current bridge.
legacy = text[start:]
replacement = '''# Representation variants use private @paramorph geometry objects. The domain
# tails keep their constructor/storage invariants while every optimizer
# constraint remains declarative.
_hr_geometry(tail::HuslerReissTail{<:Real}) =
    _HuslerReissScalarGeometry(something(tail.θ))
function _hr_geometry(tail::HuslerReissTail{<:AbstractMatrix})
    Γ = something(tail.Γ)
    return _HuslerReissMatrixGeometry(size(Γ, 1), Matrix(Γ))
end
_hr_tail(g::_HuslerReissScalarGeometry) = HuslerReissTail(g.θ)
_hr_tail(g::_HuslerReissMatrixGeometry) = HuslerReissTail(g.Γ)

_parameter_dimension(tail::HuslerReissTail, ::Val) =
    Paramorph.intrinsic_dimension(_hr_geometry(tail))
_parameter_coordinates(tail::HuslerReissTail, ::Val) =
    Paramorph.unconstrain(_hr_geometry(tail))
_from_parameter_coordinates(tail::HuslerReissTail, α, ::Val) =
    _hr_tail(Paramorph.constraint(_hr_geometry(tail), α))

_tev_geometry(tail::tEVTail{<:Any,<:Real}, d) =
    _tEVScalarGeometry(tail.ν, something(tail.ρ))
function _tev_geometry(tail::tEVTail{<:Any,<:AbstractMatrix}, d)
    R = something(tail.R)
    return _tEVMatrixGeometry(size(R, 1), tail.ν, Matrix(R))
end
_tev_tail(g::_tEVScalarGeometry) = tEVTail(g.ν, g.ρ)
_tev_tail(g::_tEVMatrixGeometry) = tEVTail(g.ν, g.R)

_parameter_dimension(tail::tEVTail, ::Val{d}) where {d} =
    Paramorph.intrinsic_dimension(_tev_geometry(tail, d); context=(; dimension=d))
_parameter_coordinates(tail::tEVTail, ::Val{d}) where {d} =
    Paramorph.unconstrain(_tev_geometry(tail, d); context=(; dimension=d))
_from_parameter_coordinates(tail::tEVTail, α, ::Val{d}) where {d} =
    _tev_tail(Paramorph.constraint(_tev_geometry(tail, d), α; context=(; dimension=d)))
'''
text = text[:start] + replacement
p.write_text(text)

# Make the integration workflow a real oracle: strict precompilation and full
# package tests, and fail if runtime Paramorph leaks outside the bridge.
p = Path('.github/workflows/paramorph-003-migration.yml')
text = p.read_text()
text = text.replace(
'''      - name: Remaining direct Paramorph coupling
        run: |
          echo '--- Paramorph method extensions ---'
          grep -R -nE '^(function )?Paramorph\\.(schema_|transformation_schema|parameter_values|reconstruct_struct|parameter_fields|is_paramorph_type)' src || true
          echo '--- outside the centralized bridge ---'
          grep -R -n 'Paramorph\\.' src/Fitting.jl src/Inference.jl src/NaturalParameters.jl src/ParameterCoefficients.jl || true
      - name: Instantiate and load Copulas
        shell: julia --project=. --color=yes {0}
        run: |
          using Pkg
          Pkg.instantiate()
          using Copulas
''',
'''      - name: Reject legacy Paramorph protocol extensions
        run: |
          if grep -R -nE '^(function )?Paramorph\\.(schema_|transformation_schema|parameter_values|reconstruct_struct|parameter_fields_override|schema_override)' src; then
            echo 'Legacy Paramorph protocol extensions remain' >&2
            exit 1
          fi
          if grep -n 'Paramorph\\.' src/Fitting.jl src/Inference.jl src/NaturalParameters.jl src/ParameterCoefficients.jl; then
            echo 'Runtime Paramorph leaked outside the centralized bridge' >&2
            exit 1
          fi
      - name: Precompile and test Copulas
        shell: julia --project=. --color=yes {0}
        run: |
          using Pkg
          Pkg.instantiate()
          Pkg.precompile(; strict=true)
          Pkg.test()
''')
p.write_text(text)

print("variant geometries and strict integration test prepared")
