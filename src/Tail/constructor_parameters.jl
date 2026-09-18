# Public constructor parameter names for tails whose storage layout differs
# from their mathematical constructor API. Keeping these as ordinary methods
# makes the generic Tail constructor extensible without a central name switch.
_tail_constructor_parameter_names(::Type{<:AsymGalambosTail}, _) =
    (:α, :θ₁, :θ₂)
_tail_constructor_parameter_names(::Type{<:BC2Tail}, _) =
    (:a, :b)
_tail_constructor_parameter_names(::Type{<:MOTail}, _) =
    (:λ₁, :λ₂, :λ₃)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail}, kwkeys) =
    :Γ in kwkeys ? (:Γ,) : (:θ,)
_tail_constructor_parameter_names(::Type{<:tEVTail}, kwkeys) =
    :R in kwkeys ? (:ν, :R) : (:ν, :ρ)
