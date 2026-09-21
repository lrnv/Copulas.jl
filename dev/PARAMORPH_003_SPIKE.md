# Paramorph 0.0.3 migration spike

This branch tests the breaking Paramorph 0.0.3 DSL against the real fitting
refactor from Copulas.jl PR #554.

The migration deliberately treats Paramorph as a geometry declaration language:
family/component definitions own their geometry, while Copulas should consume
that geometry through a small internal adapter. Family-specific
`Paramorph.schema_*` method extensions are considered migration failures unless
they reveal a genuinely missing Paramorph capability.

Sentinel cases:

- dimension-dependent scalar generator constraints (Clayton/AMH/Gumbel-Barnett);
- an opaque custom vector geometry (FGM);
- runtime-field-dependent composite geometry (Tawn);
- genuinely coupled geometry (AsymMixed, plus structural cases as encountered).

The target state is zero family-specific Paramorph method extensions outside
`@paramorph` declarations. Structural Copulas wrappers may use generic adapter
methods because they enforce domain invariants that are not parameter geometry.
