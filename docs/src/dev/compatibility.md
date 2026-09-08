# [Compatibility and release policy](@id compatibility_policy)

This policy is for maintainers. It explains how changes to the documented
public API are reviewed and released; it does not make internal implementation
details public.

## Compatibility boundary

The supported API is the surface described on the [Public API](@ref) page:

- documented names declared `export` or `public` by `Copulas`;
- documented methods that `Copulas.jl` adds to adopted interfaces such as
  `Distributions`, `StatsBase`, `StatsAPI`, `Random`, and `Base`;
- documented constructors, argument order, keywords, result shapes, and
  mathematical behaviour.

Public status alone does not promise concrete storage types, undocumented type
parameters or fields, exact floating-point results, internal helper functions,
dispatch routes, algorithms, caches, or third-party backends. The internal
interfaces in the [developer guide](@ref developer_fitting) may change without
deprecation.

## Versioning rules

`Copulas.jl` follows semantic versioning as interpreted by Julia's package
manager. Before version 1.0, a minor release may contain breaking public changes;
patch releases remain backward compatible. From version 1.0 onward, breaking
public changes require a major release.

The following changes are compatibility-sensitive when they affect documented
behaviour:

- removing or renaming a public name or supported constructor;
- changing positional argument order, keyword names, or keyword defaults;
- narrowing accepted inputs or changing documented errors for unsupported ones;
- changing result dimensions, orientation, or documented return semantics;
- changing parameter conventions or replacing a model by another concrete
  family at a constructor boundary;
- changing the statistical or mathematical meaning of an operation.

Adding a method, family, keyword, or more accurate numerical specialization is
normally compatible when existing calls retain their documented semantics and
the new dispatch does not create ambiguities. Performance, allocation behaviour,
and the selected numerical algorithm are not compatibility promises unless they
are explicitly documented.

A bug fix may intentionally change incorrect results. Such a change must carry a
regression test and release-note entry, link the corresponding issue, and state
whether users could have relied on the previous behaviour. If reasonable code
may depend on it, use the breaking-release or deprecation process instead of
silently shipping it in a patch.

## Deprecation

Prefer a working deprecation path when an old call can be translated to its
replacement without ambiguity. Document the replacement and keep the warning
for at least one normal release cycle. Removal then follows the versioning rules
above.

Immediate removal is acceptable for undocumented internals, newly introduced
unreleased code, or behaviour that cannot safely be retained. Deprecation is not
required for changing an internal extension hook, but downstream users known to
depend on it should receive a migration note when practical.

## Pull-request checklist

For every compatibility-relevant change:

1. Identify whether the affected name or behaviour is documented public API.
2. Compare exports, `public` declarations, docstrings, manuals, and adopted
   external methods; none of these sources alone is a complete inventory.
3. Preserve both documented constructor forms where applicable and avoid
   promising concrete storage representations.
4. Update public contract tests and add an independent oracle or regression when
   mathematical behaviour changes.
5. Check that new dispatch does not introduce ambiguities or bypass generic
   contracts.
6. Add a deprecation and migration note, or target an appropriate breaking
   release, when compatibility cannot be preserved.
7. Keep architectural explanations in the developer documentation rather than
   expanding the public contract accidentally.

## Release checklist

Before tagging a release:

1. Review changes to exports, `public` declarations, constructor signatures,
   keywords, and documented external-interface methods.
2. Confirm that public docstrings, the public reference, and contract tests agree.
3. List user-visible fixes, deprecations, and migrations in the release notes.
4. Verify that compatibility bounds and the Julia version in `Project.toml`
   match the CI matrix and documentation.
5. For a breaking release, enumerate every intentional incompatibility and its
   replacement; do not bundle unrelated speculative cleanup.

## Tracked compatibility work

The constructor return-type reductions tracked in issue
[#333](https://github.com/lrnv/Copulas.jl/issues/333) have been completed. The
remaining known 1.0 compatibility item is numeric-type preservation during
sampling, tracked in
[#195](https://github.com/lrnv/Copulas.jl/issues/195). New candidate breaking
changes should receive a focused issue and the `1.0` label when they are deferred
to that milestone.
