# [Compatibility and release policy](@id compatibility_policy)

`Copulas.jl` follows [semantic versioning](https://semver.org/) with the
[Julia package compatibility rules](https://pkgdocs.julialang.org/v1/compatibility/).
This page records only the package-specific decisions maintainers need when
applying those rules.

The compatibility boundary is the documented [Public API](@ref). Internal
representations and the interfaces in the
[developer guide](@ref developer_fitting) may change without deprecation.

When a public call can be translated unambiguously, keep a working deprecation
for at least one normal release cycle. A bug fix that changes observable results
must link its issue, include a regression test, and be called out in the release
notes.

## Maintainer checklist

For a compatibility-relevant PR or release:

1. Review changes to exports, `public` declarations, documented external
   methods, constructors, keywords, and result semantics.
2. Keep docstrings, the public reference, and public contract tests aligned.
3. Add a migration path for intentional public changes and list user-visible
   changes in the release notes.
4. Keep implementation details in developer documentation rather than turning
   them accidentally into public promises.

The remaining known 1.0 compatibility item is numeric-type preservation during
sampling, tracked in
[#195](https://github.com/lrnv/Copulas.jl/issues/195). Further deferred breaking
changes should receive a focused issue and the `1.0` label.
