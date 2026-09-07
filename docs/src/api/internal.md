```@meta
CurrentModule = Copulas
```

# Internal API (for reference)

These are non-public implementation details. They can change without notice. Use at your own risk.

This includes the distortion protocol, generator derivative/inversion helpers,
tail derivatives and mixed partials, and internal limit representations.
Docstrings here explain the current architecture, not supported downstream
specialization contracts. Tests of these mechanisms remain necessary even though
the mechanisms themselves are not public API.

```@autodocs
Modules = [Copulas]
Public = false
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
