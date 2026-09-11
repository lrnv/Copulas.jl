```@meta
CurrentModule = Copulas
```

# Other Copulas

Some copulas, while necessary in certain cases and very useful, are hard to classify. We gather them here for simplicity. 

## [Independence and Fréchet-Hoeffding bounds](@id bestiary_ref)

### `IndependentCopula`
See the canonical Public API entry for [`IndependentCopula`](@ref).

### `MCopula`
See the canonical Public API entry for [`MCopula`](@ref).

### `WCopula`
See the canonical Public API entry for [`WCopula`](@ref).

## Transformed Copulas

### `SurvivalCopula`

See the canonical Public API entry for [`SurvivalCopula`](@ref).

When fitting a rotated model, pass the desired flip indices explicitly because
they belong to the instance rather than its type:

```julia
using Distributions
S = SurvivalCopula(ClaytonCopula(2, 2.0), (1,))
U = rand(S, 100)
Ŝ = fit(typeof(S), U; flips=(1,))
```

## Others

### `PlackettCopula`

See the canonical Public API entry for [`PlackettCopula`](@ref).

### `FGMCopula`

Farlie-Gumbel-Morgenstern (FGM) copula

See the canonical Public API entry for [`FGMCopula`](@ref).

### `RafteryCopula`

See the canonical Public API entry for [`RafteryCopula`](@ref).

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
