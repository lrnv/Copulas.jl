```@meta
CurrentModule = Copulas
```

# Other Copulas

Some copulas, while necessary in certain cases and very useful, are hard to classify. We gather them here for simplicity. 

## [Independence and Fréchet-Hoeffding bounds](@id bestiary_ref)

### `IndependentCopula`
```@docs; canonical=false
IndependentCopula
```

### `MCopula`
```@docs; canonical=false
MCopula
```

### `WCopula`
```@docs; canonical=false
WCopula
```


## Transformed Copulas

### `SurvivalCopula`

```@docs; canonical=false
SurvivalCopula
```

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

```@docs; canonical=false
PlackettCopula
```

### `FGMCopula`

Farlie-Gumbel-Morgenstern (FGM) copula

```@docs; canonical=false
FGMCopula
```

### `RafteryCopula`

```@docs; canonical=false
RafteryCopula
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
