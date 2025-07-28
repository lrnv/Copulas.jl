```@meta
CurrentModule = Copulas
```

# Rosenblatt transformations

## Definition and usefullness


!!! definition "Definition (Rosenblatt transformation):"
    The Rosenblatt transformation considers a random vector ``X`` to be distributed as a certain multivariate cumulative distribution function ``F_{X}(x)``, and maps it back to a uniform distribution on the unit hypercube. 

    More formally, consider the map ``R_X(x)`` defined as follows: 

    ```math
    R_X(x_1,...,x_d) = (r_1 = F_{X_1}(x_1), r_2 = F_{X_2 | X_1}(x_2 | x_1), ..., r_{d} = F_{X_d | X_1,...,X_{d-1}}(x_d|x_1,...x_{d-1}))
    ```


In certain circonstances, in paritcular for Archimedean copulas, this map simplifies to tractable expressions. it has a few nice properties: 

* ``R_X(X) \sim \texttt{Uniform(Unit Hypercube)}``
* ``R_X`` is a bijection. 

These two properties are leveraged in some cases to construct the inverse rosenblatt transformations, that maps random noise to proper samples from the copula. In some cases, this is the best sampling algorithm available. 

## Implementation

As soon as the random vector ``X`` is represented by an object `X` that subtypes `SklarDist` or `Copula`, you have access to the `rosenblatt(X, x)` and `inverse_rosenblatt(X,x)` operators, which both have a straghtforward interpretation from their names. 

```@docs
rosenblatt
inverse_rosenblatt
```

!!! note "Not all copulas available !"
    Some copulas such has archimedeans have known expressions for their rosenblatt and/or inverse rosenblatt transforms, and therefore benefit from this interface and our implementation. On the other hand, some copulas have no known closed form expressions for conditional cdfs, and therefore their rosenblatt transformation is hard to implement.
    
    If you feel like you miss methods for certain particular copulas while the theory exists and it should be possible, do not hesitate to open an issue !


```@bibliography
Pages = [@__FILE__]
Canonical = false
```