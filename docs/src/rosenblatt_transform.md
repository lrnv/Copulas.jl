```@meta
CurrentModule = Copulas
```

# Rosenblatt transformations

## Definition and usefulness

!!! definition "Definition (Rosenblatt transformation):"
    The Rosenblatt transformation considers a random vector ``X`` distributed according to a certain multivariate cumulative distribution function ``F_{X}(x)``, and maps it back to a uniform distribution on the unit hypercube.

    More formally, consider the map ``R_X(x)`` defined as follows:

    ```math
    R_X(x_1, ..., x_d) = (r_1 = F_{X_1}(x_1), r_2 = F_{X_2 | X_1}(x_2 | x_1), ..., r_{d} = F_{X_d | X_1, ..., X_{d-1}}(x_d | x_1, ..., x_{d-1}))
    ```

References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.

In certain circumstances, in particular for Archimedean copulas, this map simplifies to tractable expressions. It has a few nice properties:

* ``R_X(X) \sim \texttt{Uniform(Unit Hypercube)}``
* ``R_X`` is a bijection. 

These two properties are leveraged in some cases to construct the inverse Rosenblatt transformations, which map random noise to proper samples from the copula. In some cases, this is the best sampling algorithm available. 

## Implementation

As soon as the random vector ``X`` is represented by an object `X` that subtypes `SklarDist` or `Copula`, you have access to the `rosenblatt(X, x)` and `inverse_rosenblatt(X, x)` operators, which both have a straightforward interpretation from their names. 

```@docs
rosenblatt
inverse_rosenblatt
```

!!! note "Not all copulas available!"
    Some copulas, such as Archimedeans, have known expressions for their Rosenblatt and/or inverse Rosenblatt transforms, and therefore benefit from this interface and our implementation. On the other hand, some copulas have no known closed-form expressions for conditional CDFs, and therefore their Rosenblatt transformation is hard to implement.

    In particular, we did not implement yet a suitable default for all cases. If you feel that methods for certain particular copulas are missing while the theory exists and it should be possible, do not hesitate to open an issue ! If you feel like you have a potential generic implementation that would be suitable, please reach us too. 


```@bibliography
Pages = [@__FILE__]
Canonical = false
```