# [Interoperability with PartitionedDistributions.jl](@id partitioned_distributions_example)

[PartitionedDistributions.jl](https://github.com/sethaxen/PartitionedDistributions.jl) provides a generic interface for extracting marginal and conditional distributions from multivariate distributions.

When both `Copulas.jl` and `PartitionedDistributions.jl` are loaded, an extension connects their APIs in both directions:

* `PartitionedDistributions.marginal` and `PartitionedDistributions.conditional` can be used with `Copula` and `SklarDist` objects;
* `Copulas.subsetdims`, `Copulas.condition`, `Copulas.rosenblatt`, and
  `Copulas.inverse_rosenblatt` can be used with vector-valued distributions
  supported by `PartitionedDistributions.jl`.

This lets downstream code choose either interface without having to special-case copula-based models.

## Setup

```@example partitioned
using Copulas
using Distributions
using PartitionedDistributions
```

## Using the PartitionedDistributions.jl API on copulas

Consider a three-dimensional Gaussian copula.

```@example partitioned
C = GaussianCopula{3}(0.35)
u = [0.2, 0.4, 0.7]
nothing # hide
```

`PartitionedDistributions.marginal` specifies the coordinates to keep. For copulas, it delegates to `Copulas.subsetdims`.

```@example partitioned
C13_pd = marginal(C, [1, 3])
C13_copulas = subsetdims(C, (1, 3))

(
    partitioned = logpdf(C13_pd, u[[1, 3]]),
    copulas = logpdf(C13_copulas, u[[1, 3]]),
)
```

The order of the selected dimensions is preserved:

```@example partitioned
C31_pd = marginal(C, [3, 1])
C31_copulas = subsetdims(C, (3, 1))

(
    partitioned = logpdf(C31_pd, u[[3, 1]]),
    copulas = logpdf(C31_copulas, u[[3, 1]]),
)
```

The two packages use complementary conventions for conditioning.

`PartitionedDistributions.conditional(dist, x, keep)` specifies the coordinates that remain random, whereas `Copulas.condition(dist, observed, values)` specifies the coordinates that are observed.

For example, keeping coordinates 1 and 3 means conditioning on coordinate 2:

```@example partitioned
C13_cond_pd = conditional(C, u, [1, 3])
C13_cond_copulas = condition(C, 2, u[2])

v = [0.3, 0.6]

(
    partitioned = logpdf(C13_cond_pd, v),
    copulas = logpdf(C13_cond_copulas, v),
)
```

Likewise, keeping only the first coordinate is equivalent to conditioning on coordinates 2 and 3:

```@example partitioned
C1_cond_pd = conditional(C, u, 1)
C1_cond_copulas = condition(C, (2, 3), (u[2], u[3]))

x = 0.3

(
    partitioned = logpdf(C1_cond_pd, x),
    copulas = logpdf(C1_cond_copulas, x),
)
```

Because `PartitionedDistributions.jl` defines its generic pointwise conditional log-density interface in terms of `conditional`, it also works directly with copulas:

```@example partitioned
pointwise_conditional_logpdfs(C, u)
```

## Using the PartitionedDistributions.jl API on `SklarDist`

The same interface works on distributions built with Sklar's theorem.

```@example partitioned
S = SklarDist(
    C,
    (
        Normal(0.0, 1.0),
        LogNormal(0.1, 0.5),
        Gamma(2.0, 1.0),
    ),
)

x = [0.2, 1.1, 2.0]
nothing # hide
```

Marginalization can again be expressed with either package:

```@example partitioned
S13_pd = marginal(S, [1, 3])
S13_copulas = subsetdims(S, (1, 3))

(
    partitioned = logpdf(S13_pd, x[[1, 3]]),
    copulas = logpdf(S13_copulas, x[[1, 3]]),
)
```

And the conditioning conventions remain complementary:

```@example partitioned
S13_cond_pd = conditional(S, x, [1, 3])
S13_cond_copulas = condition(S, 2, x[2])

y = [0.1, 1.8]

(
    partitioned = logpdf(S13_cond_pd, y),
    copulas = logpdf(S13_cond_copulas, y),
)
```

## Using the Copulas.jl API on other distributions

The extension also works in the opposite direction.

For vector-valued distributions supported by `PartitionedDistributions.jl`, `subsetdims` delegates to `PartitionedDistributions.marginal`, while `condition` delegates to `PartitionedDistributions.conditional`.

For example, consider a multivariate normal distribution:

```@example partitioned
μ = [0.2, -0.3, 0.7]

Σ = [
    1.0  0.3   0.1
    0.3  1.2   0.25
    0.1  0.25  0.8
]

D = MvNormal(μ, Σ)
x = [0.1, -0.4, 1.1]
nothing # hide
```

The Copulas.jl subsetting interface can now be used directly:

```@example partitioned
D13_copulas = subsetdims(D, (1, 3))
D13_pd = marginal(D, [1, 3])

(
    copulas_mean = mean(D13_copulas),
    partitioned_mean = mean(D13_pd),
)
```

The same applies to conditioning. With the Copulas.jl convention, specifying coordinate 2 means that coordinate 2 is observed and coordinates 1 and 3 remain random:

```@example partitioned
D13_cond_copulas = condition(D, 2, x[2])
D13_cond_pd = conditional(D, x, [1, 3])

(
    copulas_mean = mean(D13_cond_copulas),
    partitioned_mean = mean(D13_cond_pd),
)
```

Conditioning on several coordinates works in the same way:

```@example partitioned
D1_cond_copulas = condition(
    D,
    (2, 3),
    (x[2], x[3]),
)

D1_cond_pd = conditional(D, x, 1)

(
    copulas_mean = mean(D1_cond_copulas),
    partitioned_mean = mean(D1_cond_pd),
)
```

The marginal and conditional interface also supplies the successive laws needed
by the Rosenblatt transform and its inverse:

```@example partitioned
u = rosenblatt(D, x)
x_reconstructed = inverse_rosenblatt(D, u)

(
    uniforms = u,
    reconstruction_error = maximum(abs, x_reconstructed .- x),
)
```

## Choosing an interface

The two interfaces describe the same marginalization and conditioning operations but use different conditioning conventions:

| Operation                  | Copulas.jl                           | PartitionedDistributions.jl |
| -------------------------- | ------------------------------------ | --------------------------- |
| Keep dimensions 1 and 3    | `subsetdims(D, (1, 3))`              | `marginal(D, [1, 3])`       |
| Observe dimension 2        | `condition(D, 2, x[2])`              | `conditional(D, x, [1, 3])` |
| Observe dimensions 2 and 3 | `condition(D, (2, 3), (x[2], x[3]))` | `conditional(D, x, 1)`      |

Neither interface is required internally by user code: loading both packages activates the interoperability extension, and users can choose whichever convention best fits their application.
