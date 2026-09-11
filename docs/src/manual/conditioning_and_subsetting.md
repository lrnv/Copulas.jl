```@meta
CurrentModule = Copulas
```

# Conditioning and Subsetting

## Conditioning

This page introduces conditional distributions under a copula model and shows how to construct them programmatically using `condition`. The same interface works either on the uniform scale (copula only) or on the original scale (via `SklarDist`).

### Overview

Take a $d$-variate copula $C$ and partition the coordinates into conditioned
indices $J$ and remaining indices $I$. In the regular case, where the required
derivatives exist and the conditioning marginal has a finite, positive density,
the conditional CDF on the uniform scale is given almost everywhere by

```math
H_{I\mid J}(\mathbf u_I\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u_I, \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1_I, \mathbf u_J)}{\partial \mathbf u_J},
```

which defines a distribution on $[0,1]^{\lvert I\rvert}$ under these assumptions.
This derivative ratio is not a universal construction at singular points or
where the denominator vanishes. Supported singular models may instead use
specialized conditional distributions, including atoms. A conditional law is
determined only almost everywhere with respect to the conditioning marginal;
values outside that set require a choice of version.

Each conditional marginal can be expressed as a “distortion” $H_{i|J}(· | u_J)$,
a distribution on $[0,1]$ that need not be uniform. Under the same regularity
assumptions, its CDF is

```math
H_{i\mid J}(u\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u^{(i)}(u_I), \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1, \mathbf u_J)}{\partial \mathbf u_J},
```

where $\mathbf u^{(i)}(u_I)$ has coordinate `u` at index `i` and `1` elsewhere in `I`.

For continuous, invertible conditioning margins of
`X = SklarDist(C, (X_1,…,X_D))`, conditioning on $x_J$ can be transferred to
$u_J = (F_j(x_j))_{j\in J}$ and expressed on the remaining marginal scales:

```math
F_{X_i\mid X_J}(x\mid \mathbf x_J) = H_{i\mid J}\big(F_i(x)\mid \mathbf u_J\big).
```

For discrete conditioning margins, observing $x_j$ corresponds to an interval
of latent uniforms, not merely the endpoint $F_j(x_j)$; the formula above does
not establish correct conditioning for that case.

A copula of the conditional vector is denoted $C_{I|J}(·|u_J)$; it need not be
unique when conditional margins have atoms. The public entry point is `condition`:

- `condition(C::Copula, js, u_js)` returns the conditional distribution of the remaining `U_I` on the original copula coordinate scale for `I = setdiff(1:D, js)`. Its conditional margins need not be uniform. If `length(I) == 1`, the result is a univariate distribution supported on $[0,1]`; otherwise it is a multivariate distribution implementing the usual `Distributions.jl` interface.
- `condition(X::SklarDist, js, x_js)` returns the conditional distribution on the original scale by pushing forward each distortion through the corresponding marginal.
- Known parametric families may use specialized representations, but their concrete types are implementation details and do not change this contract.

!!! tip "Missing fast-paths?"
    If you find a conditional that should admit a faster closed-form or semi-analytic path but currently falls back to the generic construction, please open an issue, we’ll happily implement it :)

### Examples

Let us visualize a given univariate distortion: 

```@example cond1
using Copulas, Distributions, Plots, StatsBase
C = ClaytonCopula(2, 1.5)
D = condition(C, 2, 0.3)  # distortion for U₁ | U₂ = 0.3
ts = range(0.0, 1.0; length=401)
plt = plot(ts, cdf.(Ref(D), ts);
          xlabel="u", ylabel="H_{1|2}(u | 0.3)",
          title="Conditional CDF on the uniform scale",
          legend=false)
plt
```

Confirm the result by overlaying the empirical cdf of a sample: 

```@example cond1
N = 2000
αs = rand(N)
us = Distributions.quantile.(Ref(D), αs)
ECDF = ecdf(us)
plot!(ts, ECDF.(ts); seriestype=:steppost, label="empirical", alpha=0.6, color=:black)
plot!(ts, cdf.(Ref(D), ts); label="analytic", color=:blue)
plt
```

The same thing can be done on marginal scales using `SklarDist`: 

```@example cond1
C = ClaytonCopula(2, 1.5)
X = SklarDist(C, (Normal(), Normal()))
X1_given_X2 = condition(X, 2, 0.0) # distribution of X₁ | X₂ = 0.0
cdf(X1_given_X2, 1.0), quantile(X1_given_X2, 0.95)
```

```@example cond1
xs = rand(X1_given_X2, 2000)
Fx = ecdf(xs)
xs_grid = range(quantile(X1_given_X2, 0.001), quantile(X1_given_X2, 0.999); length=401)
plot(xs_grid, Distributions.cdf.(Ref(X1_given_X2), xs_grid);
  xlabel="x", ylabel="F_{X₁|X₂}(x|0)", title="Original-scale conditional CDF", label="analytic")
plot!(xs_grid, Fx.(xs_grid); seriestype=:steppost, label="empirical", alpha=0.6, color=:black)
```

When conditioning on less than $D-1$ dimensions, we obtain a multivariate object, usually a `SklarDist`: 

```@example cond1
H = condition(ClaytonCopula(4, 4.2), (2, 3), (0.25, 0.8))
```

```@example cond1
plot(H)
```

### Relation to the conditional copula

The conditional copula $C_{I|J}(·|u_J)$ is the copula of the conditional distribution $H_{I|J}(·|u_J)$. For a multivariate result, the copula and margins are available through the public `params` interface:

```@example cond1
params(H).copula
```

```@example cond1
params(H).margins
```


See the canonical Public API entry for [`condition`](@ref).

### See also

- [`condition`](@ref) — reference documentation with all calling syntaxes
- [`SklarDist`](@ref) — compound distributions via Sklar’s theorem
- [`rosenblatt`](@ref) — sequential transforms (related but different)


## Subsetting

Subsetting extracts the dependence structure among a subset of coordinates. Given a copula `C` of dimension `d` and an index tuple `dims::NTuple{p,Int}`, the function `subsetdims` returns a copula on those `p` dimensions that preserves the original dependence restricted to `dims`.

There are two entry points:

- `subsetdims(C::Copula, dims)` returns a `Copula{p}` (or `Uniform()` when `p == 1`).
- `subsetdims(X::SklarDist, dims)` returns the corresponding joint distribution
  with the selected copula coordinates and margins, in the requested order.

The concrete representation is family-dependent. Some families return a natural reduced-parameter form, while the generic path uses an internal delegating representation. Both implement the same public copula interface:

```@example subset1
using Copulas, Distributions
C = GaussianCopula([1.0 0.6 0.2; 0.6 1.0 0.3; 0.2 0.3 1.0])
S = subsetdims(C, (1,3))    # 2D copula on coordinates 1 and 3
length(S), cdf(S, [0.5, 0.5])
```

```@example subset1
X = SklarDist(C, (Normal(), Normal(1,2), LogNormal()))
X13 = subsetdims(X, (1,3))  # keeps marginals (Normal(), LogNormal()) and reduces the copula
length(params(X13).copula), length(params(X13).margins)
```

The exact result type is not part of the contract. Specialized forms may provide better performance or clearer display, while every result remains usable through the same copula API.

Subsetting and conditioning commute in the obvious way: conditioning on coordinates `J` and then extracting a subset of the remaining coordinates is equivalent to subsetting the base copula first and then conditioning on the corresponding indices.

### Examples

```@example subset1
# Archimedean example
C = ClaytonCopula(3, 2.0)
S = subsetdims(C, (1,2))        # still a ClaytonCopula with the same parameter
rand(S, 3)                      # sample 3 points
cdf(S, [0.7, 0.9])
```

```@example subset1
# Survival example with flips remapped
base = GaussianCopula([1.0 0.7 0.2; 0.7 1.0 0.1; 0.2 0.1 1.0])
S = SurvivalCopula(base, (2,))
S13 = subsetdims(S, (1,3))      # flip on 2 drops; no flips remain
length(S13), cdf(S13, [0.5, 0.5])
```

See the canonical Public API entry for [`subsetdims`](@ref).

## Non-copula random vectors

The operations introduced on this page are not limited to copula-based models.

!!! tip "Extending the interface beyond copula models"
    Loading `PartitionedDistributions.jl` activates an extension that makes
    `condition`, `subsetdims`, `rosenblatt`, and `inverse_rosenblatt` available
    for compatible vector-valued distributions that are not copulas or
    `SklarDist` models. See the
    [complete interoperability example](../examples/partitioned_distributions.md).

For example, the extension supplies the sequential transforms of a multivariate
normal distribution through its marginal and conditional distributions:

```@example noncopula
using Copulas, Distributions, PartitionedDistributions

D = MvNormal(
    [0.2, -0.3, 0.7],
    [
        1.0  0.3   0.1
        0.3  1.2   0.25
        0.1  0.25  0.8
    ],
)
x = [0.1, -0.4, 1.1]

u = rosenblatt(D, x)
x_reconstructed = inverse_rosenblatt(D, u)
(u=u, reconstruction_error=maximum(abs, x_reconstructed .- x))
```

More generally, the forward transform requires `cdf` on each successive
conditional law, while the inverse also requires `quantile`.

The usual caveat still applies: the deterministic forward and inverse
transforms are mutual inverses only when the successive conditional CDFs are
continuous and invertible on their supports.

## Rosenblatt transformations

### Definition and usefulness

::: definition Rosenblatt transformation

The Rosenblatt transformation evaluates successive conditional CDFs of a random
vector ``X``. With atomless successive conditional distributions, it transforms
``X`` into independent uniforms.

More formally, consider the map ``R_X(x)`` defined as follows:

```math
R_X(x_1, ..., x_d) = (r_1 = F_{X_1}(x_1), r_2 = F_{X_2 | X_1}(x_2 | x_1), ..., r_{d} = F_{X_d | X_1, ..., X_{d-1}}(x_d | x_1, ..., x_{d-1}))
```

:::

References:
* [rosenblatt1952](@cite) Rosenblatt, M. (1952). Remarks on a multivariate transformation. Annals of Mathematical Statistics, 23(3), 470-472.
* [joe2014](@cite) Joe, H. (2014). Dependence Modeling with Copulas. CRC Press. (Section 2.10)
* [mcneil2009](@cite) McNeil, A. J., & Nešlehová, J. (2009). Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions.

In certain circumstances, in particular for Archimedean copulas, this map simplifies to tractable expressions. It has a few nice properties:

* ``R_X(X) \sim \texttt{Uniform(Unit Hypercube)}``
* The forward and inverse transforms are inverses almost surely when the
  successive conditional CDFs are continuous and invertible on their supports.

The uniformity statement also requires atomless successive conditional laws.
For atomic or singular models, the deterministic CDF transform need not produce
independent uniforms and need not be invertible. Generalized conditional quantiles
can still generate samples from independent uniforms when those conditional laws
are implemented; this does not require a bijective forward transform. No additional
randomization within CDF jumps is implicit in `rosenblatt`.

These two properties are leveraged in some cases to construct the inverse Rosenblatt transformations, which map random noise to proper samples from the copula. In some cases, this is the best sampling algorithm available. 

For a random vector represented by a `SklarDist` or `Copula`, the public
`rosenblatt(X, x)` and `inverse_rosenblatt(X, x)` operations provide the
forward and inverse transforms.

See the canonical Public API entries for [`rosenblatt`](@ref), [`inverse_rosenblatt`](@ref).

The transforms use the same conditional laws as [`condition`](@ref), so their
availability and numerical limitations follow those of the underlying family.

### Sanity check plot

You can validate that the Rosenblatt transform maps samples to independent uniforms by checking the marginal ECDFs against the 45° line.

```@example rosen1
using Copulas, Plots, StatsBase
# pick a nontrivial copula
C = ClaytonCopula(3, 1.5)

# draw samples and apply Rosenblatt transform coordinate-wise
U = rand(C, 3000)                 # size (3, N)
S = reduce(hcat, (rosenblatt(C, U[:, i]) for i in 1:size(U, 2)))  # size (3, N)

ts = range(0.0, 1.0; length=401)
layout = @layout [a b c]
plt = plot(layout=layout, size=(900, 280), legend=false)
for k in 1:3
  Ek = ecdf(S[k, :])
  plot!(plt[k], ts, Ek.(ts); seriestype=:steppost, color=:black,
      title="ECDF of $(k)", xlabel="u", ylabel="ECDF")
  plot!(plt[k], ts, ts; color=:blue, alpha=0.7)
end
plt
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
