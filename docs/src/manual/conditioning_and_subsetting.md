```@meta
CurrentModule = Copulas
```

# Conditioning and Subsetting

## Conditioning

This page introduces conditional distributions under a copula model and shows how to
construct them programmatically using `condition`.

The same interface works either on the uniform scale (copula only) or on the original
scale (via `SklarDist`).

### Overview

Take a D-variate copula $C$ and consider a split of the indices $1,..,d$ into $J = 1,...,p$ and $I = p+1,..,d$ wihtout loss of generality (reorder the marignals if necessary). The conditional joint distribution of the I's given the J's is given, on the uniform scale, by the function

```math
H_{I\mid J}(\mathbf u_I\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u_I, \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1_I, \mathbf u_J)}{\partial \mathbf u_J},
```

which is a proper distribution on $[0,1]^{\lvert I\rvert}$. Each conditional marginal (uniform scale) can moreover be expressed as a “distortion” $H_{i|J}(· | u_J)$, which is the distribution function of a random variable with support $[0,1]$ (but non-uniform):

```math
H_{i\mid J}(u\mid\mathbf u_J)
:= \frac{\partial^p C(\mathbf u^{(i)}(u_I), \mathbf u_J)}{\partial \mathbf u_J}
\Big/ \frac{\partial^p C(\mathbf 1, \mathbf u_J)}{\partial \mathbf u_J},
```

where $\mathbf u^{(i)}(u_I)$ has coordinate `u` at index `i` and `1` elsewhere in `I`.

On the original scale for a compound distribution `X = SklarDist(C, (X_1,…,X_D))`, conditioning on values $x_J$ is obtained by mapping to uniforms $u_J = (F_j(x_j))_{j\in J}$ then pushing forward each distortion through the corresponding marginal:

```math
F_{X_i\mid X_J}(x\mid \mathbf x_J) = H_{i\mid J}\big(F_i(x)\mid \mathbf u_J\big).
```

The copula of the conditional vector $U_I | U_J = u_J$ is a genuine copula denoted $C_{I|J}(·|u_J)$, which is the copula of $H_{I|J}$. In our implementation, this is materialized by the a `ConditionalCopula(C, J, u_J)` and used internally by `condition`. The `condition` function can be used as follows: 

- `condition(C::Copula, js, u_js)` returns the conditional distribution on the uniform scale for `I = setdiff(1:D, js)`. If `length(I) == 1`, the result is a univariate distribution supported on $[0,1]$, subclass of `Distortion`, and otherwise it is a `SklarDist(::ConditionalCopula, NTuple{d,<:Distortion})`.
- `condition(X::SklarDist, js, x_js)` returns the conditional distribution on the original scale by pushing forward each distortion through the corresponding marginal.
- For known parametric families, there are fast paths implemented mostly as subclass to `Distortions` or `ConditionalCopula`, but this should be completely transparent to the user. 

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

The conditional copula $C_{I|J}(·|u_J)$ is the copula of the conditional distribution $H_{I|J}(·|u_J)$. In the implementation it is represented by `ConditionalCopula(C, js, u_js)` and is used as the copula of the conditional joint when `|I| > 1`. When `condition` returns a `SklarDist` (i.e., when `|I| > 1`), you can access this copula directly via the `.C` field of the returned object:

```@example cond1
H.C # the copula
```

```@example cond1
H.m # the marginals
```


### Implementation

```@docs; canonical=false
condition
Distortion
DistortionFromCop
DistortedDist
ConditionalCopula
```


### See also

- [`condition`](@ref) — reference documentation with all calling syntaxes
- [`SklarDist`](@ref) — compound distributions via Sklar’s theorem
- [`rosenblatt`](@ref) — sequential transforms (related but different)


## Subsetting

Subsetting extracts the dependence structure among a subset of coordinates. Given a copula `C` of dimension `d` and an index tuple `dims::NTuple{p,Int}`, the function `subsetdims` returns a copula on those `p` dimensions that preserves the original dependence restricted to `dims`.

There are two entry points:

- `subsetdims(C::Copula, dims)` returns a `Copula{p}` (or `Uniform()` when `p == 1`).
- `subsetdims(X::SklarDist, dims)` returns a `SklarDist` with copula `subsetdims(C, dims)`
  and marginals `(m[i] for i in dims)`.

Internally, we materialize subsetting with a small wrapper type `SubsetCopula{p}(C, dims)` which delegates `cdf`, `pdf`, and sampling to the base copula by saturating non-selected coordinates at 1. For many families we provide specialized constructors that return the natural reduced-parameter form instead of a wrapper (e.g., elliptical copulas return the appropriate submatrix, Archimedean keeps the same generator with reduced dimension, etc.). It can be used as follows: 

```@example subset1
using Copulas, Distributions
C = GaussianCopula([1.0 0.6 0.2; 0.6 1.0 0.3; 0.2 0.3 1.0])
S = subsetdims(C, (1,3))    # 2D copula on coordinates 1 and 3
length(S), typeof(S)
```

```@example subset1
X = SklarDist(C, (Normal(), Normal(1,2), LogNormal()))
X13 = subsetdims(X, (1,3))  # keeps marginals (Normal(), LogNormal()) and reduces the copula
length(X13.C), length(X13.m)
```

The resulting object depends on the copula familly, since some fast paths are given. If no specialization exists, a `SubsetCopula` wrapper is returned. It’s fully usable and equivalent from an API perspective; specialized forms simply yield better performance and clearer display.

Subsetting and conditioning commute in the obvious way: conditioning on coordinates `J` and then extracting a subset of the remaining coordinates is equivalent to subsetting the base copula first and then conditioning on the corresponding indices. In code, if `S = subsetdims(C, dims)`, conditioning on indices `js` within `S` is implemented by mapping `js` to indices in the base copula and delegating to `ConditionalCopula(C, ·, ·)`; the resulting conditional copula of `S` is either the base conditional copula (when all remaining coordinates are kept) or a further `SubsetCopula` of it.

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
typeof(S13), S13 isa SurvivalCopula
```

### Implementation

```@docs; canonical=false
Copulas.subsetdims
Copulas.SubsetCopula
```

## Rosenblatt transformations

### Definition and usefulness

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

### Implementation

As soon as the random vector ``X`` is represented by an object `X` that subtypes `SklarDist` or `Copula`, you have access to the `rosenblatt(X, x)` and `inverse_rosenblatt(X, x)` operators, which both have a straightforward interpretation from their names. 

```@docs; canonical=false
rosenblatt
inverse_rosenblatt
```

Once again, since the rosenblatt transform leverages the conditioning mechanisme, some fast-paths might be missing in the implementation.

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

```@bibliography
Pages = [@__FILE__]
Canonical = false
```

