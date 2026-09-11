```@meta
CurrentModule = Copulas
```

# [Hypothesis testing](@id hypothesis_testing)

Copula models make qualitative claims about dependence: coordinates may be
independent or exchangeable, a dependence structure may be radially symmetric
or max-stable, and a chosen parametric family may or may not describe the data.
Hypothesis tests turn each claim into a discrepancy that can be compared with
the fluctuations expected under a null model.

This page develops tests of:

* mutual independence;
* exchangeability;
* radial symmetry;
* extreme-value dependence (max-stability);
* goodness of fit for a specified copula;
* goodness of fit for a fitted copula family.

The common pattern is simple: define a property of the unknown copula, measure
its violation with the empirical copula, then calibrate that discrepancy by a
simulation, randomization, multiplier method or parametric bootstrap. The
procedures draw on

[genest2004independence](@cite),
[fermanian2004empirical](@cite),
[remillard2009equality](@cite),
[bucher2010bootstrap](@cite), and
[genest2009gof](@cite).

## From observations to an empirical copula

As elsewhere in `Copulas.jl`, observations are represented by a `d\times n` matrix

```math
U =
\begin{pmatrix}
U_{11} & \cdots & U_{1n}\\
\vdots &        & \vdots\\
U_{d1} & \cdots & U_{dn}
\end{pmatrix},
```

where each column

```math
\boldsymbol U_i = (U_{1i},\ldots,U_{di})^\top
```

is one `d`-dimensional observation.

The implemented tests are rank based. When

```julia
pseudo_values=false
```

the input matrix is transformed internally with [`pseudos`](@ref). If the data already consist of pseudo-observations in `[0,1]^d`, use

```julia
pseudo_values=true
```

to avoid ranking them again.

::: warning Continuous margins and ties

The currently implemented copula hypothesis tests assume continuous margins and therefore require tie-free observations in every margin. Tied or discrete data are rejected with an `ArgumentError`.

This is intentional: ordinal ranking would otherwise assign distinct ranks to tied observations and could produce apparently valid p-values without the tie-aware empirical-process or bootstrap theory required for such data. Tie-aware procedures are outside the scope of the current implementation.

:::

::: definition Empirical copula

Given pseudo-observations $\boldsymbol U_1,\ldots,\boldsymbol U_n$, the empirical copula is

```math
C_n(\boldsymbol u)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf 1
\left(
\boldsymbol U_i\le\boldsymbol u
\right),
```

where the inequality is understood componentwise.

:::

Empirical-copula processes and their weak convergence form the theoretical basis for many of the statistics and multiplier approximations used below [fermanian2004empirical](@cite).

## Reading a test result

All tests return a [`CopulaTest`](@ref), which implements `StatsAPI.HypothesisTest`.

For example,

```@example hypothesis_testing
using Copulas, Distributions, Random, StatsBase

U = rand(Xoshiro(123), ClaytonCopula(2, 3.0), 80)

test = IndependenceCopulaTest(U; N=49, rng=Xoshiro(123),)

nothing # hide
```

The common result interface is

```@example hypothesis_testing
teststatistic(test)
```

```@example hypothesis_testing
pvalue(test)
```

```@example hypothesis_testing
nobs(test)
```

The statistic measures disagreement with the null; the p-value calibrates how
unusual that disagreement is under the selected resampling scheme. It is not
the probability that the null hypothesis is true. The public result interface
consists of these accessors and the printed summary; internal fields are not an
extension API.

Printing the object gives a summary of the hypothesis, statistic, calibration, p-value, and relevant test-specific information.

```@example hypothesis_testing
test
```

::: note Number of resamples

The small values of `N` used in the documentation keep the examples fast.
For statistical work, substantially larger values should generally be used,
depending on the desired Monte Carlo precision.

:::

## Mutual independence

### Definition

::: definition Mutual independence

Let `C` denote the copula of the random vector. Mutual independence is equivalent to the product copula

```math
\Pi(\boldsymbol u)
=
\prod_{j=1}^{d}u_j.
```

Thus

```math
H_0:
C(\boldsymbol u)
=
\Pi(\boldsymbol u)
\qquad
\text{for every }
\boldsymbol u\in[0,1]^d.
```

Rank-based independence tests constructed from the empirical copula process are studied in [genest2004independence](@cite).

:::

### Measuring departure from independence

The statistic currently available in `Copulas.jl` is `:cvm`. The implementation evaluates the squared discrepancy between the empirical copula and the product copula at the observed pseudo-observations:

```math
S_n^{\mathrm{ind}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
\prod_{j=1}^{d}U_{ji}
\right]^2.
```

Large values indicate departure from mutual independence, but their scale
depends on the sample size and dimension; the statistic should therefore be
read through its calibrated p-value rather than against a fixed threshold.

### Why simulation provides the reference distribution

Under `H_0`, the coordinates are independent uniforms. The default calibration is therefore `:simulation`:

1. generate `n` observations from the `d`-dimensional product copula;
2. transform the generated sample to pseudo-observations;
3. recompute `S_n^{\mathrm{ind}}`;
4. repeat the procedure `N` times;
5. compare the observed statistic with its simulated null distribution.

### Example

```@example hypothesis_testing
Uind = rand(Xoshiro(1), IndependentCopula(3), 100)

tind = IndependenceCopulaTest(Uind; N=49, rng=Xoshiro(2),)

(teststatistic(tind), pvalue(tind))
```


## Exchangeability

### Definition

::: definition Exchangeability

A copula `C` is exchangeable when it is invariant under permutations of its coordinates.

For a permutation

```math
\pi:
\{1,\ldots,d\}
\longrightarrow
\{1,\ldots,d\},
```

write

```math
\boldsymbol u_\pi
=
(u_{\pi(1)},\ldots,u_{\pi(d)}).
```

Full exchangeability means

```math
H_0:
C(\boldsymbol u)
=
C(\boldsymbol u_\pi)
```

for every `\boldsymbol u\in[0,1]^d` and every coordinate permutation `\pi`.

Empirical-copula tests for bivariate symmetry were developed in [genest2012symmetry](@cite) and extended to arbitrary dimension by [harder2017exchangeability](@cite).

:::

### Measuring sensitivity to coordinate labels

For a collection `\mathcal G` of non-identity permutations, the implemented statistic is

```math
S_n^{\mathrm{ex}}
=
\sum_{\pi\in\mathcal G}
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_n(\boldsymbol U_{i,\pi})
\right]^2
w_\pi(\boldsymbol U_i).
```

The default weight is `weight=:wm2`.

Let

```math
m(\boldsymbol u)
=
\min_{1\le j\le d}u_j,
```

and

```math
b(\boldsymbol u)
=
d-1+m(\boldsymbol u)-\sum_{j=1}^{d}u_j.
```

For a transposition exchanging coordinates `a` and `b`, define

```math
\omega_\pi(\boldsymbol u)
=
|u_a-u_b|.
```

For a general permutation, let

```math
u_{(1)}\le\cdots\le u_{(d)}
```

denote the ordered coordinates and define the implementation's permutation separation term by

```math
\omega_\pi(\boldsymbol u)
=
\sum_{k=\lceil d/2\rceil+1}^{d}
\left(
u_{(k)}-m(\boldsymbol u)
\right).
```

The `:wm2` weight is then

```math
w_\pi(\boldsymbol u)
=
\left[
\max
\left\{
0,
\min
\left(
m(\boldsymbol u),
\omega_\pi(\boldsymbol u),
b(\boldsymbol u)
\right)
\right\}
\right]^2.
```

Alternatively,

```julia
weight=:none
```

sets `w_\pi(\boldsymbol u)=1`.

### Which permutations are compared?

The keyword `permutations` controls the set `\mathcal G`.

#### `permutations=:G2`

This is the default.

For `d=2`, it contains the only nontrivial transposition,

```math
(12).
```

For `d>2`, it uses the transposition

```math
(12)
```

together with the cyclic left shift

```math
(12\cdots d).
```

#### `permutations=:G1`

Uses the transpositions

```math
(12),(13),\ldots,(1d).
```

An explicit permutation or collection may also be supplied. Duplicate permutations
are removed. Collections are normalized once and stored as vectors.

!!! warning "Permutation sets and memory"
    The factorial choice `permutations=:all` is deliberately unsupported. The
    multiplier procedure retains one dense `n × n` matrix per permutation; a
    shared guard rejects matrix payloads above 512 MiB, before auxiliary storage.

### Why multiplier calibration is needed

The default calibration is `:multiplier`.

The empirical-copula process has a nontrivial correction caused by replacing the unknown margins with ranks. The implementation therefore constructs the corresponding multiplier representation, including finite-difference estimates of the partial derivatives of $C_n$.

The derivative bandwidth is

```math
h_n=n^{-1/2}.
```

For coordinate $j$, the derivative is approximated by a boundary-corrected finite difference of the form

```math
\dot C_{n,j}(\boldsymbol u)
\approx
\frac{
C_n(\boldsymbol u+h_n\boldsymbol e_j)
-
C_n(\boldsymbol u-h_n\boldsymbol e_j)
}{
\text{effective width}
}.
```

Independent exponential multipliers are generated and centered before applying the empirical-process representation. This type of multiplier approximation is closely related to the methods discussed in [remillard2009equality](@cite), [bucher2010bootstrap](@cite), and [harder2017exchangeability](@cite).

### Example

```@example hypothesis_testing
Uex = rand(Xoshiro(4), GumbelCopula(3, 2.0), 80)

tex = ExchangeabilityCopulaTest(Uex; permutations=:G2, weight=:wm2, N=49, rng=Xoshiro(5),)

(teststatistic(tex), pvalue(tex))
```


## Radial symmetry

### Definition

::: definition Radial symmetry

A copula is radially symmetric when

```math
\boldsymbol U
\overset{d}{=}
\boldsymbol 1-\boldsymbol U.
```

Equivalently, if $C^{\mathrm{rad}}$ denotes the copula of $\boldsymbol 1-\boldsymbol U$, then

```math
H_0:
C
=
C^{\mathrm{rad}}.
```

Nonparametric tests of copula symmetry and randomization procedures based on the corresponding invariance group are studied in [beare2020symmetry](@cite).

:::

### Comparing a sample with its reflection

Let $C_n$ denote the empirical copula of the original pseudo-observations and let $\bar{C_n}$ denote the empirical copula constructed from

```math
\boldsymbol 1-\boldsymbol U_1,
\ldots,
\boldsymbol 1-\boldsymbol U_n.
```

The implemented statistic is

```math
S_n^{\mathrm{rad}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
\bar C_n(\boldsymbol U_i)
\right]^2.
```

Large values indicate radial asymmetry.

### Why reflection gives a null experiment

Under radial symmetry, an observation and its radial reflection are distributionally equivalent. For every observation $i$, independently generate

```math
B_i\sim\operatorname{Bernoulli}(1/2),
```

and construct

```math
\boldsymbol U_i^\star
=
\begin{cases}
\boldsymbol U_i,
&
B_i=0,\\[2mm]
\boldsymbol 1-\boldsymbol U_i,
&
B_i=1.
\end{cases}
```

The randomized sample is converted back to pseudo-observations before the statistic is evaluated. Because radial reflection of rank-grid values can create exact ties even when the original sample is tie-free, ties induced by the randomization itself are reranked using average ranks. This does not relax the requirement that the original input margins be tie-free.

Thus the default reflection probability is exactly

```math
\Pr(B_i=1)=\frac12.
```

The procedure exploits the group invariance associated with radial symmetry, following the randomization-testing principle developed in [beare2020symmetry](@cite).

### Example

```@example hypothesis_testing
Urad = rand(Xoshiro(6), GaussianCopula([1.0 0.6; 0.6 1.0]), 80)

trad = RadialSymmetryCopulaTest(Urad; N=49, rng=Xoshiro(7),)

(teststatistic(trad), pvalue(trad))
```


## Extreme-value dependence

### The defining property

::: property Max-stability

Extreme-value copulas are characterized by max-stability. For any $r>0$,

```math
C(u_1^r,\ldots,u_d^r)
=
C(u_1,\ldots,u_d)^r.
```

Equivalently, for $r>1$,

```math
C(\boldsymbol u)
=
C(\boldsymbol u^{1/r})^r,
```

where

```math
\boldsymbol u^{1/r}
=
(u_1^{1/r},\ldots,u_d^{1/r}).
```

This characterization provides a direct way to test

```math
H_0:
C\text{ belongs to the extreme-value class}.
```

Large-sample tests based on this max-stability identity, the empirical copula, and multiplier approximations are developed by [kojadinovic2011extremevalue](@cite).

:::

### Measuring violations of max-stability

For a finite collection of powers

```math
\mathcal R
=
\{r_1,\ldots,r_K\},
\qquad
r_k>1,
```

the implemented statistic is

```math
S_n^{\mathrm{EV}}
=
\sum_{r\in\mathcal R}
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i^{1/r})^r
-
C_n(\boldsymbol U_i)
\right]^2.
```

The default powers are

```math
\mathcal R=\{3,4,5\}.
```

They can be changed through the `powers` keyword.

### Multiplier calibration

Approximate p-values are obtained from a multiplier representation of the empirical-copula process, following the max-stability testing strategy in
[kojadinovic2011extremevalue](@cite).

As in the exchangeability test, the finite-difference bandwidth used for the empirical partial derivatives is

```math
h_n=n^{-1/2}.
```

The multiplier variables are exponential and centered before the bootstrap process is evaluated.

### Example

```@example hypothesis_testing
Uev = rand(Xoshiro(8), GumbelCopula(2, 2.5), 80)

tev = ExtremeValueCopulaTest(Uev; powers=3:5, N=49, rng=Xoshiro(9),)

(teststatistic(tev), pvalue(tev))
```

A single power is also allowed:

```julia
ExtremeValueCopulaTest(U; powers=2)
```

All supplied powers must be finite and strictly larger than one.


## Goodness of fit

Copula goodness-of-fit procedures compare the empirical dependence structure with a proposed parametric copula model. Empirical-process and Cramér--von Mises procedures of this form are reviewed extensively in [genest2009gof](@cite).

`Copulas.jl` distinguishes a **simple** null hypothesis, in which every
parameter is fixed before seeing the data, from a **composite** null hypothesis,
in which parameters are estimated. That distinction changes what the bootstrap
must reproduce.

### A fully specified copula

::: definition Simple null hypothesis

Suppose that a fully specified copula `C_0` is given, including all its parameters.

The null hypothesis is

```math
H_0:
C=C_0.
```

:::

The implemented Cramér--von Mises-type statistic is

```math
S_n^{\mathrm{GOF}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_0(\boldsymbol U_i)
\right]^2.
```

Use

```@example hypothesis_testing
C0 = ClaytonCopula(2, 3.0)
Ugof = rand(Xoshiro(10), C0, 80)

tsimple = GOFCopulaTest(C0, Ugof; N=49, rng=Xoshiro(11),)

pvalue(tsimple)
```

This tests the fully specified copula.

#### Parametric bootstrap

For every bootstrap replicate:

1. generate `n` observations from `C_0`;
2. transform the sample to pseudo-observations;
3. compute the same goodness-of-fit statistic;
4. compare the bootstrap statistic with the observed value.

No parameters are re-estimated because `C_0` is fully specified.

### A fitted copula family

::: definition Composite null hypothesis

Suppose instead that

```math
\mathcal C
=
\{
C_\theta:\theta\in\Theta
\}
```

is a parametric copula family and that $\widehat\theta$ is estimated from the data.

The null hypothesis becomes

```math
H_0:
C\in\mathcal C,
```

:::

and the observed statistic is

```math
S_n^{\mathrm{GOF}}
=
\sum_{i=1}^{n}
\left[
C_n(\boldsymbol U_i)
-
C_{\widehat\theta}(\boldsymbol U_i)
\right]^2.
```

Because $\widehat\theta$ is estimated, the uncertainty introduced by fitting must also be reproduced in the bootstrap. Parametric-bootstrap validity for this type of semiparametric goodness-of-fit problem is studied in [genest2008bootstrap](@cite); practical copula GOF procedures and their finite sample behavior are discussed in [genest2009gof](@cite).

In `Copulas.jl`, every composite bootstrap replicate performs the following steps:

```math
\boldsymbol U_1^\star,\ldots,\boldsymbol U_n^\star
\sim
C_{\widehat\theta},
```

then refits using the **same estimator specification**,

```math
\widehat\theta^\star
=
\operatorname{fit}
\left(
\boldsymbol U_1^\star,\ldots,\boldsymbol U_n^\star
\right),
```

and computes

```math
S_n^\star
=
\sum_{i=1}^{n}
\left[
C_n^\star(\boldsymbol U_i^\star)
-
C_{\widehat\theta^\star}(\boldsymbol U_i^\star)
\right]^2.
```

Thus parameter estimation is repeated inside every bootstrap replicate rather than treating the fitted parameters as fixed.

The fitting procedure itself is also reproduced. In particular, estimator-defining runtime information such as the fitting method, method-specific keywords, and copula structure is retained whenever the fitted model records a reproducible fitting specification.
This matters for models whose fitting procedure cannot be reconstructed from the fitted copula type alone.

### Example

First fit a model:

```@example hypothesis_testing
M = fit(CopulaModel, ClaytonCopula, Ugof; method=:itau, vcov=false,)

nothing # hide
```

Then run the test directly from the fitted model:

```@example hypothesis_testing
tcomposite = GOFCopulaTest(M; N=49, rng=Xoshiro(12),)

pvalue(tcomposite)
```

`GOFCopulaTest(M)` tests the data used to fit `M`. The stored fitting input is preprocessed consistently with the original fit before the observed statistic is computed. The fitted model `M` supplies both the estimated null model and the estimator specification that is replayed in every bootstrap replicate.

A separate sample can be tested with

```julia
GOFCopulaTest(M, U)
```

In this form, `M` is interpreted as an **estimator specification**, not as a fixed set of parameter estimates. The same fitting procedure is first reapplied to `U`, so the observed statistic uses parameters estimated from the sample being tested. Each parametric-bootstrap replicate then repeats that same fitting procedure.

Consequently, the observed statistic and every bootstrap statistic are based on the same estimation rule. If the original fitting procedure cannot be reproduced safely, composite goodness-of-fit testing raises an `ArgumentError` rather than silently replacing it by a different estimator.


## Interpreting resampling p-values

Simulation, randomization and parametric bootstrap use
`(0.5 + count(Tstar >= Tobserved)) / (N + 1)`.
The EV multiplier test uses the same convention; the exchangeability multiplier
test uses the uncorrected strict proportion `count(Tstar > Tobserved) / N`.
These are resampling approximations, not exact finite-sample level guarantees.
Increasing `N` reduces Monte Carlo noise but does not repair an inappropriate
null hypothesis, dependent observations or violated regularity assumptions.

### Scope and reference variants

Observations must be independent and identically distributed. Continuous,
tie-free margins alone do not establish the regularity assumptions required
by empirical-copula multiplier theory; consult the cited procedures before
applying them to singular models.

The radial randomization implemented here reranks using average ranks.
It is a variant of the bivariate procedure of [beare2020symmetry](@cite),
which additionally breaks induced ties with a small random perturbation.
The multidimensional version implemented here extends the reflection and
reranking operations; the cited bivariate theorem does not by itself establish
its validity in higher dimensions.

The EV statistic uses the empirical copula directly, without the finite-sample
offset used by some implementations. Its multiplier approximation and default
powers `3:5` follow [kojadinovic2011extremevalue](@cite). Do not expect identical
finite-sample p-values from implementations with different corrections.
