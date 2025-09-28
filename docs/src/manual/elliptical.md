```@meta
CurrentModule = Copulas
```
# [Elliptical family](@id elliptical_copulas_header)

## Definition

The easiest families of copulas are the one derived from known families of random vectors, and the first presented one are, generally, the Elliptical families (in particular, the Gaussian and Student families are very standard in the litterature). 

!!! definition "Spherical and elliptical random vectors" 
    A random vector $\boldsymbol X$ is said to be spherical if for all orthogonal matrix $\boldsymbol A \in O_d(\mathbb R)$, $\boldsymbol A\boldsymbol X \sim \boldsymbol X$. 

    For every matrix $\boldsymbol B$ and vector $\boldsymbol c$, the random vector $\boldsymbol B \boldsymbol X + \boldsymbol c$ is then said to be elliptical.


Spherical random vectors have several interesting properties. First, the shape of the distribution must be the same in every direction since it is stable by rotations. Moreover, their characteristic functions (c.f.) only depend on the norm of their arguments. Indeed, for any $\boldsymbol A \in O_d(\mathbb R)$, 
```math
\phi(\boldsymbol t) = \mathbb E\left(e^{\langle \boldsymbol t, \boldsymbol X \rangle}\right)= \mathbb E\left(e^{\langle \boldsymbol t, \boldsymbol A\boldsymbol X \rangle}\right) = \mathbb E\left(e^{\langle \boldsymbol A\boldsymbol t, \boldsymbol X \rangle}\right) = \phi(\boldsymbol A\boldsymbol t).
```

We can therefore express this characteristic function as $\phi(\boldsymbol t) = \psi(\lVert \boldsymbol t \rVert_2^2)$, where $\psi$ is a function that characterizes the spherical family, called the *generator* of the family. Any characteristic function that can be expressed as a function of the norm of its argument is the characteristic function of a spherical random vector, since $\lVert \boldsymbol A \boldsymbol t \rVert_2 = \lVert \boldsymbol t \rVert_2$ for any orthogonal matrix $\boldsymbol A$. 

This class contains the (multivariate) Normal and Student distributions, and it is easy to construct others if needed. This is a generalization of the family of Gaussian random vectors, and they benefit from several nice properties of the former, among which, particularly interesting, the stability by convolution. Indeed, convolutions correspond to product of characteristic functions, and
```math
\phi(\boldsymbol t) = \prod_{i=1}^n \phi_i(\boldsymbol t) = \prod_{i=1}^n \psi_i(\lVert \boldsymbol t \rVert_2^2) = \psi(\lVert \boldsymbol t \rVert_2^2),
```
which is still a function of only the norm of $\boldsymbol t$. 

To fix ideas, for Gaussian random vectors, $\psi(t) = e^{-\frac{t^2}{2}}$.

!!! note "Sampling with `Distributions.jl`"
    Elliptical random vectors in the Gaussian and Student families are available from `Distributions.jl`:

    ```@example 3
    using Distributions
    Σ = [1 0.5
        0.5 1] # variance-covariance matrix.
    ν = 3 # number of degrees of freedom for the student.
    N = MvNormal(Σ)
    ```

    ```@example 3
    T = MvTDist(ν,Σ)
    ```



Elliptical copulas are simply copulas of elliptical distributions. This simplicity of definition is paid for in the expression of the copulas itself: the obtained function has usually no better expression than: 
```math
C = F \circ (F_1^{-1},...,F_d^{-1}),
```
where $F_i^{-1}$ denotes the almost-inverse of $F_i$, that is: 
```math
\forall u \in [0,1],\;F_i^{-1}(u) = \inf\left\{x :\, F_i(x) \ge u\right\},
```
and $F_i$ is usually hard to express from the elliptical assumptions.

Moreover, the form of dependence structures that can be reached inside this class is restricted. The elliptical copulas are parametrized by the corresponding univariate spherical generator and a correlation matrix, which is a very simple structure. See also [frahm2003,gomez2003,cote2019](@cite) for details on these copulas. 

On the other hand, there exist performant estimators of high-dimensional covariance matrices, and a large theory is built on the elliptical assumption of high dimensional random vectors, see e.g., [elidan2013,friedman2010,muller2019](@cite) among others. See also [derumigny2022](@cite) for a recent work on nonparametric estimation of the underlying univariate spherical distribution. 


!!! note "Note on internal implementation"
    If the exposition we just did on characteristic functions of Elliptical random vectors is fundamental to the definition of elliptical copulas, the package does not use this at all to function, and rather rely on the existence of multivariate and corresponding univariate families of distributions in `Distributions.jl`. 


You can obtain these elliptical copulas by the following code: 
```julia
using Copulas
Σ = [1 0.5
     0.5 1] # variance-covariance matrix.
ν = 3 # number of degrees of freedom for the student.
C_N = GaussianCopula(Σ)
C_T = TCopula(ν,Σ)
```

As already stated, the underlying code simply applies Sklar. In all generalities, you may define another elliptical copula by the following structure: 

```julia
struct MyElliptical{d,T} <: EllipticalCopula{d,T}
    θ:T
end
U(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the univaraite marginals, Normal() for the Gaussian case. 
N(::Type{MyElliptical{d,T}}) where {d,T} # Distribution of the mutlivariate random vector, MvNormal(\Sigma) for the Gaussian case. 
```

However, not much other cases than the Gaussian and Elliptical one are really used in the literature.

## Examples

To construct, e.g., a Student copula, you need to provide the Correlation matrix and the number of degree of freedom, as follows: 

```@example 4
using Copulas, Distributions
Σ = [1 0.5
    0.5 1] # variance-covariance matrix.
ν = 3 # number of degrees of freedom
C = TCopula(ν,Σ)
```

You can sample it and compute its density and distribution functions via the standard interface. We could try to fit a GaussianCopula on the sampled data, even if we already know that the tails will not be properly taken into account: 

```@example 4
u = rand(C,1000)
Ĉ = fit(GaussianCopula,u) # to fit on the sampled data. 
```

We see that the estimation we have on the correlation matrix is quite good, but rest assured that the tails of the distributions are not the same at all. To see that, let's plot the lower tail function (see [nelsen2006](@cite)) for both copulas: 

```@example 4
using Plots
chi(C,u) = 2 * log(1-u) / log(1 - 2u + cdf(C,[u,u])) -1
u = 0.5:0.03:0.99
plot(u,  chi.(Ref(C),u), label="True student copula")
plot!(u, chi.(Ref(Ĉ),u), label="Estimated Gaussian copula")
``` 

### Visual: Gaussian vs Student (same correlation)

```@example 4
using Plots
Σ = [1 0.7; 0.7 1]
ν = 4
CG = GaussianCopula(Σ)
CT = TCopula(ν, Σ)
plot(plot(CG), plot(CT); layout=(1,2))
```

### Conditional on original scale via SklarDist

```@example 4
XG = SklarDist(CG, (Normal(), Normal()))
XT = SklarDist(CT, (Normal(), Normal()))
X1G = condition(XG, 2, 0.0)
X1T = condition(XT, 2, 0.0)
xgrid = range(quantile(X1T, 0.001), quantile(X1T, 0.999); length=401)
plot(xgrid, Distributions.cdf.(Ref(X1G), xgrid); label="Gaussian", xlabel="x", ylabel="cdf",
    title="F_{X1|X2=0}")
plot!(xgrid, Distributions.cdf.(Ref(X1T), xgrid); label="Student")
```

The difference between the two is not very strong. 

## Implementation

```@docs
EllipticalCopula
```


## Conditionals and distortions

For an elliptical copula built from an underlying elliptical vector $X=(X_1,\dots,X_d)$ with correlation matrix $\Sigma$ and univariate CDFs $(F_i)$, conditioning follows the standard elliptical identities. Partition indices as $I\cup J=\{1,\dots,d\}$ and conformably partition $\Sigma$ as

$$\Sigma = \begin{pmatrix} \Sigma_{II} & \Sigma_{IJ} \\ \Sigma_{JI} & \Sigma_{JJ} \end{pmatrix}.$$

- For the Gaussian copula, the conditional law $X_I\,|\,X_J=x_J$ is Gaussian with

    $$\mu_{I|J} = \Sigma_{IJ}\,\Sigma_{JJ}^{-1}\,x_J, \qquad
    \Sigma_{I|J} = \Sigma_{II} - \Sigma_{IJ}\,\Sigma_{JJ}^{-1}\,\Sigma_{JI}.$$

    Mapping to the copula scale with $u_k = \Phi(x_k)$ and $x_k = \Phi^{-1}(u_k)$ yields the conditional copula via

    $$C_{I|J}(\boldsymbol u_I\mid\boldsymbol u_J) = \Pr\Big[X_I \le \Phi^{-1}(\boldsymbol u_I)\,\Big|\,X_J = \Phi^{-1}(\boldsymbol u_J)\Big],$$

    and the univariate conditional distortions

    $$H_{i|J}(u\mid\boldsymbol u_J)=\Pr\Big[X_i \le \Phi^{-1}(u)\,\Big|\,X_J = \Phi^{-1}(\boldsymbol u_J)\Big] = \Phi\!\Big(\frac{\Phi^{-1}(u) - \mu_{i|J}}{\sqrt{\Sigma_{i|J}}}\Big).$$

- For the Student-$t$ copula with degrees of freedom $\nu$, one uses the standard conditional-$t$ result: $X_I\,|\,X_J=x_J \sim t_{p}(\mu_{I|J},\,\tfrac{\nu + q_{J}}{\nu + r_J}\,\Sigma_{I|J},\,\nu+|J|)$, where $q_J=|J|$ and $r_J = (x_J)^\top\,\Sigma_{JJ}^{-1}\,x_J$.

        With $u_k = F_t(x_k;\,\nu)$ and $x_k = F_t^{-1}(u_k;\,\nu)$ (standard univariate $t$ with df $\nu$), this provides closed forms for $C_{I|J}$ and for

        $$H_{i|J}(u\mid\boldsymbol u_J) = F_t\!\Big(\,F_t^{-1}(u;\,\nu)\,;\,\mu_{i|J},\,\tfrac{\nu + r_J}{\nu + q_J}\,\Sigma_{i|J},\,\nu+|J|\Big),$$

        where $q_J=|J|$, $r_J= x_J^\top\Sigma_{JJ}^{-1}x_J$, and $F_t(\cdot;\,\mu,\sigma^2,\nu)$ is the univariate non-standard $t$ CDF.

These formulas are what the implementation relies on (via `SklarDist` for original scale and via marginal CDF transforms for the copula scale) to compute `condition` and the associated distortions efficiently.


## See also

- Bestiary: [Implemented Elliptical copulas](@ref elliptical_cops)
- Manual: [Getting Started](@ref), [Sklar's Distribution](@ref)


```@bibliography
Pages = [@__FILE__]
Canonical = false
```
