```@meta
CurrentModule = Copulas
```
# [Elliptical Copulas](@id elliptical_copulas_header)


The easiest families of copulas are the one derived from known families of random vectors, and the first presented one is, generally, the Elliptical family. 

> **Definition (Spherical and elliptical random vectors):** A random vector $\bm X$ is said to be spherical if for all orthogonal matrix $\bm A \in O_d(\mathbb R)$, $\bm A\bm X \sim \bm X$. 
>
> For every matrix $\bm B$ and vector $\bm c$, the random vector $\bm B \bm X + \bm c$ is then said to be elliptical.


Spherical random vectors have several interesting properties. First, the shape of the distribution must be the same in every direction since it is stable by rotations. Moreover, their characteristic functions (c.f.) only depend on the norm of their arguments. Indeed, for any $\bm A \in O_d(\mathbb R)$, 
```math
\phi(\bm t) = \mathbb E\left(e^{\langle \bm t, \bm X \rangle}\right)= \mathbb E\left(e^{\langle \bm t, \bm A\bm X \rangle}\right) = \mathbb E\left(e^{\langle \bm A\bm t, \bm X \rangle}\right) = \phi(\bm A\bm t).
```

We can therefore express this characteristic function as $\phi(\bm t) = \psi(\lVert \bm t \rVert_2^2)$, where $\psi$ is a function that characterizes the spherical family, called the *generator* of the family. Any characteristic function that can be expressed as a function of the norm of its argument is the characteristic function of a spherical random vector, since $\lVert \bm A \bm t \rVert_2 = \lVert \bm t \rVert_2$ for any orthogonal matrix $\bm A$. 

This class contains the (multivariate) Normal and Student distributions, and it is easy to construct others if needed. This is a generalization of the family of Gaussian random vectors, and they benefit from several nice properties of the former, among which, particularly interesting, the stability by convolution. Indeed, convolutions correspond to product of characteristic functions, and
```math
\phi(\bm t) = \prod_{i=1}^n \phi_i(\bm t) = \prod_{i=1}^n \psi_i(\lVert \bm t \rVert_2^2) = \psi(\lVert \bm t \rVert_2^2),
```
which is still a function of only the norm of $\bm t$. To fix ideas, for Gaussian random vectors, $\psi(t) = e^{-\frac{t^2}{2}}$.



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


!!! note "Discrepancy with the code"
    If the exposition we just did on characteristic functions of Elliptical random vectors is fundamental to the definition of elliptical copulas, the package does not use this at all to function, and rather rely on the existence of multivariate and corresponding univariate families of distributions in `Distributions.jl`. 



```@docs
EllipticalCopula
```


```@bibliography
Pages = ["generalities.md"]
Canonical = false
```