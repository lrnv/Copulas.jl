```@meta
CurrentModule = Copulas
```

# [General Discussion](@id archimedean_copulas_header)

An important parametric class of copulas is the class of Archimedean copulas. To define Archimedean copulas, we must take a look at their *generators*, which are unrelated to spherical generators, and must be $d$-monotone functions. 

## Generators and d-monotony

Archimedean generators can be defined as follows:
!!! definition "Definition (Archimedean generator):" 
    A $d$-Archimedean generator is a $d$-monotone function 

    $\phi :\mathbb R_+ \to [0,1]$ such that $\phi(0) = 1$ and $\phi(+\infty) = 0$.

where the notion of $d$-monotone function is defined (see e.g. [mcneil2009](@cite)) as follows:

!!! definition "Definition (d-monotony):"
    A function $\phi$ is said to be $d$-monotone if it has $d-2$ derivatives which satisfy 

    $(-1)^k \phi^{(k)} \ge 0 \;\forall k \in \{1,..,d-2\},$ and if $(-1)^{d-2}\phi^{(d-2)}$ is a non-increasing and convex function. 

    A function that is $d$-monotone for all $d$ is called **completely monotone**.


In this package, there is an abstract class [`Generator`](@ref) that contains those generators. Many Archimedean generators are already implemented for you ! See [the list of implemented archimedean generator](@ref available_archimedean_models) to get an overview. 

If you do not find the one you need, you may define it yourself by subtyping `Generator`. The API does not ask for much information, which is really convenient. Only the two following methods are required:

* The `ϕ(G::MyGenerator,t)` function returns the value of the archimedean generator itself. 
* The `max_monotony(G::MyGenerator)` returns its maximum monotony, that is the greater integer $d$ making the generator $d$-monotonous.

Thus, a new generator implementation may simply look like:

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
ϕ(G::MyGenerator,t) = exp(-G.θ * t) # can you recognise this one ?
max_monotony(G::MyGenerator) = Inf
```
!!! tip "Win-Win strategy"
    These two functions are enough to sample the corresponding Archimedean copula (see how in the [Inverse Williamson $d$-transforms](@ref w_trans_section) section of the documentation). However, if you know a bit more about your generator, implementing a few more simple methods can largely fasten the algorithms. You'll find more details on these methods in the [`Generator`](@ref) docstring.


For example, Here is a graph of a few Clayton Generators: 
```@example
using Copulas: ϕ,ClaytonGenerator,IndependentGenerator
using Plots
plot( x -> ϕ(ClaytonGenerator(-0.5),x), xlims=(0,5), label="ClaytonGenerator(-0.5)")
plot!(x -> ϕ(IndependentGenerator(),x), label="IndependentGenerator()")
plot!(x -> ϕ(ClaytonGenerator(0.5),x), label="ClaytonGenerator(0.5)")
plot!(x -> ϕ(ClaytonGenerator(1),x), label="ClaytonGenerator(1)")
plot!(x -> ϕ(ClaytonGenerator(5),x), label="ClaytonGenerator(5)")
```

And the corresponding inverse functions: 

```@example
using Copulas: ϕ⁻¹,ClaytonGenerator,IndependentGenerator
using Plots
plot( x -> ϕ⁻¹(ClaytonGenerator(-0.5),x), xlims=(0,1), ylims=(0,5), label="ClaytonGenerator(-0.5)")
plot!(x -> ϕ⁻¹(IndependentGenerator(),x), label="IndependentGenerator()")
plot!(x -> ϕ⁻¹(ClaytonGenerator(0.5),x), label="ClaytonGenerator(0.5)")
plot!(x -> ϕ⁻¹(ClaytonGenerator(1),x), label="ClaytonGenerator(1)")
plot!(x -> ϕ⁻¹(ClaytonGenerator(5),x), label="ClaytonGenerator(5)")
```

```@docs
Generator
```

Note that the rate at which these functions are reaching 0 (and their inverse reaching infinity on the left boundary) can vary a lot from one to the other. Note also that the difference between each of them is easier to grasp on the inverse plot. 




## Williamson d-transform

An easy way to construct new $d$-monotonous generators is the use of the Williamson $d$-transform.

!!! definition "Definition (Williamson d-transformation):"
    For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer $d\ge 2$, the Williamson-d-transform of ``X`` is the real function supported on $[0,\infty[$ given by:

    $\phi(t) = 𝒲_{d}(X)(t)$
    $=\int_{t}^{\infty} \left(1 - \frac{t}{x}\right)^{d-1} dF(x)$
    $= \mathbb E\left( (1 - \frac{t}{X})^{d-1}_+\right) \mathbb 1_{t > 0} + \left(1 - F(0)\right)\mathbb 1_{t <0}$

In this package, we implemented it through the [`WilliamsonGenerator`](@ref) class. It can be used as follows: 

`WilliamsonGenerator(X::UnivariateRandomVariable, d)`.

This function computes the Williamson d-transform of the provided random variable $X$ using the [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl) package. See [williamson1955multiply, mcneil2009](@cite) for the literature. 

!!! warn "`max_monotony` of Williamson generators"
    The $d$-transform of a positive random variable is $d$-monotonous but not $k$-monotonous for any $k > d$. Its max monotony is therefore $d$. This has a few implications, one of the biggest one is that the $d$-variate Archimedean copula that corresponds has no density. 
    
    More genrally, if you want your Archimedean copula to have a density, you have to use a generator that is more-monotonous that the dimension of your model. 

```@docs
WilliamsonGenerator
```

## [Inverse Williamson d-transform](@id w_trans_section)

The Williamson d-transform is a bijective transformation[^1] from the set of positive random variables to the set of generators. It therefore has an inverse transformation (called, surprisingly, the inverse Williamson $d$-transform) that construct the positive random variable *R* from a generator $\phi$.

[^1]:

    This bijection is to be taken carefuly: the bijection is between random variables *with unit scales* and generators *with common value at 1*, sicne on both rescaling does not change the underlying copula. 

This transformation is implemented through one method in the Generator interface that is worth talking a bit about : `williamson_dist(G::Generator, d)`. This function computes the inverse Williamson d-transform of the d-monotone archimedean generator ϕ, still using the [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl) package. See [williamson1955multiply, mcneil2009](@cite).

To put it in a nutshell, for ``\phi`` a ``d``-monotone archimedean generator, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``R``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\phi)(x) = 1 - \frac{(-x)^{d-1} \phi_+^{(d-1)}(x)}{k!} - \sum_{k=0}^{d-2} \frac{(-x)^k \phi^{(k)}(x)}{k!}
```

The [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl) package implements this transformation (and its inverse, the Williamson d-transfrom) in all generality. It returns this cumulative distribution function in the form of the corresponding random variable `<:Distributions.ContinuousUnivariateDistribution` from `Distributions.jl`. You may then compute : 
* The cdf via `Distributions.cdf`
* The pdf via `Distributions.pdf` and the logpdf via `Distributions.logpdf`
* Samples from the distribution via `rand(X,n)`.


As an example of a generator produced by the Williamson transformation and its inverse, we propose to construct a generator from a LogNormal distribution:

```@example
using Distributions
using Copulas: i𝒲, ϕ⁻¹, IndependentGenerator
using Plots
G = i𝒲(LogNormal(), 2)
plot(x -> ϕ⁻¹(G,x), xlims=(0.1,0.9), label="G")
plot!(x -> ϕ⁻¹(IndependentGenerator(),x), label="Independence")
```

The `i𝒲` alias stands for `WiliamsonGenerator`. To stress the generality of the approach, remark that any positive distribution is allowed, including discrete ones: 

```@example
using Distributions
using Copulas: i𝒲, ϕ⁻¹
using Plots
G1 = i𝒲(Binomial(10,0.3), 2)
G2 = i𝒲(Binomial(10,0.3), 3)
plot(x -> ϕ⁻¹(G1,x), xlims=(0.1,0.9), label="G1")
plot!(x -> ϕ⁻¹(G2,x), xlims=(0.1,0.9), label="G2")
```

As obvious from the definition of the Williamson transform, using a discrete distribution produces piecewise-linear generators, where the number of pieces is dependent on the order of the transformation. 

## Archimedean copulas

Let's first define formally archimedean copulas: 

!!! definition "Definition (Archimedean copula):"
    If $\phi$ is a $d$-monotonous Archimedean generator, then the function 

    $$C(\bm u) = \phi\left(\sum\limits_{i=1}^d \phi^{-1}(u_i)\right)$$ is a copula. 

There are a few archimedean generators that are worth noting since they correspond to known archimedean copulas families: 
* [`IndependentGenerator`](@ref): $\phi(t) =e^{-t} \text{ generates } \Pi$.
* [`ClaytonGenerator`](@ref): $\phi_{\theta}(t) = \left(1+t\theta\right)^{-\theta^{-1}}$ generates the $\mathrm{Clayton}(\theta)$ copula.
* [`GumbelGenerator`](@ref): $\phi_{\theta}(t) = \exp\{-t^{\theta^{-1}}\}$ generates the $\mathrm{Gumbel}(\theta)$ copula.
* [`FrankGenerator`](@ref): $\phi_{\theta}(t) = -\theta^{-1}\ln\left(1+e^{-t-\theta}-e^{-t}\right)$ generates the $\mathrm{Franck}(\theta)$ copula.

There are a lot of others implemented in the package, see our [large list of implemented archimedean generator](@ref available_archimedean_models). 

Archimedean copulas have a nice decomposition, called the Radial-simplex decomposition, developed in [mcneil2008,mcneil2009](@cite): 

!!! property "Property (Radial-simplex decomposition ):"
    A $d$-variate random vector $\bm U$ following an Archimedean copula with generator $\phi$ can be decomposed into 

    $\bm U = \phi.(\bm S R),$
    where $\bm S$ is uniform on the $d$-variate simplex and $R$ is a non-negative random variable, independent form $\bm S$, defined as the inverse Williamson $d$-transform of $\phi$.  


This is why `williamson_dist(G::Generator,d)` is such an important function in the API: it allows to generator the radial part and sample the Archimedean copula. You may call this function directly to see what distribution will be used: 

```@example
using Copulas: williamson_dist, FrankCopula
williamson_dist(FrankCopula(3,7))
```

For the Frank Copula, as for many classic copulas, the distribution used is known. We pull some of them from `Distributions.jl` but implement a few more, as this Logarithmic one. Another useful example are negatively-dependent Clayton copulas: 

```@example
using Copulas: williamson_dist, ClaytonCopula
williamson_dist(ClaytonCopula(3,-0.2))
```

for which the corresponding distribution is known but has no particular name, thus we implemented it under the `ClaytonWilliamsonDistribution` name.

!!! note "Frailty decomposition for completely monotonous generators"
    It is well-known that completely monotone generators are Laplace transforms of non-negative random variables. This gives rise to another decomposition in [hofert2013](@cite):

    !!! property "Property (Frailty decomposition):"
        When $\phi$ is completely monotone, it is the Laplace transform of a non-negative random variable $W$ such that

        $$\bm U = \phi(\bm Y / W),$$  where $\bm Y$ is a vector of independent and identically distributed (i.i.d.) exponential distributions.

    The link between the distribution of $R$ and the distribution of $W$ can be explicited. We exploit this link and provide the `WilliamsonFromFrailty()` constructor that construct the distribution of $R$ from the distribution of $W$ and returns the corresponding  `WilliamsonGenerator` from the frailty distribution itself. The corresponding ϕ is simply the laplace transform of $W$. This is another potential way of constructing new archimedean copulas !  

    We use this fraily approach for several generators, since sometimes it is faster, including e.g. the Clayton one with positive dependence:
    ```@example
    using Copulas: williamson_dist, ClaytonCopula
    williamson_dist(ClaytonCopula(3,10))
    ```

```@docs
ArchimedeanCopula
```


<!-- 
TODO: Make a few graphs of bivariate archimedeans pdfs and cdfs. And provide a few more standard tools for these copulas ? 
-->

```@bibliography
Pages = ["generalities.md"]
Canonical = false
```
