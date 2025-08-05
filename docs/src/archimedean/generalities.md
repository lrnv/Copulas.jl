```@meta
CurrentModule = Copulas
```

# [General Discussion](@id archimedean_copulas_header)

Archimedean copulas are an important parametric class of copulas. To define Archimedean copulas, we must consider their *generators*, which are unrelated to spherical generators and must be $d$-monotone functions. 

## Generators and d-monotony

Archimedean generators can be defined as follows:
!!! definition "Definition (Archimedean generator):" 
    A $d$-Archimedean generator is a $d$-monotone function

    $\phi : \mathbb{R}_+ \to [0,1]$ such that $\phi(0) = 1$ and $\phi(+\infty) = 0$.

where the notion of $d$-monotone function is defined (see e.g. [mcneil2009](@cite)) as follows:

!!! definition "Definition (d-monotony):"
    A function $\phi$ is $d$-monotone if it has $d-2$ derivatives which satisfy

    $(-1)^k \phi^{(k)} \ge 0$ for all $k \in \{1, ..., d-2\}$, and if $(-1)^{d-2}\phi^{(d-2)}$ is a non-increasing and convex function.

    A function that is $d$-monotone for all $d$ is called **completely monotone**.

In this package, there is an abstract class [`Generator`](@ref) that contains those generators.

!!! tip "Available Archimedean generators"
    The package covers every archimedean generators that exists through a generic implementation of the Williamson d-transform, see the next section. 

    On the other hand, many parametric Archimedean generators are specifically implemented, see [this list of implemented archimedean generator](@ref available_archimedean_models) to get an overview of which ones are availiable. 

If you do not find the generator you need, you may define it yourself by subtyping `Generator`. The API requires only two methods:

* The `φ(G::MyGenerator, t)` function returns the value of the Archimedean generator itself.
* The `max_monotony(G::MyGenerator)` returns its maximum monotony, i.e., the greatest integer $d$ for which the generator is $d$-monotone.

Thus, a new generator implementation may simply look like:

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
ϕ(G::MyGenerator,t) = exp(-G.θ * t) # can you recognise this one ?
max_monotony(G::MyGenerator) = Inf
```
!!! tip "Win-Win strategy"
    These two functions are enough to sample the corresponding Archimedean copula (see the [Inverse Williamson $d$-transforms](@ref w_trans_section) section of the documentation). However, if you know more about your generator, implementing a few additional methods can greatly speed up the algorithms. More details on these methods are in the [`Generator`](@ref) docstring.


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

Note that the rate at which these functions approach 0 (and their inverse approaches infinity on the left boundary) can vary significantly between generators. The difference between each is easier to see on the inverse plot.


## Williamson d-transform

An easy way to construct new $d$-monotonous generators is the use of the Williamson $d$-transform.

!!! definition "Definition (Williamson d-transformation):"
    For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and integer $d \ge 2$, the Williamson-d-transform of ``X`` is the real function supported on $[0, \infty[$ given by:

    $\phi(t) = \mathcal{W}_d(X)(t)$
    $= \int_{t}^{\infty} \left(1 - \frac{t}{x}\right)^{d-1} dF(x)$
    $= \mathbb{E}\left( (1 - \frac{t}{X})^{d-1}_+ \right) \mathbb{1}_{t > 0} + (1 - F(0)) \mathbb{1}_{t < 0}$

In this package, we implemented it through the [`WilliamsonGenerator`](@ref) class. It can be used as follows: 

`WilliamsonGenerator(X::UnivariateRandomVariable, d)`.

This function computes the Williamson d-transform of the provided random variable $X$ using the [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl) package. See [williamson1955multiply, mcneil2009](@cite) for the literature. 

!!! note "`max_monotony` of Williamson generators"
    The $d$-transform of a positive random variable is $d$-monotone but not $k$-monotone for any $k > d$. Its max monotony is therefore $d$. This has a few implications, one of the biggest is that the $d$-variate Archimedean copula that corresponds has no density.
    
    More generally, if you want your Archimedean copula to have a density, you must use a generator that is more-monotone than the dimension of your model. 

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

## Archimedean Copulas

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
williamson_dist(FrankGenerator(7), Val{3}())
```

For the Frank Copula, as for many classic copulas, the distribution used is known. We pull some of them from `Distributions.jl` but implement a few more, as this Logarithmic one. Another useful example are negatively-dependent Clayton copulas: 

```@example
using Copulas: williamson_dist, ClaytonCopula
williamson_dist(ClaytonGenerator(-0.2), Val{3}())
```

for which the corresponding distribution is known but has no particular name, thus we implemented it under the `ClaytonWilliamsonDistribution` name.

!!! note "Frailty decomposition for completely monotone generators"
    It is well-known that completely monotone generators are Laplace transforms of non-negative random variables. This gives rise to another decomposition in [hofert2013](@cite):

    !!! property "Property (Frailty decomposition):"
        When $\phi$ is completely monotone, it is the Laplace transform of a non-negative random variable $W$ such that

        $$\bm U = \phi(\bm Y / W),$$  where $\bm Y$ is a vector of independent and identically distributed (i.i.d.) exponential distributions.

    The link between the distribution of $R$ and the distribution of $W$ can be made explicit. We provide the `WilliamsonFromFrailty()` constructor to build the distribution of $R$ from the distribution of $W$ and return the corresponding `WilliamsonGenerator` from the frailty distribution itself. The corresponding φ is simply the Laplace transform of $W$. This is another way to construct new Archimedean copulas !  

    We use this fraily approach for several generators, since sometimes it is faster, including e.g. the Clayton one with positive dependence:
    ```@example
    using Copulas: williamson_dist, ClaytonCopula
    williamson_dist(ClaytonGenerator(10), Val{3}())
    ```


```@docs
ArchimedeanCopula
```

<!-- 
TODO: Make a few graphs of bivariate archimedeans pdfs and cdfs. And provide a few more standard tools for these copulas ? 
-->

# Liouville Copulas

!!! todo "Not merged yet !"
    Liouville copulas are coming in this PR : https://github.com/lrnv/Copulas.jl/pull/83, but the work is not finished. 

Archimedean copulas have been widely used in the literature due to their nice decomposition properties and easy parametrization. The interested reader can refer to the extensive literature [hofert2010,hofert2013a,mcneil2010,cossette2017,cossette2018,genest2011a,dibernardino2013a,dibernardino2013a,dibernardino2016,cooray2018,spreeuw2014](@cite) on Archimedean copulas, their nesting extensions and most importantly their estimation. 

One major drawback of the Archimedean family is that these copulas have exchangeable marginals (i.e., $C(\bm u) = C(p(\bm u))$ for any permutation $p(\bm u)$ of $u_1, ..., u_d$): the dependence structure is symmetric, which might not be desirable. However, from the Radial-simplex expression, we can extrapolate and take for $\bm S$ a non-uniform distribution on the simplex. 

Liouville's copulas share many properties with Archimedean copulas, but are not exchangeable anymore. This is an easy way to produce non-exchangeable dependence structures. See [cote2019](@cite) for a practical use of this property.

Note that Dirichlet distributions are constructed as $\bm S = \frac{\bm G}{\langle \bm 1, \bm G \rangle}$, where $\bm G$ is a vector of independent Gamma distributions with unit scale (and potentially different shapes: taking all shapes equal yields the Archimedean case). 




```@bibliography
Pages = [@__FILE__]
Canonical = false
```
