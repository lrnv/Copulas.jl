```@meta
CurrentModule = Copulas
```

# Vines Copulas

!!! todo "Not implemented yet!"
    Do not hesitate to come talk on [our GitHub](https://github.com/lrnv/Copulas.jl) !

One more noticeable class of copulas are the Vines copulas. These distributions use a graph of conditional distributions to encode the distribution of the random vector. To define such a model, working with conditional densities, and given any ordered partition $\bm i_1,...\bm i_p$ of $1,...d$, we write:
 
$$f(\bm x) = f(x_{\bm i_1}) \prod\limits_{j=1}^{p-1} f(x_{\bm i_{j+1}} | x_{\bm i_j}).$$

Of course, the choice of the partition, of its order, and of the conditional models is left to the practitioner. The goal when dealing with such dependency graphs is to tailor the graph to reduce the error of approximation, which can be a tricky task. There exists simplifying assumptions that help with this matter, and we refer to [durante2017a,nagler2016,nagler2018,czado2013,czado2019,graler2014](@cite) for a deep dive into the vine theory, along with some nice results and extensions. 


```@bibliography
Pages = ["Liouville.md"]
Canonical = false
```