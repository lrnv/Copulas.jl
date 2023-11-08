```@meta
CurrentModule = Copulas
```

# Implement your own 

If you think that the WilliamsonCopula interface is too barebone and does not provide you with enough flexibility in your modeling of an archimedean copula, you might be intersted in the possiiblity to directly subtype `ArchimedeanCopula` and implement your own. This is actually a fairly easy process and you only need to implement a few functions. Let's here together try to reimplement come archimedean copula with the follçowing generator: 

```math
my_generator
```

(describe the process...)

```julia
struct MyAC{d,T} <: ArchimedeanCopula{d}
    par::T
end
ϕ(C::MyAC{d},x)
ϕ⁻¹(C::MyAC{d},x)
τ(C::MyAC{d})
τ⁻¹(::MyAC{d},τ)
williamson_dist(C::MyAC{d})
```