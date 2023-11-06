```@meta
CurrentModule = Copulas
```

# Implement your own 

Explain how easy it is to implement your own archimedean copulas and work with them. methods: 

```julia
struct MyAC{d,T} <: ArchimedeanCopula{d}
    par::T
end
ϕ(C::MyAC{d},x)
ϕ⁻¹(C::MyAC{d},x)
τ(C::MyAC{d})
τ⁻¹(::MyAC{d},τ)
radial_dist(C::MyAC{d})
```
