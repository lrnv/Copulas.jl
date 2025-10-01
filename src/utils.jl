# Three small utility functions. 

@inline _δ(t) = oftype(t, 1e-12)
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))
_invmono(f; tol=1e-8, θmax=1e6, a=0.0, b=1.0) = begin
    fa,fb = f(0.0), f(1.0)
    while fb ≤ 0 && b < θmax
        b = min(2b, θmax); fb = f(b)
        !isfinite(fb) && (b = θmax; break)
    end
    (fa < 0 && fb > 0) || error("Could not bound root at [0, $θmax].")
    Roots.find_zero(f, (a,b), Roots.Brent(); atol=tol, rtol=tol)
end
