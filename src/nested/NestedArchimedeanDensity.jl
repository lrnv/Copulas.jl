# =============================================================================
# Density kernels for nested Archimedean copulas via Faà di Bruno's formula.
#
# A nested Archimedean copula is a tree of Archimedean generators: an outer
# generator `G` acts on a sum of inverse-generator terms, some of which are bare
# marginal coordinates and some of which are the CDFs of *inner* nested copulas.
# Its density is the mixed partial of that nested CDF over the differentiated
# (observed) coordinates.
#
# Differentiating the composition of generators is exactly Faà di Bruno's
# formula; the bookkeeping is carried by the partial Bell polynomials, here
# expressed through truncated Taylor series. Concretely, for each node we form
# the Taylor expansion of the inner-to-outer change of variables `ϕ⁻¹_outer ∘
# ϕ_inner`, raise it to the appropriate power (the leaf-count of the child),
# convolve the children's contributions (a Cauchy product of the per-child
# Taylor coefficient vectors), and finally contract the resulting coefficient
# vector against the derivatives `ϕ⁽ᵏ⁾` of the outer generator. The same
# recursion, restricted to differentiating the observed coordinates only, yields
# the per-variable censored (survival) likelihood.
#
# Follows the nested-density and per-variable censored-likelihood algorithm of
# Yang & Li, "Archimedean Copula Inference via Taylor-Mode AD," arXiv:2605.23134
# (2026); see the `NestedArchimedeanCopula` docstring.
#
# Everything is written generically in the value type `T`. The default working
# type is `Float64`; passing `BigFloat` (or any high-precision real, e.g.
# `Double64`) coordinates flows that precision through the whole recursion, which
# is the recommended option for adversarial high-dimensional or deep-tail inputs
# where the alternating-sign Faà di Bruno sum can lose Float64 precision.
#
# These helpers build *only* on the public Copulas generator interface
# (`ϕ`, `ϕ⁻¹`, `ϕ⁽ᵏ⁾`, `ϕ⁻¹⁽¹⁾`); they do not introduce a parallel generator
# system. The Taylor route requires `ϕ` (and, for nested nodes, `ϕ⁻¹` of the
# outer generator) to accept a `TaylorSeries.Taylor1` argument; that holds for
# the closed-form generators shipped with the package.
# =============================================================================

# Internal tree representation built from a NestedArchimedeanCopula and a data
# point. A node carries its outer generator, its bare leaves (each a coordinate
# value and a censoring flag), and its child sub-trees.
struct _NestedLeaf{T}
    u::T
    censored::Bool
end
struct _NestedNode{TG<:Generator, T}
    G::TG
    leaves::Vector{_NestedLeaf{T}}
    children::Vector{_NestedNode}
end

# Overloadable hook for the edge composition. Each parent→child edge needs the
# Taylor coefficients [h'(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!] of the change of variables
# h = ϕ⁻¹_outer ∘ ϕ_inner at t₀, the inner-to-outer link in the Faà di Bruno
# recursion. `_process_node` calls THIS hook, so overrides take effect.
#
# The DEFAULT forwards to `_composition_taylor_direct` (jet over the explicit
# composition) — this is the ONLY generic method shipped, so the default
# behaviour is byte-unchanged. Switch the global default to the implicit
# App. A.4 solver by redefining this generic method, e.g.
#   Copulas.composition_taylor(o::Copulas.Generator, i::Copulas.Generator, t₀, d) =
#       Copulas._composition_taylor_implicit(o, i, t₀, d)
# or register a closed form for a specific generator pair by adding a more-
# specific method (see the shipped Clayton/Clayton example below). This mirrors
# the existing per-generator ϕ⁽ᵏ⁾ override idiom: no keyword/flag, pure
# dispatch, most-specific method wins. The working type T flows from t₀ into the
# backend, so BigFloat/Double64 precision is carried through.
composition_taylor(outer::Generator, inner::Generator, t₀, d::Int) =
    _composition_taylor_direct(outer, inner, t₀, d)

# DEFAULT direct method: Taylor-expand the explicit composition ϕ⁻¹_outer ∘ ϕ_inner
# at t₀ to order d via Copulas' generic `taylor` jet primitive. Returns the
# coefficients [h'(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!] (constant h₀ dropped) as a Vector{T}.
function _composition_taylor_direct(outer::Generator, inner::Generator, t₀::T, d::Int) where {T}
    coefs = taylor(x -> ϕ⁻¹(outer, ϕ(inner, x)), t₀, d)
    return T[coefs[k+1] for k in 1:d]
end

# Implicit-equation method (paper App. A.4 / acopula compose.py
# `_solve_composition_taylor`). Instead of differentiating ϕ⁻¹ (ill-conditioned
# high-order inverse derivatives; composed-chain underflow for fast-tail
# generators), use that h satisfies ϕ_outer(h(t)) = ϕ_inner(t). With
# h₀ = ϕ⁻¹_outer(ϕ_inner(t₀)) (a single SCALAR inverse) and
# h(t₀+ε) = h₀ + Q(ε), Q(ε) = Σ qₘ εᵐ, write aₘ = ϕ⁽ᵐ⁾(outer,h₀)/m! and
# bₘ = ϕ⁽ᵐ⁾(inner,t₀)/m!. Matching εᵏ in Σ aₘ Qᵐ = Σ bₘ εᵐ gives the triangular
# solve q₁ = b₁/a₁; qₖ = (bₖ − Σ_{m≥2} aₘ [εᵏ]Qᵐ)/a₁. Uses ONLY the scalar
# k-th derivatives ϕ⁽ᵏ⁾(outer)/ϕ⁽ᵏ⁾(inner) and one scalar ϕ⁻¹(outer); never ϕ
# or ϕ⁻¹ on a Taylor1, never the inverse high-order derivatives. Returns the
# IDENTICAL convention to `_composition_taylor_direct`: Vector{T} of length d,
# element k = h⁽ᵏ⁾(t₀)/k!, keyed off T = typeof(t₀) so BigFloat exactness flows.
function _composition_taylor_implicit(outer::Generator, inner::Generator, t₀::T, d::Int) where {T}
    # Derivative-side coefficients: a[m] = ϕ⁽ᵐ⁾(outer,h₀)/m!, b[m] = ϕ⁽ᵐ⁾(inner,t₀)/m!
    # for m = 1..d (scalar k-th derivatives — NO Taylor1 on ϕ/ϕ⁻¹).
    b  = T[ϕ⁽ᵏ⁾(inner, m, t₀) / T(factorial(big(m))) for m in 1:d]
    h₀ = ϕ⁻¹(outer, ϕ(inner, t₀))                      # single scalar inverse
    a  = T[ϕ⁽ᵏ⁾(outer, m, h₀) / T(factorial(big(m))) for m in 1:d]

    q = zeros(T, d)
    q[1] = b[1] / a[1]
    d == 1 && return q

    # Amortized O(d³) column-update ladder (port of acopula's exact recurrence).
    # Q[j+1] = q_j is the coefficient of εʲ in Q (Q has no constant term, Q[1]=0).
    # C[m+1, j+1] = [εʲ] Q^{m+2}, maintained as Q's coefficients fill in.
    npw = d - 1                                        # number of powers Q²..Q^d
    Q = zeros(T, d + 1); Q[2] = q[1]                   # Q = q₁ε initially
    C = zeros(T, npw, d + 1)
    for m in 0:npw-1
        (m + 2) <= d && (C[m+1, (m+2)+1] = q[1]^(m+2)) # [ε^{m+2}] (q₁ε)^{m+2}
    end
    for k in 2:d
        # correction = Σ_{m=0}^{npw-1} a[m+2] · [εᵏ] Q^{m+2}  (column k of C, final).
        corr = zero(T)
        for m in 0:npw-1
            (m + 2) <= d && (corr += a[m+2] * C[m+1, k+1])
        end
        q[k] = (b[k] - corr) / a[1]
        Q[k+1] = q[k]
        # Fill column j = k+1 of every power: C[m+1, j+1] = Σ_{i=1}^{k} q_i · prevₘ[j-i+1],
        # with prevₘ = Q^{m+1} (Q for m=0, else the lower power row). All reads touch
        # columns ≤ k, already finalised in prior outer steps, so rows update in parallel.
        j = k + 1
        if j <= d
            for m in 0:npw-1
                prev = m == 0 ? Q : view(C, m, :)      # prevₘ = Q^{m+1}
                s = zero(T)
                for i in 1:k
                    ji = j - i
                    (1 <= ji) && (ji + 1 <= d + 1) && (s += Q[i+1] * prev[ji+1])
                end
                C[m+1, j+1] = s
            end
        end
    end
    return q
end

# Worked closed-form override: same-family Clayton/Clayton edge. For the package
# parametrization ϕ_θ(t) = (1+θt)^(-1/θ), ϕ⁻¹_θ(u) = (u^(-θ)-1)/θ, the inner-to-
# outer link h(t) = ϕ⁻¹_outer(ϕ_inner(t)) = ((1+θ_in·t)^r − 1)/θ_out, with
# r = θ_out/θ_in, is a reparametrised power map. Its Taylor coefficients are
# available in closed form, avoiding the inverse high-order derivatives entirely.
# This overrides the default purely by dispatch (same idiom as per-generator
# ϕ⁽ᵏ⁾); `_process_node` already calls the `composition_taylor` hook. NOTE: do
# NOT use the (1+t)^r−1 form — that is the θ=1 special case and is WRONG for this
# generator (θ lives inside ϕ). θ is promoted into T so Float64/BigFloat/Double64
# all stay correct.
function composition_taylor(outer::ClaytonGenerator, inner::ClaytonGenerator, t₀::T, d::Int) where {T}
    θ_out = T(outer.θ)
    θ_in  = T(inner.θ)
    r     = θ_out / θ_in
    base  = 1 + θ_in * t₀                              # expansion base 1 + θ_in·t₀
    h = Vector{T}(undef, d)
    binom = one(T)                                     # generalized binomial C(r,k), incremental
    θ_in_pow = one(T)                                  # θ_in^k
    for k in 1:d
        binom    *= (r - (k - 1)) / k                  # C(r,k) = C(r,k-1)·(r-k+1)/k
        θ_in_pow *= θ_in
        h[k] = (θ_in_pow / θ_out) * binom * base^(r - k)
    end
    return h                                            # [h₁,…,h_d], length d, no constant
end

# Partial-Bell-polynomial step: given the Taylor coefficients `p` of the link
# h (without its constant term) and the inner contribution `β` (the child's own
# Faà di Bruno coefficient vector), return the order-`d` coefficient vector of
# the composed contribution: it combines the derivatives of the composition with
# the inner block's coefficients, with the partial Bell polynomial B_{n,k}
# accumulated implicitly through the running polynomial powers `q`.
function _faa_di_bruno_coeffs(p::AbstractVector{T}, β::AbstractVector{T}, d::Int) where {T}
    n = d + 1
    p_pad = zeros(T, n)
    p_pad[2:min(n, length(p) + 1)] .= p[1:min(d, length(p))]
    facts = T[factorial(big(j)) for j in 0:d]
    invf  = T[1 / factorial(big(j)) for j in 0:d]
    β_scaled = T[β[j] * facts[j] for j in 1:n]
    α = zeros(T, n)
    q1 = copy(p_pad)
    q2 = zeros(T, n)
    for k in 1:d
        s = zero(T)
        for j in 1:n
            s += β_scaled[j] * q1[j]
        end
        α[k + 1] = s * invf[k + 1]
        if k < d
            fill!(q2, zero(T))
            for i in 1:n, j in 1:n
                idx = i + j - 1
                idx <= n && (q2[idx] += q1[i] * p_pad[j])
            end
            q1, q2 = q2, q1
        end
    end
    return α
end

# Cauchy product (polynomial convolution) of the per-child coefficient vectors,
# truncated to order `d`. Combining sibling sub-trees under one parent generator
# multiplies their Faà di Bruno generating polynomials.
function _cauchy_product(coeffs::Vector{Vector{T}}, d::Int) where {T}
    n = d + 1
    isempty(coeffs) && return T[one(T)]
    r1 = zeros(T, n)
    a1 = coeffs[1]
    r1[1:min(n, length(a1))] .= a1[1:min(n, length(a1))]
    r2 = zeros(T, n)
    for i in 2:length(coeffs)
        a = coeffs[i]
        fill!(r2, zero(T))
        for j in 1:n, k in 1:min(length(a), n - j + 1)
            r2[j + k - 1] += r1[j] * a[k]
        end
        r1, r2 = r2, r1
    end
    return r1
end

# Process one node: accumulate its argument t = Σ ϕ⁻¹(leaf/child-CDF), the node
# CDF C = ϕ(G, t), the number `d` of differentiated coordinates in the subtree,
# and its Faà di Bruno coefficient vector `β`. Censored leaves shift `t` but do
# not add a differentiation order. Returns (t, C, d, β).
function _process_node(node::_NestedNode{TG, T}) where {TG, T}
    G = node.G
    psum = zero(T)
    coeffs = Vector{T}[]
    dtot = 0
    for leaf in node.leaves
        psum += ϕ⁻¹(G, leaf.u)
        if leaf.censored
            push!(coeffs, T[one(T)])
        else
            push!(coeffs, T[zero(T), one(T)])
            dtot += 1
        end
    end
    for child in node.children
        (tc, Cc, dc, βc) = _process_node(child)
        psum += ϕ⁻¹(G, Cc)
        if dc == 0
            push!(coeffs, T[one(T)])
        else
            p = composition_taylor(G, child.G, tc, dc)
            β_pad = zeros(T, dc + 1)
            β_pad[1:min(dc + 1, length(βc))] .= βc[1:min(dc + 1, length(βc))]
            push!(coeffs, _faa_di_bruno_coeffs(p, β_pad, dc))
            dtot += dc
        end
    end
    t = psum
    C = ϕ(G, t)
    β = dtot == 0 ? T[one(T)] : _cauchy_product(coeffs, dtot)
    return (t, C, dtot, β)
end

# Log-Jacobian of the marginal change of variables: Σ log|ϕ⁻¹′(u)| over the
# differentiated (observed) leaves, recursively over the tree.
function _leaf_log_jacobian(node::_NestedNode{TG, T}) where {TG, T}
    s = zero(T)
    for leaf in node.leaves
        leaf.censored || (s += log(abs(ϕ⁻¹⁽¹⁾(node.G, leaf.u))))
    end
    for child in node.children
        s += _leaf_log_jacobian(child)
    end
    return s
end

# Contract the root coefficient vector against the outer generator derivatives:
# the density on the generator scale is Σₖ β[k] ϕ⁽ᵏ⁾(G, t). Done directly in the
# working type `T` (no log-space), so a high-precision `T` keeps the alternating
# sum exact.
function _assemble_density(β::Vector{T}, G::Generator, t::T, d::Int) where {T}
    d == 0 && return ϕ(G, t)
    s = zero(T)
    for k in 0:d
        ϕᵏ = k == 0 ? ϕ(G, t) : ϕ⁽ᵏ⁾(G, k, t)
        s += β[k + 1] * ϕᵏ
    end
    return s
end

# Full nested log-density of a node, including the marginal log-Jacobian. With
# no censored leaves this is the ordinary nested-Archimedean log-density; with
# censored leaves it is the mixed partial of the nested CDF over the observed
# coordinates only — the numerator kernel behind the condition/subsetdims
# fast path (see nested/NestedConditioning.jl).
function _nested_logpdf(node::_NestedNode{TG, T}) where {TG, T}
    (t, C, d, β) = _process_node(node)
    return log(abs(_assemble_density(β, node.G, t, d))) + _leaf_log_jacobian(node)
end

# Nested CDF of a node: C(u) = ϕ_G(Σ_leaves ϕ⁻¹(u) + Σ_children ϕ⁻¹(C_child)).
# Closed form, no Faà di Bruno / numerical integration needed.
function _nested_cdf(node::_NestedNode{TG, T}) where {TG, T}
    G = node.G
    s = zero(T)
    for leaf in node.leaves
        s += ϕ⁻¹(G, leaf.u)
    end
    for child in node.children
        s += ϕ⁻¹(G, _nested_cdf(child))
    end
    return ϕ(G, s)
end
