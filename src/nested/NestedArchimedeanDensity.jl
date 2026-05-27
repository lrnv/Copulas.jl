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

# Family-agnostic k-th derivative of the generator, ϕ⁽ᵏ⁾(G, t), obtained from the
# Taylor expansion of ϕ rather than a family-specialised closed form: the k-th
# Taylor coefficient of ϕ(G, t + s) in s, times k!. This avoids the integer
# Stirling-number overflow that caps some closed-form ϕ⁽ᵏ⁾ at moderate orders and
# keeps the whole recursion in the working type `T`.
function _phi_deriv(G::Generator, k::Int, t::T) where {T}
    k == 0 && return ϕ(G, t)
    return TaylorSeries.getcoeff(ϕ(G, t + TaylorSeries.Taylor1(T, k)), k) * T(factorial(big(k)))
end

# Taylor coefficients [h'(t₀)/1!, …, h⁽ᵈ⁾(t₀)/d!] of the change of variables
# h = ϕ⁻¹_outer ∘ ϕ_inner at t₀, to order d. This is the inner-to-outer link in
# the Faà di Bruno recursion for a child sub-tree.
function _composition_taylor(outer::Generator, inner::Generator, t₀::T, d::Int) where {T}
    h(x) = ϕ⁻¹(outer, ϕ(inner, x))
    ht = h(t₀ + TaylorSeries.Taylor1(T, d))
    return T[TaylorSeries.getcoeff(ht, k) for k in 1:d]
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
            p = _composition_taylor(G, child.G, tc, dc)
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
        s += β[k + 1] * _phi_deriv(G, k, t)
    end
    return s
end

# Full nested log-density of a node, including the marginal log-Jacobian. With
# no censored leaves this is the ordinary nested-Archimedean log-density; with
# censored leaves it is the mixed partial of the nested CDF over the observed
# coordinates only (the survival likelihood). See `censored_logpdf`.
function _nested_logpdf(node::_NestedNode{TG, T}) where {TG, T}
    (t, C, d, β) = _process_node(node)
    return log(abs(_assemble_density(β, node.G, t, d))) + _leaf_log_jacobian(node)
end
