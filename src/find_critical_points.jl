# find_critical_points.jl — robust O-point / X-point finding on the reduced
# quintic Hermite FEM ψ field via a globalized (damped) 2-D Newton, plus the
# M3D-C1 boundary-flux (LCFS) determination built on top of it.
#
# The magnetic axis (O-point) and the separatrix X-point are BOTH critical
# points of ψ (∇ψ=0): they solve the same 2×2 system F(R,Z)=∇ψ=0 with the
# Hessian ∇²ψ as Jacobian, differing only by (a) the initial seed and (b) the
# Hessian discriminant D=ψ_RR·ψ_ZZ−ψ_RZ² at the solution — D>0 ⇒ extremum
# (O-point), D<0 ⇒ saddle (X-point).
#
# `damped_newton_2d` is a field-agnostic solver; `critical_point_system`
# encapsulates the FEM evaluation as the (residual, Jacobian) closure it
# consumes. This mirrors the globalization strategy of IMAS `_damped_newton2d`
# but evaluates the exact FEM polynomial (elements.jl), not a grid interpolant.
#
# `find_lcfs` mirrors M3D-C1 `diagnostics.f90:lcfs`: gather boundary-flux
# candidates — wall nodes, X-point 1/2 saddles, fixed limiter points — and the
# candidate whose ψ is closest to the axis ψ defines the LCFS.

"""
    damped_newton_2d(eqs, R0, Z0; tol=1e-11, maxit=50, αmin=1e-3)
        -> (; R, Z, converged, resnorm, n_iter)

Globalized 2×2 Newton for `F(R,Z)=0`. `eqs(R,Z)` returns
`(F1, F2, J11, J12, J21, J22, ok)`: the residual `F`, the row-major 2×2 Jacobian
`J`, and `ok::Bool` (`false` ⇒ the point is invalid, e.g. off-mesh). Robustness
beyond a plain Newton step:

* **backtracking line search** on `‖F‖` — the full step is halved until it
  strictly decreases the residual norm, guaranteeing monotone progress where a
  pure Newton step would overshoot or diverge (e.g. near the separatrix, where
  `det J → 0` blows up the step);
* **gradient-descent fallback** when `det J` is near-singular — a small
  normalized step along `−Jᵀ F` rather than an exploding `−J⁻¹ F`;
* **domain guard** — a probe returning `ok=false` is rejected by the line
  search, so an iterate can never leave the valid region.

Returns the final point, a convergence flag (`‖F‖ ≤ tol`), the residual norm,
and the iteration count.
"""
function damped_newton_2d(
        eqs, R0::Real, Z0::Real;
        tol::Real = 1.0e-11, maxit::Integer = 50, αmin::Real = 1.0e-3
    )
    R = Float64(R0);  Z = Float64(Z0)
    F1, F2, J11, J12, J21, J22, ok = eqs(R, Z)
    ok || return (; R, Z, converged = false, resnorm = NaN, n_iter = 0)
    nrm = hypot(F1, F2)
    n_iter = 0
    for it in 1:maxit
        n_iter = it
        nrm ≤ tol && return (; R, Z, converged = true, resnorm = nrm, n_iter)
        det = J11 * J22 - J12 * J21
        if isfinite(det) && abs(det) > 1.0e-13          # Newton step −J⁻¹F
            dR = (J22 * F1 - J12 * F2) / det
            dZ = (J11 * F2 - J21 * F1) / det
        else                                          # near-singular J: descend ½‖F‖²
            dR = J11 * F1 + J21 * F2                   # (JᵀF)_R
            dZ = J12 * F1 + J22 * F2                   # (JᵀF)_Z
            s = hypot(dR, dZ);  s > 0 && (dR /= s; dZ /= s)
            dR *= 1.0e-2;  dZ *= 1.0e-2
        end
        α = 1.0;  stepped = false
        while α ≥ αmin
            q1 = R - α * dR;  q2 = Z - α * dZ
            g1, g2, j11, j12, j21, j22, okq = eqs(q1, q2)
            if okq && hypot(g1, g2) < nrm             # accept only a decrease in ‖F‖
                R, Z = q1, q2
                F1, F2, J11, J12, J21, J22 = g1, g2, j11, j12, j21, j22
                nrm = hypot(F1, F2);  stepped = true;  break
            end
            α /= 2
        end
        stepped || return (; R, Z, converged = nrm ≤ tol, resnorm = nrm, n_iter)
    end
    return (; R, Z, converged = nrm ≤ tol, resnorm = nrm, n_iter)
end

"""
    critical_point_system(coef_plane, elems_plane) -> eqs

Return the `eqs(R,Z)` closure for the `∇ψ=0` system on the reduced quintic FEM
field: residual = `∇ψ` and Jacobian = the Hessian `∇²ψ`, both rotated from the
containing element's local `(ξ,η)` frame into global `(R,Z)`, with `ok=false`
when `(R,Z)` is off-mesh. Consumed by [`damped_newton_2d`](@ref) and shared by
the O-point and X-point solves.
"""
function critical_point_system(coef_plane::AbstractMatrix, elems_plane::AbstractMatrix)
    bcol = view(elems_plane, 2, :);  θcol = view(elems_plane, 4, :)
    xcol = view(elems_plane, 5, :);  zcol = view(elems_plane, 6, :)
    return function (R::Real, Z::Real)
        k = locate_element(R, Z, elems_plane)
        k == 0 && return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, false)
        θ = Float64(θcol[k]);  co = cos(θ);  sn = sin(θ)
        ξ, η = global_to_local(
            R, Z, Float64(xcol[k]), Float64(zcol[k]),
            Float64(bcol[k]), θ
        )
        _, pξ, pη, pξξ, pξη, pηη = eval_psi_and_derivs(view(coef_plane, :, k), ξ, η)
        # rotate ∇ψ and ∇²ψ from local (ξ,η) to global (R,Z): H_RZ = Rot·H_ξη·Rotᵀ
        F1 = pξ * co - pη * sn                                  # ∂ψ/∂R
        F2 = pξ * sn + pη * co                                  # ∂ψ/∂Z
        J11 = co * co * pξξ - 2co * sn * pξη + sn * sn * pηη    # ∂²ψ/∂R²
        J22 = sn * sn * pξξ + 2co * sn * pξη + co * co * pηη    # ∂²ψ/∂Z²
        J12 = co * sn * (pξξ - pηη) + (co * co - sn * sn) * pξη # ∂²ψ/∂R∂Z
        return (F1, F2, J11, J12, J12, J22, true)              # symmetric: J21=J12
    end
end

"""
    classify_critical(D; tol=0) -> Symbol

Classify a critical point by the Hessian discriminant `D = ψ_RR·ψ_ZZ − ψ_RZ²`
(a rotation invariant, so it may be evaluated in the local frame):
`:extremum` (`D > tol`, a magnetic axis / O-point), `:saddle` (`D < −tol`, an
X-point), or `:degenerate` (`|D| ≤ tol`).
"""
function classify_critical(D::Real; tol::Real = 0.0)
    D > tol && return :extremum
    D < -tol && return :saddle
    return :degenerate
end

"""
    find_critical_point(coef_plane, elems_plane, R_seed, Z_seed;
                        tol=1e-11, maxit=50, αmin=1e-3)
        -> (; R, Z, ψ, kind, D, converged, grad_norm, n_iter, elem_idx)

Locate the critical point (`∇ψ=0`) of the reduced quintic FEM field near
`(R_seed, Z_seed)` with the globalized [`damped_newton_2d`](@ref), then classify
it via the Hessian discriminant (`kind` = `:extremum` for an O-point, `:saddle`
for an X-point). Seed the magnetic axis from the stored `(xmag, zmag)` and the
X-point from `(xnull, znull)`; both use the same solver and system.

`ψ`, `D`, `kind`, and `elem_idx` are evaluated at the converged point;
`elem_idx=0` (and `kind=:unknown`) means the solve walked off the mesh.
"""
function find_critical_point(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix,
        R_seed::Real, Z_seed::Real;
        tol::Real = 1.0e-11, maxit::Integer = 50, αmin::Real = 1.0e-3
    )
    eqs = critical_point_system(coef_plane, elems_plane)
    sol = damped_newton_2d(eqs, R_seed, Z_seed; tol, maxit, αmin)
    k = locate_element(sol.R, sol.Z, elems_plane)
    ψ = NaN;  D = NaN;  kind = :unknown
    if k > 0
        θ = Float64(elems_plane[4, k])
        ξ, η = global_to_local(
            sol.R, sol.Z, Float64(elems_plane[5, k]),
            Float64(elems_plane[6, k]), Float64(elems_plane[2, k]), θ
        )
        ψ, _, _, pξξ, pξη, pηη = eval_psi_and_derivs(view(coef_plane, :, k), ξ, η)
        D = pξξ * pηη - pξη * pξη                      # det(Hessian) — rotation invariant
        kind = classify_critical(D)
    end
    return (;
        R = sol.R, Z = sol.Z, ψ, kind, D,
        converged = sol.converged, grad_norm = sol.resnorm,
        n_iter = sol.n_iter, elem_idx = k,
    )
end

"""
    find_o_point(coef_plane, elems_plane, R_seed, Z_seed; kwargs...)

Find the magnetic axis (O-point): [`find_critical_point`](@ref) seeded at the
stored axis `(xmag, zmag)`. `kind` is expected to be `:extremum`; a `@warn` is
issued if a `:saddle` is reached instead (a sign of a poor seed).
"""
function find_o_point(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix,
        R_seed::Real, Z_seed::Real; kwargs...
    )
    r = find_critical_point(coef_plane, elems_plane, R_seed, Z_seed; kwargs...)
    (r.converged && r.kind === :saddle) &&
        @warn "find_o_point converged to a saddle, not an extremum — check the (xmag,zmag) seed" R = r.R Z = r.Z
    return r
end

"""
    find_x_point(coef_plane, elems_plane, R_seed, Z_seed; kwargs...)

Find the separatrix X-point: [`find_critical_point`](@ref) seeded at the stored
`(xnull, znull)`. `kind` is expected to be `:saddle`; a `@warn` is issued if an
`:extremum` is reached instead (a sign of a poor seed).
"""
function find_x_point(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix,
        R_seed::Real, Z_seed::Real; kwargs...
    )
    r = find_critical_point(coef_plane, elems_plane, R_seed, Z_seed; kwargs...)
    (r.converged && r.kind === :extremum) &&
        @warn "find_x_point converged to an extremum, not a saddle — check the (xnull,znull) seed" R = r.R Z = r.Z
    return r
end

# M3D-C1 null-seed convention: xnull ≤ 0 (or missing → NaN) means "not tracked".
_null_active(x::Real, z::Real) = isfinite(x) && isfinite(z) && x > 0

"""
    find_lcfs(coef_plane, elems_plane; xmag, zmag,
              xnull=NaN, znull=NaN, xnull2=NaN, znull2=NaN,
              xlim=NaN, zlim=NaN, xlim2=NaN, zlim2=NaN,
              wall_rz=nothing, kwargs...)
        -> (; psi_axis, psi_bound, is_diverted, limited_by,
             axis, x1, x2, psib, psilim, psilim2)

Boundary-flux (LCFS) determination, mirroring M3D-C1 `diagnostics.f90:lcfs`:
find the magnetic axis (`ψ0`), gather boundary candidates, and pick the one
whose ψ is **closest to ψ0** (the innermost limiting surface):

1. **wall** — over caller-provided `wall_rz` (iterable of `(R, Z)` first-wall
   points; M3D-C1 scans its FIRSTWALL nodes when `iwall_is_limiter=1`), keeping
   nodes that pass both M3D-C1 filters: ψ locally increasing away from the axis
   (`((R−R₀)ψ_R + (Z−Z₀)ψ_Z)(ψ−ψ0) ≥ 0`, excludes private-flux nodes) and
   `Z·Z₀ ≥ 0` (same vertical half as the axis). Best survivor → `psib`.
2. **X-points 1, 2** — [`find_x_point`](@ref) seeded at `(xnull, znull)` /
   `(xnull2, znull2)` (skipped unless the M3D-C1 "active" convention `x > 0`
   holds). A winning X-point sets `is_diverted = true`. Stricter than the
   Fortran: a candidate must be a converged, genuine `:saddle`.
3. **limiter points 1, 2** — ψ evaluated **at** the fixed input points
   `(xlim, zlim)` / `(xlim2, zlim2)` (from [`limiter_points`](@ref); skipped
   when `xlim == 0`, M3D-C1's "no limiter", or off-mesh). A winning limiter
   sets `is_diverted = false` (plasma limited, not diverted).

`limited_by ∈ (:wall, :xpoint1, :xpoint2, :limiter1, :limiter2, :none)` names
the winner; `axis`/`x1`/`x2` carry the refined critical-point solves (`x1`/`x2`
are `nothing` when not seeded). Pass the toroidally-averaged coefficients for
an n=0 reference (M3D-C1 itself evaluates at φ=0, giving its plane-1 scalars),
and the *total* ψ (M3D-C1 adds the coil field when `icsubtract=1`). `kwargs`
(`tol`, `maxit`, `αmin`) forward to the Newton solves.

    find_lcfs(file::M3DC1File, ts; wall_rz=nothing, kwargs...)

Convenience method: reads slice `ts`, toroidally averages ψ, pulls the axis /
X-point seeds from the slice scalars and the limiter points from the file
attributes, then calls the low-level method above.
"""
function find_lcfs(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix;
        xmag::Real, zmag::Real,
        xnull::Real = NaN, znull::Real = NaN,
        xnull2::Real = NaN, znull2::Real = NaN,
        xlim::Real = NaN, zlim::Real = NaN,
        xlim2::Real = NaN, zlim2::Real = NaN,
        wall_rz = nothing, kwargs...
    )
    o = find_o_point(coef_plane, elems_plane, xmag, zmag; kwargs...)
    (o.converged && o.kind === :extremum) ||
        error("find_lcfs: magnetic-axis solve failed (converged=$(o.converged), kind=$(o.kind)) — check (xmag, zmag)")
    ψ0 = o.ψ

    # ordered boundary candidates: (ψ, tag, sets is_diverted)
    cands = Tuple{Float64, Symbol, Bool}[]

    # 1) wall scan (M3D-C1: FIRSTWALL nodes, iwall_is_limiter=1)
    psib = NaN
    if wall_rz !== nothing
        for (Rw, Zw) in wall_rz
            k = locate_element(Rw, Zw, elems_plane)
            k == 0 && continue
            θ = Float64(elems_plane[4, k]);  co = cos(θ);  sn = sin(θ)
            ξ, η = global_to_local(
                Rw, Zw, Float64(elems_plane[5, k]),
                Float64(elems_plane[6, k]), Float64(elems_plane[2, k]), θ
            )
            ψw, pξ, pη, _, _, _ = eval_psi_and_derivs(view(coef_plane, :, k), ξ, η)
            ψR = pξ * co - pη * sn;  ψZ = pξ * sn + pη * co
            # private-flux filter: ψ must increase away from the axis
            ((Rw - o.R) * ψR + (Zw - o.Z) * ψZ) * (ψw - ψ0) < 0 && continue
            Zw * o.Z < 0 && continue          # same vertical half as the axis
            (isnan(psib) || abs(ψw - ψ0) < abs(psib - ψ0)) && (psib = ψw)
        end
        isfinite(psib) && push!(cands, (psib, :wall, false))
    end

    # 2) X-point candidates (must be converged saddles)
    x1 = _null_active(xnull, znull) ?
        find_x_point(coef_plane, elems_plane, xnull, znull; kwargs...) : nothing
    x2 = _null_active(xnull2, znull2) ?
        find_x_point(coef_plane, elems_plane, xnull2, znull2; kwargs...) : nothing
    (x1 !== nothing && x1.converged && x1.kind === :saddle) &&
        push!(cands, (x1.ψ, :xpoint1, true))
    (x2 !== nothing && x2.converged && x2.kind === :saddle) &&
        push!(cands, (x2.ψ, :xpoint2, true))

    # 3) fixed limiter points (ψ evaluated at the given (R, Z); xlim == 0 → none)
    psilim = NaN;  psilim2 = NaN
    if isfinite(xlim) && xlim != 0
        psilim = eval_axisym_at(coef_plane, elems_plane, xlim, zlim)
        isfinite(psilim) && push!(cands, (psilim, :limiter1, false))
        if isfinite(xlim2) && xlim2 > 0
            psilim2 = eval_axisym_at(coef_plane, elems_plane, xlim2, zlim2)
            isfinite(psilim2) && push!(cands, (psilim2, :limiter2, false))
        end
    end

    # selection: closest |ψ − ψ0| wins (strict <, in the M3D-C1 candidate order)
    psi_bound = NaN;  limited_by = :none;  is_diverted = false
    best = Inf
    for (ψc, tag, div) in cands
        d = abs(ψc - ψ0)
        if d < best
            best = d;  psi_bound = ψc;  limited_by = tag;  is_diverted = div
        end
    end

    return (;
        psi_axis = ψ0, psi_bound, is_diverted, limited_by,
        axis = o, x1, x2, psib, psilim, psilim2,
    )
end

function find_lcfs(file::M3DC1File, ts::Integer; wall_rz = nothing, kwargs...)
    sl = read_timeslice(file, ts; fields = (:psi,))
    coef = average_toroidal_axisymmetric(sl.fields[:psi], file.nplanes)
    lim = limiter_points(file)
    return find_lcfs(
        coef, elems_plane(file);
        xmag = sl.xmag, zmag = sl.zmag,
        xnull = sl.xnull, znull = sl.znull,
        xnull2 = sl.xnull2, znull2 = sl.znull2,
        lim.xlim, lim.zlim, lim.xlim2, lim.zlim2,
        wall_rz, kwargs...
    )
end

"""
    trace_lcfs(coef_plane, elems_plane, axis, psi_axis, psi_bound;
               ntheta=257, x_points=(), nsamp=512,
               id_map=nothing, R_grid=nothing, Z_grid=nothing) -> (; R, Z, nmiss)

Ordered outline of the **last closed flux surface** — the ψ = `psi_bound`
contour that encloses the magnetic axis — for the IMAS
`equilibrium…boundary.outline`. Returns the closed polygon (first point
repeated last) in mesh units, plus `nmiss`, the number of angles that could not
be resolved. `nothing` when the surface cannot be traced at all.

Rather than contouring a sampled ψ map, this casts `ntheta` rays from `axis`
and bisects the **exact FEM ψ** along each one, so the result is independent of
any evaluation grid. That works because the closed LCFS is star-shaped about
the magnetic axis: every ray from an interior point crosses it exactly once, and
that first crossing is by construction the confined-region boundary — so the
private-flux region and the divertor legs (which a naive contour of the ψ map
picks up, since they share ψ = ψ_bound) are excluded without any masking. A ray
aimed at an X-point stops there: ψ_N rises to 1 at the saddle and falls again
beyond it.

`x_points` (mesh units) are snapped into the outline at their own angle when
they sit on the boundary flux, which restores the corner that a finite `ntheta`
would otherwise round off. Angles whose ray leaves the mesh before reaching
`psi_bound` are dropped (counted in `nmiss`) — the polygon then carries a chord
across the gap.

Pass a [`build_grid_to_element_map`](@ref) `id_map` with its `R_grid`/`Z_grid`
to make the element lookup O(1): each ray point takes its containing element
from the surrounding grid cells (falling back to the previous point's element,
then to the O(nelms) [`locate_element`](@ref) scan). Without it the trace costs
one linear element scan per sample — ~20 s/slice on an 11k-element mesh.
"""
function trace_lcfs(
        coef_plane::AbstractMatrix, elems_plane::AbstractMatrix,
        axis, psi_axis::Real, psi_bound::Real;
        ntheta::Integer = 257, x_points = NTuple{2, Float64}[],
        nsamp::Integer = 512, rmax::Real = 0.0,
        id_map::Union{Nothing, AbstractMatrix{<:Integer}} = nothing,
        R_grid = nothing, Z_grid = nothing
    )
    ψ0 = Float64(psi_axis);  ψ1 = Float64(psi_bound)
    span = ψ1 - ψ0
    (isfinite(span) && span != 0) || return nothing
    R0 = Float64(axis[1]);  Z0 = Float64(axis[2])
    (isfinite(R0) && isfinite(Z0)) || return nothing

    # march no further than the mesh bounding box can reach from the axis
    rlo, rhi = extrema(view(elems_plane, 5, :))
    zlo, zhi = extrema(view(elems_plane, 6, :))
    rspan = rmax > 0 ? Float64(rmax) :
        hypot(max(rhi - R0, R0 - rlo), max(zhi - Z0, Z0 - zlo)) * 1.05
    rspan > 0 || return nothing
    h = rspan / nsamp

    # --- O(1) element lookup: grid-map candidates, then the previous element,
    # then the linear scan. `klast` carries the spatial coherence along a ray.
    ea = view(elems_plane, 1, :);  eb = view(elems_plane, 2, :)
    ec = view(elems_plane, 3, :);  eθ = view(elems_plane, 4, :)
    ex = view(elems_plane, 5, :);  ez = view(elems_plane, 6, :)
    @inline function _in(R, Z, k)
        ξ, η = global_to_local(R, Z, ex[k], ez[k], eb[k], eθ[k])
        return is_in_element_local(ξ, η, ea[k], eb[k], ec[k]), ξ, η
    end
    hR = (R_grid === nothing || length(R_grid) < 2) ? 0.0 : Float64(R_grid[2] - R_grid[1])
    hZ = (Z_grid === nothing || length(Z_grid) < 2) ? 0.0 : Float64(Z_grid[2] - Z_grid[1])
    # a degenerate grid (zero span) would index with Inf — fall back to the scan
    use_map = id_map !== nothing && hR > 0 && hZ > 0 &&
        size(id_map) == (length(R_grid), length(Z_grid))
    klast = Ref(0)
    function ψ_at(R, Z)
        k = 0
        if use_map
            i0 = clamp(floor(Int, (R - R_grid[1]) / hR) + 1, 1, length(R_grid))
            j0 = clamp(floor(Int, (Z - Z_grid[1]) / hZ) + 1, 1, length(Z_grid))
            @inbounds for j in j0:min(j0 + 1, length(Z_grid)), i in i0:min(i0 + 1, length(R_grid))
                kc = id_map[i, j]
                kc == 0 && continue
                ok, ξ, η = _in(R, Z, kc)
                ok && (k = kc; break)
            end
        end
        if k == 0 && klast[] != 0
            ok, _, _ = _in(R, Z, klast[])
            ok && (k = klast[])
        end
        k == 0 && (k = locate_element(R, Z, elems_plane))
        k == 0 && return NaN
        klast[] = k
        _, ξ, η = _in(R, Z, k)
        return eval_axisym_at_local(view(coef_plane, :, k), ξ, η)
    end

    ψn(r, θ) = (ψ_at(R0 + r * cos(θ), Z0 + r * sin(θ)) - ψ0) / span

    Rout = Float64[];  Zout = Float64[];  Θ = Float64[]
    nmiss = 0
    r_prev = 0.0                                # previous ray's crossing radius
    for k in 1:(ntheta - 1)                     # θ = -π … π, endpoint excluded
        θ = -π + 2π * (k - 1) / (ntheta - 1)
        klast[] = 0                             # new ray: restart the hint at the axis
        # The boundary radius varies smoothly with θ, so skip the march over the
        # core: any radius where ψ_N < 1 lies between the axis and the first
        # crossing, and is therefore a valid start. Verified before use — if the
        # surface pinched in since the previous angle, fall back to marching
        # from the axis. (Cuts the outward march ~6×, which dominates the cost.)
        m0 = 1;  ra = 0.0
        if r_prev > 0
            r_try = 0.7 * r_prev
            f = ψn(r_try, θ)
            if isfinite(f) && f < 1.0
                ra = r_try;  m0 = max(1, ceil(Int, r_try / h) + 1)
            else
                klast[] = 0
            end
        end
        # first bracket where ψ_N crosses 1 going outward
        hit = false;  rb = 0.0
        for m in m0:nsamp
            r = m * h
            f = ψn(r, θ)
            isfinite(f) || break                # ray left the mesh
            if f >= 1.0
                rb = r;  hit = true;  break
            end
            ra = r
        end
        if !hit
            nmiss += 1;  r_prev = 0.0
            continue
        end
        # bisect (ψ_N is monotone across the bracket for a nested surface)
        lo, hi = ra, rb
        for _ in 1:60
            mid = 0.5 * (lo + hi)
            f = ψn(mid, θ)
            isfinite(f) || break
            f < 1.0 ? (lo = mid) : (hi = mid)
            hi - lo <= 1.0e-12 * rspan && break
        end
        r = 0.5 * (lo + hi)
        r_prev = r
        push!(Rout, R0 + r * cos(θ));  push!(Zout, Z0 + r * sin(θ));  push!(Θ, θ)
    end
    length(Rout) >= 8 || return nothing

    # snap boundary X-points in at their own angle (sharpens the corner)
    dθ = 2π / (ntheta - 1)
    for xp in x_points
        xr = Float64(xp[1]);  xz = Float64(xp[2])
        (isfinite(xr) && isfinite(xz)) || continue
        ψx = eval_axisym_at(coef_plane, elems_plane, xr, xz)
        (isfinite(ψx) && abs(ψx - ψ1) <= 1.0e-3 * abs(span)) || continue
        θx = atan(xz - Z0, xr - R0)
        j = findfirst(t -> abs(rem(t - θx, 2π, RoundNearest)) <= 0.5 * dθ, Θ)
        if j === nothing
            j = searchsortedfirst(Θ, θx)
            insert!(Rout, j, xr);  insert!(Zout, j, xz);  insert!(Θ, j, θx)
        else
            Rout[j] = xr;  Zout[j] = xz;  Θ[j] = θx
        end
    end

    push!(Rout, Rout[1]);  push!(Zout, Zout[1])      # close the polygon
    return (; R = Rout, Z = Zout, nmiss = nmiss)
end
