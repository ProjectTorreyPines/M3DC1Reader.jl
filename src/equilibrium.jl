# equilibrium.jl — experimental: a custom M3D-aligned axisymmetric field object
# (single source of truth) + opt-in adapters to MXHEquilibrium / EFIT / IMAS.
# The existing export_imas pipeline is independent and unaffected.

"""
    M3DAxisymField{C}

The n=0 (toroidally-averaged) axisymmetric field of one M3D-C1 slice, in SI units:
2D ψ(R,Z) map, 1D ψ grid, F(ψ)=R·Bφ, optional pressure `p1d` and safety factor
`q1d` (FSA identity, per-radian; endpoints NaN), magnetic axis, recomputed
X-points, ψ axis/boundary, R0/B0, slice `time`. This is the axisymmetric *field
snapshot*, not a guaranteed force-balance equilibrium — its flux surfaces are
physically equilibrium-like only when nested/closed (quasi-static slices).

The **type parameter `C` is the COCOS index** (see `docs/cocos_conventions.md`):
`M3DAxisymField{1}` is the native per-radian field `axisym_field` constructs
(per-radian ψ, B = +∇ψ×∇φ, q = +dΦ/(2π dψ) — all stored quantities are
COCOS-3-consistent), `M3DAxisymField{11}` an IMAS-convention copy from
[`to_cocos`](@ref). The convention is therefore visible in `typeof(eq)` and
dispatchable; `eq.cocos` still reads `C` for compatibility. Construct with the
keyword constructor (`M3DAxisymField(; cocos = 3, ...)`). Expressed in
equilibrium formats by the `to_mxh` / `to_geqdsk` / `to_imas` adapters in the
package extensions — the adapters require a native per-radian field
(`C` ∈ 1–8; the label steers their sign conventions, e.g. `cocos = 5` makes
the MXH/gEQDSK outputs report the σ_ρθφ-flipped, positive-q-for-this-device
view).
"""
struct M3DAxisymField{C}
    time::Float64
    R::AbstractVector{Float64}
    Z::AbstractVector{Float64}
    psi_rz::Matrix{Float64}
    psi1d::Vector{Float64}
    F1d::Vector{Float64}
    p1d::Union{Nothing, Vector{Float64}}
    q1d::Union{Nothing, Vector{Float64}}
    axis::NTuple{2, Float64}
    psi_axis::Float64
    psi_boundary::Float64
    r0::Float64
    b0::Float64
    x_points::Vector{NTuple{2, Float64}}
    function M3DAxisymField{C}(
            time, R, Z, psi_rz, psi1d, F1d, p1d, q1d, axis,
            psi_axis, psi_boundary, r0, b0, x_points
        ) where {C}
        C isa Int ||
            throw(ArgumentError("M3DAxisymField type parameter must be an Int COCOS index, got $(repr(C))"))
        return new{C}(
            time, R, Z, psi_rz, psi1d, F1d, p1d, q1d, axis,
            psi_axis, psi_boundary, r0, b0, x_points
        )
    end
end

function M3DAxisymField(;
        cocos::Integer = 3, time::Real, R, Z, psi_rz, psi1d, F1d,
        p1d = nothing, q1d = nothing, axis, psi_axis::Real,
        psi_boundary::Real, r0::Real, b0::Real,
        x_points = NTuple{2, Float64}[]
    )
    return M3DAxisymField{Int(cocos)}(
        time, R, Z, psi_rz, psi1d, F1d, p1d, q1d,
        axis, psi_axis, psi_boundary, r0, b0, x_points
    )
end

# `eq.cocos` keeps working (reads the type parameter); everything else is a field
Base.getproperty(eq::M3DAxisymField{C}, s::Symbol) where {C} =
    s === :cocos ? C : getfield(eq, s)
Base.propertynames(::M3DAxisymField) = (:cocos, fieldnames(M3DAxisymField)...)

# the adapters consume the native per-radian field; a COCOS-11 copy would
# silently mis-scale/mislabel their output — fail loudly instead
_assert_native(eq::M3DAxisymField{C}) where {C} = 1 <= C <= 8 ? nothing :
    throw(
        ArgumentError(
            "this adapter expects the native per-radian M3DAxisymField (cocos 1-8), " *
            "got M3DAxisymField{$C}; adapt the native field first, convert afterwards"
        )
    )

"""
    axisym_field(file, ts; ngrid=200, nbins=128, adj=:linear, cocos=3) -> M3DAxisymField

Assemble one slice's n=0 axisymmetric field: reuse `reduce_axisym_slice` for the 2D ψ
map, 1D ψ grid, magnetic axis and ψ axis/boundary, and add F(ψ)=R·Bφ from the `:I`
field. SI units throughout. `cocos` is the M3D-C1 COCOS (validate empirically).

The axis, `psi_axis`, and `psi_boundary` are recomputed per slice on the n=0
averaged ψ by the [`find_lcfs`](@ref) machinery inside `reduce_axisym_slice` (O-point
Newton + X-point/limiter boundary candidates, seeded from the stored plane-1
scalars and the file's limiter attributes), so every downstream conversion
(`to_mxh` / `to_geqdsk` / `to_imas`) inherits a self-consistent ψ_N/ρ_pol.
"""
function axisym_field(
        file::M3DC1File, ts::Integer;
        ngrid::Integer = 200, nbins::Integer = 128,
        adj::Symbol = :linear, cocos::Integer = 3
    )
    # cocos=3 is what the stored data actually is: per-radian ψ (e_Bp=0) with
    # **σ_Bp = −1**. M3D-C1 writes B = +∇ψ×∇φ + F∇φ (B_R = −ψ_Z/R,
    # B_Z = +ψ_R/R), while COCOS puts σ_Bp in front of the reversed cross
    # product ∇φ×∇ψ — so the plus sign there is a MINUS σ_Bp here. Operationally
    # sign(ψ_edge − ψ_axis) = −sign(Ip) on every run, and M3D-C1's
    # `toroidal_current` is the physical current (it agrees with Ampère applied
    # to the ψ map; it is `jphi` — which is +Δ*ψ, not a current density — that
    # runs opposite). OMAS `identify_cocos` on the native (b0, Ip, q, ψ) returns
    # [4, 14, 3, 13] for the JET / DIII-D / ITER cases alike; σ_RφZ = +1 (M3D-C1
    # is right-handed, x = R cos φ) picks 3 over 4, and the raw
    # q1d = +dΦ/(2π dψ) < 0 for Ip·B0 > 0 picks 3 over 7.
    # Use `to_cocos(eq, 11)` for an IMAS-convention copy — that is a REAL
    # transform (ψ ×(−2π), q ×(−1)), not a pure rescale.
    norm = normalization(file)
    ep = elems_plane(file)
    Rg_n = collect(range(extrema(ep[5, :])..., length = ngrid))
    Zg_n = collect(range(extrema(ep[6, :])..., length = ngrid))
    id_map = build_grid_to_element_map(Rg_n, Zg_n, ep)
    utime = unit_factor(norm, :time; system = :si)

    sl = read_timeslice(file, ts; fields = (:psi,))
    lim = limiter_points(file)
    fields = Dict{Symbol, Matrix{Float64}}(:psi => sl.fields[:psi])
    Icoef = _try_field(file, ts, :I)                # F(ψ) + q via reduce_axisym_slice
    Icoef === nothing || (fields[:I] = Icoef)
    s = reduce_axisym_slice(
        fields, ep, file.nplanes, norm, sl.psi_axis, sl.psi_lcfs,
        sl.xmag, sl.zmag, sl.time * utime, Rg_n, Zg_n, id_map;
        nbins = nbins, adj = adj,
        sl.xnull, sl.znull, sl.xnull2, sl.znull2,
        lim.xlim, lim.zlim, lim.xlim2, lim.zlim2
    )
    F1d = s.F1d === nothing ? fill(NaN, length(s.rho)) : collect(Float64, s.F1d)

    R = collect(Float64, s.Rg); Z = collect(Float64, s.Zg)
    r0 = b0 = NaN
    try
        h5open(file.path, "r") do f
            haskey(attrs(f), "rzero") && (r0 = Float64(read_attribute(f, "rzero")) * unit_factor(norm, :length; system = :si))
            haskey(attrs(f), "bzero") && (b0 = Float64(read_attribute(f, "bzero")) * unit_factor(norm, :magnetic_field; system = :si))
        end
    catch err
        @debug "axisym_field: vacuum field attrs unavailable" exception = err
    end

    return M3DAxisymField(;
        cocos = Int(cocos), time = Float64(s.time), R = R, Z = Z,
        psi_rz = Array{Float64}(s.psi_rz), psi1d = collect(Float64, s.psi1d),
        F1d = F1d, p1d = s.pressure === nothing ? nothing : collect(Float64, s.pressure),
        q1d = s.q1d === nothing ? nothing : collect(Float64, s.q1d),
        axis = (Float64(s.R_axis), Float64(s.Z_axis)),
        psi_axis = Float64(s.psi_axis), psi_boundary = Float64(s.psi_boundary),
        r0 = r0, b0 = b0, x_points = s.x_points
    )
end

# Fill NaN entries of a 2D field by averaging finite 4-neighbours, repeated over
# multiple passes (grid dilation) until every reachable NaN is filled. Off-mesh
# points (outside the plasma mesh, left NaN by interpolate_axisym_to_grid) get a
# smoothed value spreading in from the in-mesh region, so a downstream 2D spline
# stays smooth across the plasma boundary and the flux-surface tracers in both
# adapters can close outer surfaces. If the array is all finite it returns a copy
# unchanged. Shared by the to_mxh / to_imas extensions; pure array logic (no deps).
function _fill_nan_dilate(A::AbstractMatrix)
    B = Array{Float64}(A)
    any(isnan, B) || return B
    nr, nc = size(B)
    nanmask = isnan.(B)
    while any(nanmask)
        updated = false
        newvals = copy(B)
        @inbounds for j in 1:nc, i in 1:nr
            nanmask[i, j] || continue
            s = 0.0; n = 0
            if i > 1  && !isnan(B[i - 1, j])
                s += B[i - 1, j]; n += 1
            end
            if i < nr && !isnan(B[i + 1, j])
                s += B[i + 1, j]; n += 1
            end
            if j > 1  && !isnan(B[i, j - 1])
                s += B[i, j - 1]; n += 1
            end
            if j < nc && !isnan(B[i, j + 1])
                s += B[i, j + 1]; n += 1
            end
            if n > 0
                newvals[i, j] = s / n
                nanmask[i, j] = false
                updated = true
            end
        end
        B = newvals
        updated || break   # no finite neighbours anywhere (all-NaN) → stop
    end
    return B
end

# Linear resample of y(x) onto xq (x assumed sorted, possibly non-uniform).
# Shared by the to_mxh / to_geqdsk extensions; pure array logic (no deps).
function _resample(x, y, xq)
    xv = collect(Float64, x); yv = collect(Float64, y)
    return [_lininterp(xv, yv, Float64(qi)) for qi in xq]
end
function _lininterp(x, y, q)
    q <= x[1] && return y[1]
    q >= x[end] && return y[end]
    k = searchsortedlast(x, q)
    t = (q - x[k]) / (x[k + 1] - x[k])
    return (1 - t) * y[k] + t * y[k + 1]
end

"""
    to_cocos(eq::M3DAxisymField, cocos_out::Integer) -> M3DAxisymField{cocos_out}

Re-express the field in another COCOS. Currently only `3 → 11` (native →
IMAS standard) is implemented. **M3D-C1's native data is COCOS 3, not 1**:
it writes `B = +∇ψ×∇φ + F∇φ` (B_R = −ψ_Z/R, B_Z = +ψ_R/R), while COCOS carries
σ_Bp on the reversed cross product ∇φ×∇ψ, so σ_Bp = **−1**. The transform is
therefore `cocos_transform(3, 11)` applied verbatim:

| quantity | factor |
|---|---|
| `psi_rz`, `psi1d`, `psi_axis`, `psi_boundary` | **×(−2π)** |
| `q1d` | **×(−1)** |
| `F1d`, `p1d`, Φ, ρ_tor, axis, X-points, geometry | ×(+1) |

Getting only one of the first two would land on COCOS 5/15 (ψ alone) or
7/17 (q alone), not 11. The convention lands in the **type parameter**
(`M3DAxisymField{11}`), so converted and native fields are distinguishable at
the type level. The returned object is a thin re-wrap: only the ψ and q arrays
are newly allocated; every other array is **shared** with the input.

`b0` is returned with its sign aligned to `F1d` (the C1.h5 `bzero` attribute
is an unsigned magnitude; COCOS-11 requires the signed vacuum field). `Ip` is
never touched by a COCOS transform — IP is invariant for 3 → 11.

The adapters (`to_mxh`/`to_geqdsk`/`to_imas`) expect a native per-radian
field — convert for hand-off or recording, not before adapting.
"""
function to_cocos(eq::M3DAxisymField, cocos_out::Integer)
    cocos_out == eq.cocos && return eq
    cocos_out == 11 ||
        throw(ArgumentError("to_cocos: only COCOS-11 output is implemented (got $cocos_out)"))
    eq.cocos == 3 ||
        throw(
        ArgumentError(
            "to_cocos: only the native COCOS-3 field has an implemented 3→11 " *
                "transform; got cocos $(eq.cocos)"
        )
    )
    fs = filter(isfinite, eq.F1d)
    b0 = isempty(fs) ? eq.b0 : sign(last(fs)) * abs(eq.b0)   # edge F ≈ vacuum field
    return M3DAxisymField(;
        cocos = 11, time = eq.time, R = eq.R, Z = eq.Z,
        psi_rz = eq.psi_rz .* (-2π), psi1d = eq.psi1d .* (-2π),
        F1d = eq.F1d, p1d = eq.p1d,
        q1d = eq.q1d === nothing ? nothing : -1.0 .* eq.q1d,
        axis = eq.axis, psi_axis = eq.psi_axis * (-2π),
        psi_boundary = eq.psi_boundary * (-2π),
        r0 = eq.r0, b0 = b0, x_points = eq.x_points
    )
end

# adapter generics — methods live in ext/M3DC1Reader*Ext.jl (loaded on demand)
function to_mxh end
function to_geqdsk end
function to_imas end

"""
    fsa_imas(eq::M3DAxisymField, field_rz, R, Z; wall_r=Float64[], wall_z=Float64[])
        -> (rho, avg)

Independent flux-surface average of a 2D field map via IMAS's contour tracer, for
cross-checking the in-package `:bin` / `:cumulative` estimators
([`reduce_axisym_slice`](@ref)). `field_rz` is the field sampled on the same
`(R, Z)` grid as `eq.psi_rz`. Traces the flux surfaces of `eq` with
`IMAS.trace_surfaces` (one per `eq.psi1d` node), bilinearly samples `field_rz`
along each surface, and returns `IMAS.flux_surface_avg` at the ρ_pol = √ψ_N
nodes. Only the interior band `clip = (ψN_lo, ψN_hi)` is traced — the tracer
aborts on the axis-degenerate innermost surfaces and the (open) separatrix, and
one failure aborts the whole call — so clipped nodes come back `NaN`. Requires
the IMAS extension (IMASdd + IMAS loaded); errors otherwise.
"""
function fsa_imas end
