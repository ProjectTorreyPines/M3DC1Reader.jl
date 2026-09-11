# export_imas.jl — orchestrate the FSA reduction over all slices and assemble
# an OMAS-path IR dict for write_omas_h5.

using Printf: @sprintf

_maybe!(ir, k, v) = (v !== nothing && (ir[k] = v); nothing)

# shared per-IDS boilerplate (properties + time base)
function _ids_props!(ir, ids::AbstractString, meta, times)
    ir["$ids.ids_properties.homogeneous_time"] = 1
    ir["$ids.ids_properties.source"] = meta.source
    ir["$ids.ids_properties.provider"] = meta.provider
    ir["$ids.ids_properties.comment"] = meta.comment
    ir["$ids.code.name"] = meta.code_name
    ir["$ids.time"] = times
    return nothing
end

"""
    assemble_ir(slices, meta) -> Dict{String,Any}

Turn per-slice SI result NamedTuples into a flat dict of dotted IMAS paths
(consumed by `write_omas_h5`). Populates `core_profiles` and `equilibrium` for
every slice, plus impurity charge-state species (neutral + ion[1..Z]) and a
`disruption` radiated-power-density profile when those per-slice fields are
present. `meta` carries IDS metadata + ion/vacuum-field constants. Missing
per-slice fields (`nothing`) are omitted.

An optional `meta.globals` NamedTuple of SI 0D traces (each a per-slice vector
or `nothing`; see `_read_globals`) adds the `summary` IDS
(`global_quantities.{ip, energy_thermal, v_loop, power_radiated}.value`), the
per-slice `equilibrium…global_quantities.ip`, and the `disruption` globals
`power_radiated_total` (= the radiation trace) and `power_ohm` (= Ip·V_loop,
raw M3D-C1 signs).
"""
function assemble_ir(slices::AbstractVector, meta::NamedTuple)
    ir = Dict{String, Any}()
    times = Float64[s.time for s in slices]
    for ids in ("core_profiles", "equilibrium")
        _ids_props!(ir, ids, meta, times)
    end
    gq = get(meta, :globals, nothing)
    gval(k) = gq === nothing ? nothing : get(gq, k, nothing)
    ip_trace = gval(:ip)
    isfinite(meta.r0) && (ir["equilibrium.vacuum_toroidal_field.r0"] = meta.r0)
    isfinite(meta.b0) &&
        (ir["equilibrium.vacuum_toroidal_field.b0"] = fill(Float64(meta.b0), length(slices)))
    # core_profiles carries its own copy of the vacuum field, so a consumer
    # reading only that IDS still knows the B0/R0 its profiles refer to.
    isfinite(meta.r0) && (ir["core_profiles.vacuum_toroidal_field.r0"] = meta.r0)
    isfinite(meta.b0) &&
        (ir["core_profiles.vacuum_toroidal_field.b0"] = fill(Float64(meta.b0), length(slices)))

    for (it, s) in enumerate(slices)
        i = it - 1                                   # 0-based OMAS index
        # the ψ written on the core_profiles/disruption grids; a convention
        # layer may override it (`grid_psi1d`, e.g. the MHDsimDB layout) while
        # the equilibrium IDS keeps `psi1d`
        gpsi = get(s, :grid_psi1d, s.psi1d)
        cp = "core_profiles.profiles_1d.$i"
        ir["$cp.grid.rho_pol_norm"] = s.rho
        ir["$cp.grid.psi"] = gpsi
        _maybe!(ir, "$cp.e_field.parallel", get(s, :epar1d, nothing))
        _maybe!(ir, "$cp.electrons.temperature", s.te)
        _maybe!(ir, "$cp.electrons.density", s.ne)
        _maybe!(ir, "$cp.ion.0.temperature", s.ti)
        _maybe!(ir, "$cp.ion.0.density", s.ni)
        _maybe!(ir, "$cp.q", get(s, :q1d, nothing))
        # FSA ⟨|B|⟩ — non-standard ML feature; same custom path the fusion-io
        # mapper used (fusion_io_mappers.py:127)
        _maybe!(ir, "$cp.custom.b_field_torus_average", get(s, :babs1d, nothing))
        _maybe!(ir, "$cp.custom.deltab_over_b", get(s, :deltab1d, nothing))
        if s.ni !== nothing
            ir["$cp.ion.0.label"] = meta.ion_label
            ir["$cp.ion.0.z_ion"] = Float64(meta.z_ion)
            ir["$cp.ion.0.element.0.a"] = Float64(meta.ion_a)
            ir["$cp.ion.0.element.0.z_n"] = Int(round(meta.z_ion))
            ir["$cp.ion.0.element.0.atoms_n"] = 1
        end

        # impurity charge states: neutral.0 (state 0) + ion.1..Z (states 1..Z)
        imp = get(s, :imp_dens, nothing)
        if imp !== nothing
            _maybe!(ir, "$cp.neutral.0.density", imp[1])
            if imp[1] !== nothing
                ir["$cp.neutral.0.label"] = meta.imp_label
                ir["$cp.neutral.0.element.0.z_n"] = Int(meta.imp_z)
                ir["$cp.neutral.0.element.0.a"] = Float64(meta.imp_a)
                ir["$cp.neutral.0.element.0.atoms_n"] = 1
            end
            for iz in 1:(length(imp) - 1)
                d = imp[iz + 1]
                d === nothing && continue
                ip = "$cp.ion.$iz"
                ir["$ip.density"] = d
                ir["$ip.label"] = "$(meta.imp_label)$iz+"
                ir["$ip.z_ion"] = Float64(iz)
                ir["$ip.element.0.z_n"] = Int(meta.imp_z)
                ir["$ip.element.0.a"] = Float64(meta.imp_a)
                ir["$ip.element.0.atoms_n"] = 1
            end
            # optional quasi-neutral electron density: nₑ = n_main + Σ_iz iz·n_imp,iz
            if get(meta, :recompute_ne, false) && s.ni !== nothing
                ne_qn = copy(s.ni)
                for iz in 1:(length(imp) - 1)
                    imp[iz + 1] === nothing ||
                        (ne_qn = ne_qn .+ Float64(iz) .* ifelse.(isfinite.(imp[iz + 1]), imp[iz + 1], 0.0))
                end
                ir["$cp.electrons.density"] = ne_qn
            end
        end

        ts = "equilibrium.time_slice.$i"
        ir["$ts.time"] = Float64(s.time)
        ir["$ts.profiles_1d.psi"] = s.psi1d
        _maybe!(ir, "$ts.profiles_1d.pressure", s.pressure)
        _maybe!(ir, "$ts.profiles_1d.q", get(s, :q1d, nothing))
        _maybe!(ir, "$ts.profiles_1d.phi", get(s, :phi1d, nothing))
        _maybe!(ir, "$ts.profiles_1d.f", get(s, :F1d, nothing))
        _maybe!(ir, "$ts.profiles_1d.f_df_dpsi", get(s, :ffprime1d, nothing))
        _maybe!(ir, "$ts.profiles_1d.dpressure_dpsi", get(s, :pprime1d, nothing))
        _maybe!(ir, "$ts.profiles_1d.j_tor", get(s, :jtor1d, nothing))
        # rho_tor = sqrt(Φ/(π·b0)) (IMAS DD); needs the toroidal flux and the
        # vacuum field. Φ and b0 share a sign under COCOS 11 / :mhdsimdb (where
        # `_cocos11_meta` signs b0 like F), but `cocos = nothing` keeps the
        # unsigned `bzero` magnitude — so take |Φ/b0|, which is the same number
        # whenever the signs are consistent and keeps rho_tor a length (≥0).
        let phi = get(s, :phi1d, nothing)
            if phi !== nothing && isfinite(meta.b0) && meta.b0 != 0
                rt = [sqrt(abs(p / (π * meta.b0))) for p in phi]
                if any(isfinite, rt) && (rte = last(filter(isfinite, rt))) > 0
                    ir["$ts.profiles_1d.rho_tor"] = rt
                    ir["$ts.profiles_1d.rho_tor_norm"] = rt ./ rte
                end
            end
        end
        ir["$ts.profiles_2d.0.grid.dim1"] = s.Rg
        ir["$ts.profiles_2d.0.grid.dim2"] = s.Zg
        ir["$ts.profiles_2d.0.grid_type.index"] = 1
        ir["$ts.profiles_2d.0.grid_type.name"] = "rectangular"
        ir["$ts.profiles_2d.0.psi"] = s.psi_rz
        _maybe!(ir, "$ts.profiles_2d.0.b_field_r", get(s, :br_rz, nothing))
        _maybe!(ir, "$ts.profiles_2d.0.b_field_z", get(s, :bz_rz, nothing))
        _maybe!(ir, "$ts.profiles_2d.0.b_field_tor", get(s, :bphi_rz, nothing))
        _maybe!(ir, "$ts.profiles_2d.0.phi", get(s, :phi_rz, nothing))
        ir["$ts.global_quantities.psi_axis"] = Float64(s.psi_axis)
        ir["$ts.global_quantities.psi_boundary"] = Float64(s.psi_boundary)
        ir["$ts.global_quantities.magnetic_axis.r"] = Float64(s.R_axis)
        ir["$ts.global_quantities.magnetic_axis.z"] = Float64(s.Z_axis)
        ip_trace === nothing || (ir["$ts.global_quantities.ip"] = Float64(ip_trace[it]))
        let b = get(s, :lcfs_rz, nothing)
            if b !== nothing
                ir["$ts.boundary.outline.r"] = b.R
                ir["$ts.boundary.outline.z"] = b.Z
                ir["$ts.boundary.psi"] = Float64(s.psi_boundary)
            end
        end
        # recomputed X-points (find_lcfs saddles) → boundary IDS
        for (k, xp) in enumerate(get(s, :x_points, NTuple{2, Float64}[]))
            ir["$ts.boundary.x_point.$(k - 1).r"] = xp[1]
            ir["$ts.boundary.x_point.$(k - 1).z"] = xp[2]
        end
    end

    # summary IDS: 0D global traces (only when at least one trace was found)
    if gq !== nothing && any(v -> v !== nothing, values(gq))
        _ids_props!(ir, "summary", meta, times)
        _maybe!(ir, "summary.global_quantities.ip.value", ip_trace)
        _maybe!(ir, "summary.global_quantities.energy_thermal.value", gval(:energy_thermal))
        _maybe!(ir, "summary.global_quantities.v_loop.value", gval(:v_loop))
        _maybe!(ir, "summary.global_quantities.power_radiated.value", gval(:power_radiated))
    end

    # disruption IDS: radiated-power-density profiles and/or 0D globals.
    # power_radiated_total is the volume-integrated `radiation` trace as-is
    # (M3D-C1's Te-sink sum — see the caveat in `export_imas`' docstring);
    # power_ohm = Ip·V_loop, the product of the traces with raw M3D-C1 signs.
    have_prad = any(s -> get(s, :prad1d, nothing) !== nothing, slices)
    vloop = gval(:v_loop);  prad_tot = gval(:power_radiated)
    # power_ohm stays Ip·V_loop from M3D-C1's own traces. NOTE it is identically
    # zero on every run in this database: their `loop_voltage` scalar is never
    # written (they are current-controlled, `i_control`), so this trace only
    # LOOKS like data. The usable ohmic information is the per-slice
    # NOTE it is identically zero on every run in this database.
    p_ohm = (ip_trace !== nothing && vloop !== nothing) ? ip_trace .* vloop : nothing
    if have_prad || prad_tot !== nothing || p_ohm !== nothing
        _ids_props!(ir, "disruption", meta, times)
        _maybe!(ir, "disruption.global_quantities.power_radiated_total", prad_tot)
        _maybe!(ir, "disruption.global_quantities.power_ohm", p_ohm)
        for (it, s) in enumerate(slices)
            p = get(s, :prad1d, nothing)
            p === nothing && continue
            db = "disruption.profiles_1d.$(it - 1)"
            ir["$db.grid.rho_pol_norm"] = s.rho
            ir["$db.grid.psi"] = get(s, :grid_psi1d, s.psi1d)
            ir["$db.time"] = Float64(s.time)
            ir["$db.power_density_radiative_losses"] = p
        end
    end
    return ir
end

# 0D scalar traces exported as global quantities: IR key → (scalars/* dataset,
# unit-registry quantity). Dataset choices ground-truthed from the M3D-C1
# source (diagnostics.f90 / output.f90):
#   toroidal_current — whole-domain jφ integral (halo included, the Rogowski-
#     equivalent Ip; `toroidal_current_p` would be the LCFS-interior variant)
#   W_P — plasma-region ∫p/(γ−1)dV (thermal energy; `E_P` integrates the whole
#     computational domain, vacuum region included)
#   loop_voltage — the controller trace `vloop`; normalized as flux/time
#     (model.f90: dψ/dt += vloop/2π), which is exactly the `:voltage` factor
#   radiation — volume-integrated KPRAD loss power (sum definition pending —
#     see the caveat in `export_imas`' docstring)
const _GLOBAL_SCALARS = (
    ip = ("toroidal_current", :current),
    energy_thermal = ("W_P", :energy),
    v_loop = ("loop_voltage", :voltage),
    power_radiated = ("radiation", :power),
)

# Read the `_GLOBAL_SCALARS` traces, sample each at the exported slices'
# `ntimestep`s, and convert to SI. Traces a file does not provide (or that are
# too short for the requested steps) come back `nothing` and are omitted
# downstream.
function _read_globals(
        file::M3DC1File, norm::M3DNormalization,
        nsteps::AbstractVector{<:Integer}
    )
    vals = map(keys(_GLOBAL_SCALARS)) do key
        name, qty = getproperty(_GLOBAL_SCALARS, key)
        tr = try
            read_scalar(file, name)
        catch err
            @debug "export_imas: scalar trace $name unavailable" exception = err
            nothing
        end
        (tr === nothing || any(n -> n + 1 > length(tr), nsteps)) && return key => nothing
        u = unit_factor(norm, qty; system = :si)
        return key => Float64[tr[n + 1] * u for n in nsteps]
    end
    return (; vals...)
end

# field symbol → unit-registry quantity for SI conversion
const _FIELD_QTY = (
    te = :temperature, ne = :density, ti = :temperature,
    den = :density, P = :pressure,
    E_par = :electric_field,
)

"""
    reduce_axisym_slice(fields, ep, nplanes, norm, psi_axis_plane1, psi_lcfs_plane1,
                  xmag, zmag, time_s, Rg_n, Zg_n, id_map;
                  nbins=128, adj=:linear, fsa_method=:bin, fsa_window=4.0, kprad_z=-1,
                  lcfs_ntheta=257, weighted=true, axis_aug=true,
                  sigma_cells=1.5, ntrunc=2.0,
                  xnull=NaN, znull=NaN, xnull2=NaN, znull2=NaN,
                  xlim=NaN, zlim=NaN, xlim2=NaN, zlim2=NaN, wall_rz=nothing)

Reduce one slice's 80-coef `fields` to SI 1D profiles + 2D ψ. `fields` must
have `:psi`; optional `:te,:ne,:ti,:den,:P`, `:fp` (f′ = ∂f/∂φ; legacy files
carry `:f` instead — with `:I` either enables the `deltab1d` δB/B fluctuation
profile, [`deltab_over_b_rz`](@ref)), and `:I` (=F=R·Bφ) which enables
`F1d`, the safety factor `q1d` (FSA identity q = F·⟨1/R²⟩·V′/4π², per-radian;
q_dd(COCOS-11) = 2π·q; separatrix node NaN, axis region from an integral
V-fit), the toroidal flux `phi1d = 2π∫q dψ`, and the axisymmetric B field:
2D maps `br_rz`/`bz_rz`/`bphi_rz` [T] (B_R = −ψ_Z/R, B_Z = ψ_R/R, Bφ = I/R
from the exact FEM ∇ψ) with the FSA `babs1d` = ⟨|B|⟩ [T], plus the confined-
region toroidal-flux map `phi_rz` (NaN outside the LCFS).
When `kprad_z ≥ 0`, also FSA the impurity charge-state densities
`:kprad_n_00..:kprad_n_{kprad_z}` (→ `imp_dens`). Independently, if any KPRAD
radiation field (`:kprad_rad/brem/reck/recp`) is present in `fields`, FSA
their sum (→ `prad1d`, W/m³). Grids `Rg_n,Zg_n` are in mesh (normalized)
units; outputs are SI. Returns the per-slice result NamedTuple (including the
recomputed SI `x_points`).

`lcfs_ntheta` sets the angular resolution of the `lcfs_rz` boundary outline
([`trace_lcfs`](@ref)); the cost is ~0.2 s/slice at the default 257 on an 11k-
element mesh, and X-points are snapped in exactly regardless of the resolution.

Every 1D reduction is a physical flux-surface average: samples carry the
toroidal volume measure `cell_area · R` as quadrature weight (`weighted=false`
reverts to the plain unweighted bin mean — and disables `q1d`, whose V′ needs
the volume measure), private-flux-region samples (across a recomputed X-point
from the axis, sharing ψ_N < 1 with confined surfaces) are excluded, and a
finer sampling patch around the recomputed magnetic axis (`axis_aug`, 50×50
over ±12% of the R span, contributing at ρ<0.2) fills the innermost ρ bins on
coarse grids. `adj` selects the binning kernel (`:linear` default, `:gauss`
for built-in smoothing with `sigma_cells`/`ntrunc`); see
[`reduce_1d_psi_func`](@ref).

`fsa_method` selects the estimator for the ratio-type profiles (`te, ne, ti, ni,
pressure, babs1d, deltab1d`, impurity densities, radiated power): `:bin` (default) is the
per-bin kernel average above; `:cumulative` computes the smoother
`⟨f⟩(ψ)=W′(ψ)/V′(ψ)` from cumulative volume integrals (`_fsa_cumulative`, the
same integrate-then-differentiate approach that de-noises `q1d`) — it replaces
the per-bin shot noise (∝1/√samples-per-bin) with the lower density-estimation
floor, and ignores `adj`/`weighted` (it is inherently volume-weighted). It also
covers `jtor1d`, whose two reductions (⟨jφ/R⟩ and ⟨1/R⟩) are ordinary surface
averages of fields that genuinely vary *on* a surface. It does **not** apply to
`F1d` / `q1d` / `phi1d` / `ffprime1d` / `pprime1d`, which are pinned to `:bin`:
`q1d`/`phi1d` are already cumulative by construction, and the others are smooth
flux functions (or ψ-derivatives of them) whose surface average is exact, so
there is no shot noise for the regression window to trade against — it can only
bias them. Measured on the JET SPI slice 0, `:cumulative` puts F ~2e-3 off near
the axis (vs 3e-5 for `:bin`) and the bias *grows* with grid refinement instead
of converging away, which in turn breaks the `f` ↔ `f_df_dpsi` consistency by
50×. `fsa_window` (default 4)
sets the local-regression window width (∝ grid cells) for `:cumulative` only:
**smaller preserves sharp edge features (e.g. a pedestal) at the cost of a little
more noise; larger smooths harder.** The default was tuned against the
[`fsa_imas`](@ref) contour reference (which does no cross-ψ smoothing) on a real
SPI slice: `16` over-smooths the ne pedestal by ~4.4%, `4` holds it to ~1.1% at
negligible extra roughness (below ~6 the `wmin` floor dominates, so 2≈4).

ψ_N (and hence ρ_pol, shared by every profile) uses the O-point and boundary
flux recomputed **per slice on the n=0 averaged ψ** by [`find_lcfs`](@ref),
seeded from the stored plane-1 scalars: the axis from `(xmag, zmag)`, the
X-points from `(xnull, znull)` / `(xnull2, znull2)`, the limiter candidates
from `(xlim, zlim)` / `(xlim2, zlim2)` ([`limiter_points`](@ref)), plus an
optional `wall_rz` first-wall scan. Whatever candidates are absent are simply
skipped; if the recomputation fails entirely (or finds no boundary candidate),
the stored plane-1 `psi_axis_plane1` / `psi_lcfs_plane1` flow through as fallbacks.
"""
@with_pool spool function reduce_axisym_slice(
        fields::AbstractDict{Symbol}, ep, nplanes::Integer,
        norm::M3DNormalization, psi_axis_plane1::Real, psi_lcfs_plane1::Real,
        xmag::Real, zmag::Real, time_s::Real,
        Rg_n::AbstractVector, Zg_n::AbstractVector,
        id_map::AbstractMatrix{<:Integer};
        nbins::Integer = 128, adj::Symbol = :linear, fsa_method::Symbol = :bin,
        fsa_window::Real = 4.0, kprad_z::Integer = -1, lcfs_ntheta::Integer = 257,
        weighted::Bool = true, axis_aug::Bool = true,
        sigma_cells::Real = 1.5, ntrunc::Real = 2.0,
        xnull::Real = NaN, znull::Real = NaN,
        xnull2::Real = NaN, znull2::Real = NaN,
        xlim::Real = NaN, zlim::Real = NaN,
        xlim2::Real = NaN, zlim2::Real = NaN, wall_rz = nothing
    )
    fsa_method === :bin || fsa_method === :cumulative || throw(
        ArgumentError(
            "reduce_axisym_slice: fsa_method must be :bin or :cumulative " *
                "(:imas is a separate contour reference — see fsa_imas), got $fsa_method"
        )
    )
    uflux = unit_factor(norm, :magnetic_flux; system = :si)
    ulen = unit_factor(norm, :length; system = :si)

    # n=0 coefficient buffers are slice-local temporaries (consumed by the 2D
    # interpolation / FSA, never returned) drawn from the slice pool `spool` and
    # reused across slices. All fields share one (ncoef, npp) shape.
    _coef_buf(key) = average_toroidal_axisymmetric!(
        acquire!(spool, Float64, size(fields[key], 1), size(fields[key], 2) ÷ nplanes),
        fields[key], nplanes
    )
    psi_axisym_coef = _coef_buf(:psi)
    # O-point + boundary flux recomputed on the n=0 averaged ψ; stored plane-1
    # scalars are the fallback when the solve fails or finds no boundary candidate.
    ψ0 = Float64(psi_axis_plane1);  ψ1 = Float64(psi_lcfs_plane1)
    R_ax = Float64(xmag);           Z_ax = Float64(zmag)
    lc = try
        find_lcfs(
            psi_axisym_coef, ep; xmag = Float64(xmag), zmag = Float64(zmag),
            xnull, znull, xnull2, znull2, xlim, zlim, xlim2, zlim2, wall_rz
        )
    catch err
        @warn "reduce_axisym_slice: axis/LCFS recomputation failed; keeping stored plane-1 scalars" exception = err
        nothing
    end
    if lc !== nothing
        ψ0 = lc.psi_axis;  R_ax = lc.axis.R;  Z_ax = lc.axis.Z
        if isfinite(lc.psi_bound)
            # large deviation from the stored value signals a candidate we could
            # not evaluate (e.g. plasma gone wall-limited with no wall_rz given)
            abs(lc.psi_bound - ψ1) > 0.05 * abs(ψ1 - ψ0) &&
                @warn "reduce_axisym_slice: recomputed boundary flux deviates >5% of the span from stored psi_lcfs" recomputed = lc.psi_bound stored = ψ1 limited_by = lc.limited_by
            ψ1 = lc.psi_bound
        end
    end

    # ψ AND its exact gradient on the grid (∇ψ feeds the B-field assembler)
    pg = interpolate_axisym_gradient_to_grid(psi_axisym_coef, ep, Rg_n, Zg_n; id_map = id_map)
    psi_rz = pg.val
    ψn = psi_to_psi_norm.(vec(psi_rz), ψ0, ψ1)
    ρpol = psi_n_to_rho_pol.(ψn)
    ρgrid = collect(range(0.0, 1.0, length = nbins))

    # FSA quadrature weight: cell_area · R — the toroidal volume measure (the
    # 1/|∇ψ| factor is implicit in uniform-density sampling). Only weight
    # ratios within a bin matter, so normalized mesh units are fine.
    dA_g = (Rg_n[2] - Rg_n[1]) * (Zg_n[2] - Zg_n[1])
    w_g = vec([dA_g * Float64(Rg_n[i]) for i in eachindex(Rg_n), _ in eachindex(Zg_n)])
    any(x -> x > 0, w_g) || fill!(w_g, 1.0)     # degenerate grid (zero span/R): plain mean

    # confined-region mask: private-flux-region samples share ψ_N < 1 with
    # confined surfaces but lie across an X-point — drop them from every FSA
    # (same dot test as M3D-C1's wall filter / IMAS' region validation).
    conf = trues(length(ρpol))
    if lc !== nothing
        nR = length(Rg_n)
        for x in (lc.x1, lc.x2)
            (x !== nothing && x.converged && x.kind === :saddle) || continue
            # half-plane cut at the X-point: keep grid points on the axis side of
            # the line through the X-point ⟂ to (axis − X). Applied in place — no
            # per-X-point Bool matrix (`vec([… for i,j])` column-major → idx =
            # (j-1)·nR + i, matching ρpol/w_g).
            xR = Float64(x.R);   xZ = Float64(x.Z)
            dRax = R_ax - xR;    dZax = Z_ax - xZ
            @inbounds for j in eachindex(Zg_n), i in eachindex(Rg_n)
                inside = (Float64(Rg_n[i]) - xR) * dRax + (Float64(Zg_n[j]) - xZ) * dZax > 0
                conf[(j - 1) * nR + i] &= inside
            end
        end
    end

    # axis-cell augmentation: a finer sampling patch around the recomputed axis
    # feeds extra (ρ, value, weight) samples into every reduction, filling the
    # innermost ρ bins on coarse grids. Only when the axis solve succeeded.
    Ra = Za = nothing;  ida = nothing;  pga = nothing
    ρ_a = Float64[];  amask = Bool[];  w_a = Float64[]
    if axis_aug && lc !== nothing
        δ = 0.12 * (maximum(Rg_n) - minimum(Rg_n));  na = 50
        Ra = collect(range(R_ax - δ, R_ax + δ; length = na))
        Za = collect(range(Z_ax - δ, Z_ax + δ; length = na))
        ida = build_grid_to_element_map(Ra, Za, ep)
        # ψ AND ∇ψ on the patch in one pass: ψ sets the patch ρ coordinate,
        # ∇ψ feeds ⟨|B|⟩ and the ψ-derivative profiles below.
        pga = interpolate_axisym_gradient_to_grid(psi_axisym_coef, ep, Ra, Za; id_map = ida)
        ψ_a = vec(pga.val)
        ρ_a = psi_n_to_rho_pol.(psi_to_psi_norm.(ψ_a, ψ0, ψ1))
        amask = isfinite.(ρ_a) .& (ρ_a .< 0.2)
        dA_a = (Ra[2] - Ra[1]) * (Za[2] - Za[1])
        w_a = vec([dA_a * Ra[i] for i in 1:na, _ in 1:na])
    end

    # Combined (grid + axis-patch) sample geometry — field-INDEPENDENT, so build
    # the ρ and weight vectors once and reuse them for every field's reduction
    # (previously re-`vcat`ed per field). Each field folds its confined mask into
    # its own value vector as NaN, so `reduce_1d_psi_func` does the single finite
    # filter (no per-field ρ/w compaction here).
    patch_on = Ra !== nothing
    ρ_all = patch_on ? vcat(ρpol, ρ_a[amask]) : ρpol
    w_all = patch_on ? vcat(w_g, w_a[amask]) : w_g
    np_all = length(ρ_all)

    # n=0 averaged coefficients, cached per key (grid map + axis patch share them)
    coef_cache = Dict{Symbol, Matrix{Float64}}()
    field_axisym(key::Symbol) = haskey(fields, key) ?
        get!(() -> _coef_buf(key), coef_cache, key) :
        nothing
    field_rz(key::Symbol) = (
        c = field_axisym(key);
        c === nothing ? nothing :
            interpolate_axisym_to_grid(c, ep, Rg_n, Zg_n; id_map = id_map)
    )
    field_patch(key::Symbol) = (
        c = field_axisym(key);
        (c === nothing || Ra === nothing) ? nothing :
            vec(interpolate_axisym_to_grid(c, ep, Ra, Za; id_map = ida))
    )

    # reduce grid samples (+ axis-patch samples); returns the raw reduction
    # NamedTuple (func_bin + den) or nothing. The confined mask is folded into
    # the value vector as NaN so reduce_1d_psi_func does the single finite filter;
    # the value vector `vv` is a per-field temporary drawn from the pool (rewound
    # at the block's return — no early return inside, so `@with_pool` is safe).
    #
    # `method` overrides `fsa_method` for the quantities where the `:cumulative`
    # smoothing is inappropriate: it exists to de-noise *ratio* profiles whose
    # per-bin estimate is shot-noise-limited (ne/Te/δB/B), and it pays for that
    # with a regression window. F(ψ) and the ψ-derivatives are smooth flux
    # functions with no shot-noise problem, so the window only biases them (F
    # measured 2e-3 off near the axis under `:cumulative` vs 3e-5 under `:bin`,
    # and the bias grows with grid refinement instead of converging away).
    function reduce_raw(vgrid, vpatch; method::Symbol = fsa_method)
        vgrid === nothing && return nothing
        vgv = vec(vgrid)
        ng = length(vgv)
        return @with_pool fp begin
            vv = acquire!(fp, Float64, np_all)
            @inbounds for i in 1:ng
                vv[i] = conf[i] ? vgv[i] : NaN     # drop private-flux samples
            end
            if np_all > ng                          # axis-patch samples (trivially confined)
                j = ng
                if vpatch === nothing
                    @inbounds for jj in (ng + 1):np_all
                        vv[jj] = NaN
                    end
                else
                    @inbounds for t in eachindex(amask)
                        amask[t] || continue
                        j += 1;  vv[j] = vpatch[t]
                    end
                end
            end
            cnt = 0
            @inbounds for i in 1:np_all
                (isfinite(ρ_all[i]) & isfinite(vv[i])) && (cnt += 1)
            end
            if cnt < 2
                nothing
            elseif method === :cumulative
                # smooth W′/V′ estimator (inherently volume-weighted; ignores adj).
                # fsa_window sets the regression window (∝ grid cells) — smaller
                # preserves edge pedestals, larger smooths more (see _vprime_cumvol).
                _fsa_cumulative(ρ_all, vv, w_all, ρgrid; wfac = fsa_window)
            else
                reduce_1d_psi_func(
                    ρ_all, vv; psi_grid = ρgrid, adj = adj,
                    weights = weighted ? w_all : nothing,
                    sigma_cells = sigma_cells, ntrunc = ntrunc
                )
            end
        end
    end
    reduce_si(vgrid, vpatch, quantity::Symbol) = (
        r = reduce_raw(vgrid, vpatch);
        r === nothing ? nothing : r.func_bin .* unit_factor(norm, quantity; system = :si)
    )
    prof(key::Symbol) = reduce_si(field_rz(key), field_patch(key), getproperty(_FIELD_QTY, key))

    # impurity charge-state densities: FSA of kprad_n_00 .. kprad_n_{kprad_z}
    imp_dens = kprad_z < 0 ? nothing :
        Union{Nothing, Vector{Float64}}[
            (
                k = Symbol("kprad_n_" * lpad(iz, 2, '0'));
                reduce_si(field_rz(k), field_patch(k), :density)
            ) for iz in 0:kprad_z
        ]

    # radiated-power-density profile: FSA of the summed KPRAD radiation fields
    prad_present = filter(k -> haskey(fields, k), (:kprad_rad, :kprad_brem, :kprad_reck, :kprad_recp))
    prad1d = isempty(prad_present) ? nothing :
        reduce_si(
            reduce(.+, (vec(field_rz(k)) for k in prad_present)),
            Ra === nothing ? nothing : reduce(.+, (field_patch(k) for k in prad_present)),
            :power_density
        )

    psi1d = (ψ0 .+ ρgrid .^ 2 .* (ψ1 - ψ0)) .* uflux

    # F(ψ)=R·Bφ (from the :I field) and the safety factor via the FSA identity
    #   q(ψ) = F · ⟨1/R²⟩ · V′(ψ) / (4π²) = F · U′(ψ) / (4π²),
    # where U(ψ) = ∫_{ψ′<ψ} (1/R²) dV — the ⟨1/R²⟩·V′ product collapses to ONE
    # cumulative derivative (⟨1/R²⟩ = U′/V′ exactly). U(ψ_N) is computed
    # exactly (sorted confined samples + cumsum) and differentiated by local
    # quadratic regression (`_vprime_cumvol`); this replaced the per-bin `den`
    # estimate whose shell shot noise used to put ~5% bin-to-bin jitter on q
    # (the old axis-only V-fit is the ψ_N→0 limit of this). Residual noise is
    # the density-estimation floor ∝ 1/√(samples per window) — for a smoother
    # reference use the IMAS contour cross-validation (gated testset in
    # test_equilibrium_imas.jl; median q_imas/(2π·q) within 1%). The ρ=1 node
    # stays NaN (q diverges at the separatrix).
    F1d = nothing;  q1d = nothing;  phi1d = nothing
    rI = reduce_raw(field_rz(:I), field_patch(:I); method = :bin)
    if rI !== nothing
        F1d = rI.func_bin .* (ulen * unit_factor(norm, :magnetic_field; system = :si))
        # the innermost ρ bin can be empty on a coarse grid (the shell has no
        # samples) — rebuild only those nodes, never overwrite a finite F
        _fill_axis_band!(F1d, ρgrid .^ 2; only_nonfinite = true)
        if weighted   # U′ needs the volume measure
            Δψ_si = (ψ1 - ψ0) * uflux
            cm = conf .& isfinite.(ψn) .& (ψn .>= 0.0) .& (ψn .<= 1.0)
            xs = ψn[cm]
            # dU = dV/R² = 2π dA·R·ulen³ / (R·ulen)² = 2π dA ulen / R
            us = (2π * dA_g * ulen) .* vec(
                [1.0 / Float64(Rg_n[i]) for i in eachindex(Rg_n), _ in eachindex(Zg_n)]
            )[cm]
            p = sortperm(xs);  xs = xs[p];  Uc = cumsum(us[p])
            if length(xs) >= 25
                Up = _vprime_cumvol(xs, Uc, ρgrid, Δψ_si)
                q1d = F1d .* Up ./ (4π^2)
                q1d[end] = NaN
                isfinite(q1d[1]) || (q1d[1] = q1d[2])
                # toroidal flux Φ(ψ) = 2π ∫ q dψ (per-radian ψ); endpoints of q
                # filled with the nearest interior value for the quadrature
                qf = copy(q1d);  qf[1] = qf[2];  qf[end] = qf[end - 1]
                if all(isfinite, qf)
                    phi1d = zeros(nbins)
                    for k in 2:nbins
                        phi1d[k] = phi1d[k - 1] + π * (qf[k] + qf[k - 1]) * (psi1d[k] - psi1d[k - 1])
                    end
                end
            end
        end
    end

    # ψ-derivatives of the flux functions F and p → IMAS `f_df_dpsi` [T²m²/Wb]
    # and `dpressure_dpsi` [Pa/Wb]. Differentiating the *binned* `F1d`/`pressure`
    # would be far worse: the ρ_pol-uniform grid makes Δψ ∝ ρ² collapse toward
    # the axis, so any binning residual is amplified by 1/Δψ (measured 12% of
    # the FF′ scale over the whole profile, 18–24% inside ψ_N < 0.2). Instead
    # the derivative is taken *before* the average — exactly and pointwise, from
    # the FEM representation: for a flux function f(ψ),
    #     df/dψ = ∇f·∇ψ / |∇ψ|²   ⇒   FF′ = I·(∇I·∇ψ)/|∇ψ|²,  p′ = (∇p·∇ψ)/|∇ψ|²
    # which are ordinary 2D scalar fields that then go through the *same* FSA
    # binning as every other profile (mask, axis patch, volume weights and all).
    # The ∇-pair cancels the length normalization, so the SI factor is just the
    # quantity's own unit ratio. Pinned to `:bin` for the reason in `reduce_raw`.
    #
    # Validated against the run's own EFIT input (JET SPI slice 0): 0.021% of
    # the ffprim scale over ψ_N 0.2–0.98 at ngrid = 200 (0.007% at 400), 0.036%
    # over the whole profile after `_fill_axis_band!`. Integrating FF′ back
    # recovers `F1d` to 8.7e-6 relative — the exported pair is self-consistent
    # to better than its own accuracy (which needs `F1d` on `:bin`, see above).
    #
    # The p / `dpressure_dpsi` pair does NOT close that tightly, and the gap is
    # mostly physics rather than numerics. `pressure` is a ratio profile (so it
    # follows `fsa_method`, `:cumulative` in the production scripts) while `p′`
    # is pinned to `:bin`; on the JET SPI t=0 slice, where p is still an exact
    # flux function, matching the estimators takes the closure from 4.0e-3 to
    # 4.1e-4. Later in the run it stays ~2-3e-3 under either estimator, because
    # p has stopped being a flux function: measured on that run, p varies 7% on
    # the ψ_N = 0.8 surface by t = 1.4 ms and 18% by t = 2.5 ms, tracking the
    # pellet inward, while F stays a flux function to <0.03% throughout. So
    # `f_df_dpsi` is exact at every slice, and `dpressure_dpsi` is a best-fit
    # flux function once the cooling front has passed a given surface.
    ffprime1d = nothing;  pprime1d = nothing
    let g2 = pg.dR .^ 2 .+ pg.dZ .^ 2
        gmax = 0.0
        @inbounds for v in g2
            (isfinite(v) && v > gmax) && (gmax = v)
        end
        g2a = pga === nothing ? nothing : (pga.dR .^ 2 .+ pga.dZ .^ 2)
        gfloor = _GRAD_PSI_FLOOR * gmax
        ψN_nodes = ρgrid .^ 2
        # FSA of df/dψ (× f when `times_f`), from the exact FEM ∇f and ∇ψ
        function dpsi_prof(key::Symbol, ufac::Real; times_f::Bool = false)
            c = (gmax > 0) ? field_axisym(key) : nothing
            c === nothing && return nothing
            fg = interpolate_axisym_gradient_to_grid(c, ep, Rg_n, Zg_n; id_map = id_map)
            vg = fill(NaN, size(g2))
            @inbounds for i in eachindex(g2)
                g2[i] > gfloor || continue
                vg[i] = (times_f ? fg.val[i] : 1.0) *
                    (fg.dR[i] * pg.dR[i] + fg.dZ[i] * pg.dZ[i]) / g2[i]
            end
            vp = nothing
            if g2a !== nothing
                fa = interpolate_axisym_gradient_to_grid(c, ep, Ra, Za; id_map = ida)
                vp = fill(NaN, length(g2a))
                @inbounds for i in eachindex(vp)
                    g2a[i] > gfloor || continue
                    vp[i] = (times_f ? fa.val[i] : 1.0) *
                        (fa.dR[i] * pga.dR[i] + fa.dZ[i] * pga.dZ[i]) / g2a[i]
                end
            end
            r = reduce_raw(vg, vp; method = :bin)
            r === nothing && return nothing
            return _fill_axis_band!(r.func_bin .* ufac, ψN_nodes)
        end
        ub_si = unit_factor(norm, :magnetic_field; system = :si)
        ffprime1d = dpsi_prof(:I, (ulen * ub_si)^2 / uflux; times_f = true)
        pprime1d = dpsi_prof(:P, unit_factor(norm, :pressure; system = :si) / uflux)
    end

    # toroidal current density (needs :jphi).
    #
    # **M3D-C1's `jphi` output field is NOT a current density: it is +Δ*ψ, i.e.
    # −μ₀·R·J_φ.** The HDF5 write (`output.f90:1307`) takes `jphi_field`, which
    # `newpar.f90:844` solves as M⁻¹·G·ψ against `NV_GS_MATRIX`, and
    # `newvar.f90:239` makes that matrix `intxx2(mu(OP_1), nu(OP_GS))` with
    # `OP_GS = ∂_RR + ∂_ZZ − (1/R)∂_R` (`m3dc1_nint.f90:302`) — a pure L2
    # projection of Δ*ψ, no R weighting and no sign. Ampère in M3D-C1's own
    # representation is μ₀J_φ = −Δ*ψ/R (`doc/numerical_methods.tex:26-28`;
    # `model.f90:514` "no toroidal current = -Delta*(psi)/R"), so jφ = −jphi/R
    # in mesh units. The NEGATION below is that minus sign.
    #
    # The old code took jφ = +jphi/R on the strength of
    # `∫(jphi/R)dA = 2.41250 MA = |toroidal_current|` — an ABSOLUTE-value check,
    # structurally incapable of catching the inversion. Cross-checked two ways
    # after the fix: the curl of the exported B gives J_φ(axis) = −1.4229e6 A/m²,
    # and the Grad–Shafranov source R·p′ + F·F′/(μ₀R) gives −1.4166e6 (0.4%).
    # This is a plain bug fix, independent of COCOS, so it applies on every
    # output path including `cocos = nothing` and `:mhdsimdb`.
    #
    # IMAS defines profiles_1d.j_tor as ⟨jφ/R⟩/⟨1/R⟩ (not the plain FSA), so it
    # is assembled from two reductions sharing the mask/patch/weights; the R
    # normalization cancels in the ratio, leaving a pure `:current_density`
    # conversion.
    jtor1d = nothing
    let jphi_rz = field_rz(:jphi)
        if jphi_rz !== nothing
            nRg = length(Rg_n);  nZg = length(Zg_n)
            jn = [jphi_rz[i, j] / Float64(Rg_n[i])^2 for i in 1:nRg, j in 1:nZg]  # jφ/R
            inv_r = [1.0 / Float64(Rg_n[i]) for i in 1:nRg, j in 1:nZg]           # 1/R
            jp = field_patch(:jphi)
            jn_p = inv_r_p = nothing
            if Ra !== nothing && jp !== nothing
                na = length(Ra)
                Rcol = vec([Float64(Ra[i]) for i in 1:na, _ in 1:na])
                jn_p = jp ./ Rcol .^ 2
                inv_r_p = 1.0 ./ Rcol
            end
            rj = reduce_raw(jn, jn_p)
            ri = reduce_raw(inv_r, inv_r_p)
            if rj !== nothing && ri !== nothing
                # −1: jphi = +Δ*ψ = −μ₀R·J_φ (see above)
                ujd = -unit_factor(norm, :current_density; system = :si)
                jtor1d = [
                    (isfinite(a) && isfinite(b) && b != 0) ? a / b * ujd : NaN
                        for (a, b) in zip(rj.func_bin, ri.func_bin)
                ]
                # under `:bin` the innermost ρ shell can hold no samples at all
                # (same empty-bin case `F1d` hits) — rebuild only those nodes
                _fill_axis_band!(jtor1d, ρgrid .^ 2; only_nonfinite = true)
            end
        end
    end

    # axisymmetric B components (needs :I): B = ∇ψ×∇φ + I∇φ with per-radian ψ,
    # so B_R = −ψ_Z/R, B_Z = +ψ_R/R, Bφ = I/R (M3D-C1 m3dc1_nint.f90:1172 uses
    # exactly |B|² = (ψ_R²+ψ_Z²+I²)/R²). ⟨|B|⟩ is a plain weighted FSA (ratio —
    # the axis patch is fine here, unlike V′); 2D maps are returned in Tesla.
    br_rz = bz_rz = bphi_rz = nothing;  babs1d = nothing
    I_rz = field_rz(:I)
    if I_rz !== nothing
        ub = unit_factor(norm, :magnetic_field; system = :si)
        nRg = length(Rg_n);  nZg = length(Zg_n)
        br_n = [-pg.dZ[i, j] / Float64(Rg_n[i]) for i in 1:nRg, j in 1:nZg]
        bz_n = [ pg.dR[i, j] / Float64(Rg_n[i]) for i in 1:nRg, j in 1:nZg]
        bphi_n = [ I_rz[i, j] / Float64(Rg_n[i]) for i in 1:nRg, j in 1:nZg]
        babs_n = sqrt.(br_n .^ 2 .+ bz_n .^ 2 .+ bphi_n .^ 2)
        babs_patch = nothing
        if pga !== nothing
            Ia = field_patch(:I)
            if Ia !== nothing
                na = length(Ra)
                Rcol = vec([Float64(Ra[i]) for i in 1:na, _ in 1:na])
                babs_patch = sqrt.(
                    (vec(pga.dZ) ./ Rcol) .^ 2 .+
                        (vec(pga.dR) ./ Rcol) .^ 2 .+ (Ia ./ Rcol) .^ 2
                )
            end
        end
        rB = reduce_raw(babs_n, babs_patch)
        babs1d = rB === nothing ? nothing : rB.func_bin .* ub
        br_rz = br_n .* ub;  bz_rz = bz_n .* ub;  bphi_rz = bphi_n .* ub
    end

    # ── E∥, ⟨J·B⟩, σ∥, Zeff, ohmic power ─────────────────────────────────────
    # All are ordinary surface averages of fields that genuinely vary ON a flux
    # surface, so (unlike F/FF′/p′) they follow `fsa_method` like te/ne.
    #
    # `E_par` is M3D-C1's own E·B/|B|. Cross-check on the JET SPI slice 0: at the
    # axis B is essentially pure toroidal with Bφ = F/R < 0, so E∥ must equal
    # −E_φ — measured +0.17965 vs −0.17956 V/m (0.05%), which validates the
    # :electric_field unit factor independently of this path. Note E_φ itself is
    # NOT usable as a flux-surface quantity: it mixes the inductive and v×B
    # parts, which nearly cancel, so R·E_φ (a flux function in ideal MHD) was
    # measured to vary 4–10% poloidally already at t=0 and >100% later, while
    # E∥ stays smooth and physical throughout.
    epar1d = prof(:E_par)



    # δB/B: plane-sampled toroidal-fluctuation map (needs :I; f′ = ∂f/∂φ
    # carries the non-axisymmetric part of B — see docs/deltab_over_b.md and
    # deltab_over_b_rz). Modern files store f′ as the `fp` dataset; legacy
    # ifprime=0 files store only `f`, whose ∂/∂φ is the second ζ-layer.
    # Dimensionless, so the FSA needs no unit factor; single-plane
    # (axisymmetric) runs correctly give an all-zero profile.
    deltab1d = nothing
    if haskey(fields, :I)
        f3 = get(fields, :fp, nothing)
        fprime_f = f3 !== nothing
        fprime_f || (f3 = get(fields, :f, nothing))
        dbm = deltab_over_b_rz(
            fields[:psi], fields[:I], f3,
            nplanes, ep, Rg_n, Zg_n; id_map = id_map, fprime = fprime_f
        )
        # axis patch fills the innermost ρ bins (ratio FSA — patch is fine,
        # same as ⟨|B|⟩; NIMROD's profile is finite at every node)
        db_patch = nothing
        if Ra !== nothing
            dbp = deltab_over_b_rz(
                fields[:psi], fields[:I], f3,
                nplanes, ep, Ra, Za; id_map = ida, fprime = fprime_f
            )
            db_patch = vec(dbp.db)
        end
        rdb = reduce_raw(dbm.db, db_patch)
        deltab1d = rdb === nothing ? nothing : rdb.func_bin
    end

    # 2D toroidal-flux map Φ(ψ(R,Z)) from the 1D Φ profile (confined region
    # only — Φ is undefined on open surfaces, so ψ_N > 1 stays NaN)
    phi_rz = nothing
    if phi1d !== nothing
        nRg = length(Rg_n)
        phi_rz = [
            (
                    ψnv = ψn[i + (j - 1) * nRg];
                    (isfinite(ψnv) && 0.0 <= ψnv <= 1.0) ?
                    _lininterp(ρgrid, phi1d, sqrt(ψnv)) : NaN
                )
                for i in 1:nRg, j in 1:length(Zg_n)
        ]
    end

    # recomputed X-points (SI), for downstream consumers (boundary IDS, masks)
    x_points = NTuple{2, Float64}[]
    x_mesh = NTuple{2, Float64}[]                # same points, mesh units
    if lc !== nothing
        for x in (lc.x1, lc.x2)
            (x !== nothing && x.converged && x.kind === :saddle) || continue
            push!(x_points, (x.R * ulen, x.Z * ulen))
            push!(x_mesh, (Float64(x.R), Float64(x.Z)))
        end
    end

    # LCFS outline → equilibrium…boundary.outline. Traced on the exact FEM ψ by
    # [`trace_lcfs`](@ref) (rays from the axis, not a contour of `psi_rz`), so
    # it is independent of `ngrid` and needs no private-flux masking. Validated
    # against the run's own EFIT boundary on JET SPI slice 0: mean radial
    # deviation 1.0 mm = 0.09% of the minor radius (DIII-D 0.05%, KSTAR 0.20%).
    # `nothing` on the degenerate slices where ψ_axis == ψ_boundary (the plasma
    # is gone) — those already carry no `q1d` either.
    lcfs_rz = nothing
    if lc !== nothing && isfinite(ψ1 - ψ0) && ψ1 != ψ0
        tr = try
            trace_lcfs(
                psi_axisym_coef, ep, (R_ax, Z_ax), ψ0, ψ1;
                ntheta = lcfs_ntheta, x_points = x_mesh,
                id_map = id_map, R_grid = Rg_n, Z_grid = Zg_n
            )
        catch err
            @warn "reduce_axisym_slice: LCFS outline trace failed; boundary.outline omitted" exception = err
            nothing
        end
        if tr !== nothing
            tr.nmiss == 0 ||
                @warn "reduce_axisym_slice: LCFS outline has $(tr.nmiss)/$(lcfs_ntheta - 1) unresolved angles (rays left the mesh); outline chords across the gaps"
            lcfs_rz = (; R = tr.R .* ulen, Z = tr.Z .* ulen)
        end
    end

    return (;
        time = Float64(time_s),
        rho = ρgrid, psi1d = psi1d,
        te = prof(:te), ne = prof(:ne), ti = prof(:ti),
        ni = prof(:den), pressure = prof(:P),
        imp_dens = imp_dens, prad1d = prad1d,
        F1d = F1d, q1d = q1d, phi1d = phi1d, babs1d = babs1d,
        jtor1d = jtor1d, ffprime1d = ffprime1d, pprime1d = pprime1d,
        epar1d = epar1d,
        deltab1d = deltab1d, lcfs_rz = lcfs_rz,
        psi_rz = Array{Float64}(psi_rz) .* uflux,
        br_rz = br_rz, bz_rz = bz_rz, bphi_rz = bphi_rz, phi_rz = phi_rz,
        Rg = collect(Float64, Rg_n) .* ulen, Zg = collect(Float64, Zg_n) .* ulen,
        psi_axis = ψ0 * uflux, psi_boundary = ψ1 * uflux,
        R_axis = R_ax * ulen, Z_axis = Z_ax * ulen,
        x_points = x_points,
    )
end

# ψ-derivative FSA tuning. `_GRAD_PSI_FLOOR` drops the |∇ψ|→0 neighbourhood of
# the magnetic axis, where df/dψ = ∇f·∇ψ/|∇ψ|² is numerically 0/0 (threshold is
# relative to the slice's own max |∇ψ|², so it scales with the equilibrium).
# `_AXIS_FILL_*` then rebuild the innermost nodes from the well-resolved band.
const _GRAD_PSI_FLOOR = 1.0e-6
const _AXIS_FILL_LO = 0.01
const _AXIS_FILL_BAND = (0.01, 0.30)

# Rebuild the innermost ψ_N nodes of a ψ-derivative profile from a polynomial
# fit of the well-resolved band. Two effects need this: the axis node is 0/0
# (see `_GRAD_PSI_FLOOR`), and the binning kernel is one-sided at ρ→0 (no ρ<0
# samples) with a volume weight ∝ρ, so the axis-adjacent nodes are biased
# outward. The bias is an estimator edge effect, not a discretization error —
# measured at 0.6% of the FF′ scale at ρ=0, decaying to zero by ψ_N ≈ 0.005,
# and *identical* at ngrid 200 and 400. A cubic extrapolation over the
# innermost 1% removes it (FF′ rms over ψ_N<0.02: 0.55% → 0.083%; over the
# whole profile 0.118% → 0.036%, vs the run's own EFIT input). Note the EFIT
# reference is itself unreliable at that node — its ffprim is an extrapolation
# of the fit basis, e.g. 2.518 next to 2.064 one node in on the KSTAR case,
# where the extrapolated value here continues the smooth trend instead.
function _fill_axis_band!(
        v::Vector{Float64}, psin::AbstractVector,
        lo::Real = _AXIS_FILL_LO, band::Tuple{<:Real, <:Real} = _AXIS_FILL_BAND;
        deg::Integer = 3, only_nonfinite::Bool = false
    )
    tgt = [k for k in eachindex(v) if psin[k] < lo && (!only_nonfinite || !isfinite(v[k]))]
    isempty(tgt) && return v
    idx = [k for k in eachindex(v) if band[1] <= psin[k] <= band[2] && isfinite(v[k])]
    length(idx) >= deg + 2 || return v          # too sparse to extrapolate safely
    x = Float64[psin[k] for k in idx]
    A = [x[i]^(j - 1) for i in eachindex(x), j in 1:(deg + 1)]
    c = try
        A \ v[idx]
    catch err
        @debug "_fill_axis_band!: fit failed; leaving the axis nodes as-is" exception = err
        return v
    end
    all(isfinite, c) || return v
    @inbounds for k in tgt
        v[k] = sum(c[j] * psin[k]^(j - 1) for j in 1:(deg + 1))
    end
    return v
end

# V′(ψ) at the ρ nodes from the exact cumulative confined volume: per-bin
# `den` counting is shot-noise-limited (~1/√(samples/bin)), but the cumulative
# V(ψ_N) from sorted samples is exact up to cell quantization — only the
# differentiation needs a smoothing choice. Local quadratic regression in ψ_N
# over an adaptive window (`wfac` node spacings Δx≈2ρh, floored at `wmin`)
# gives V′ = the linear coefficient; the one-sided window at the axis
# reproduces the V ≈ c₁ψ_N + c₂ψ_N² behaviour the previous axis-only fit
# hard-coded (q₀ validated +0.1..0.8% vs IMAS).
function _vprime_cumvol(
        x::Vector{Float64}, V::Vector{Float64},
        ρnodes, Δψ_si::Real;
        wmin::Real = 0.05, wfac::Real = 16.0
    )
    Vp = fill(NaN, length(ρnodes))
    hρ = length(ρnodes) > 1 ? Float64(ρnodes[2] - ρnodes[1]) : 0.01
    for (k, ρ) in enumerate(ρnodes)
        xk = Float64(ρ)^2
        w = max(Float64(wmin), Float64(wfac) * Float64(ρ) * hρ)
        i1 = searchsortedfirst(x, xk - w);  i2 = searchsortedlast(x, xk + w)
        i2 - i1 + 1 < 25 && continue
        S1 = S2 = S3 = S4 = T0 = T1 = T2 = 0.0
        @inbounds for i in i1:i2
            d = x[i] - xk;  Vi = V[i]
            S1 += d;      S2 += d^2;     S3 += d^3;    S4 += d^4
            T0 += Vi;     T1 += Vi * d;  T2 += Vi * d^2
        end
        N = Float64(i2 - i1 + 1)
        β = try
            [N S1 S2; S1 S2 S3; S2 S3 S4] \ [T0, T1, T2]
        catch
            continue
        end
        Vp[k] = β[2] / Float64(Δψ_si)
    end
    return Vp
end

# Smooth cumulative flux-surface average: ⟨f⟩(ψ) = W′(ψ)/V′(ψ), the
# volume-weighted FSA written as the ratio of two cumulative-volume derivatives
# — the same integrate-then-differentiate trick that de-noises q via
# `_vprime_cumvol`. W(ψ_N)=∫_{ψ'<ψ} f dV and V(ψ_N)=∫_{ψ'<ψ} dV are exact from
# sorted samples + cumsum; each is differentiated by local quadratic regression,
# so the per-bin shot noise (∝1/√samples-per-bin) of `reduce_1d_psi_func` is
# replaced by the far lower density-estimation floor. The Δψ scale and the ρ↔ψ_N
# Jacobian cancel in the W′/V′ ratio, so both derivatives are taken vs ψ_N with
# unit scale. `ρ_all` are ρ_pol = √ψ_N sample coordinates; `values`/`w_all` carry
# NaN for dropped (off-mesh / private-flux) samples. Returns the same
# (psi_grid, func_bin, den) shape as `reduce_1d_psi_func` (den = V′), or `nothing`
# when too few confined samples remain (< 25, the regression floor).
function _fsa_cumulative(
        ρ_all::AbstractVector, values::AbstractVector,
        w_all::AbstractVector, ρgrid::AbstractVector; wfac::Real = 4.0
    )
    n = length(ρ_all)
    (n == length(values) && n == length(w_all)) ||
        error("_fsa_cumulative: ρ/value/weight length mismatch")
    xs = Float64[];  fw = Float64[];  ww = Float64[]
    sizehint!(xs, n);  sizehint!(fw, n);  sizehint!(ww, n)
    @inbounds for i in 1:n
        ρi = ρ_all[i];  fi = values[i];  wi = w_all[i]
        (isfinite(ρi) & isfinite(fi) & isfinite(wi)) || continue
        push!(xs, Float64(ρi)^2)              # ψ_N = ρ_pol²
        push!(fw, Float64(fi) * Float64(wi))  # f·dV integrand
        push!(ww, Float64(wi))                # dV integrand
    end
    length(xs) ≥ 25 || return nothing
    p = sortperm(xs)
    xss = xs[p]
    Wc = cumsum(fw[p])                          # cumulative ∫ f dV
    Vc = cumsum(ww[p])                          # cumulative ∫ dV
    Wp = _vprime_cumvol(xss, Wc, ρgrid, 1.0; wfac = wfac)
    Vp = _vprime_cumvol(xss, Vc, ρgrid, 1.0; wfac = wfac)
    func_bin = [
        (isfinite(Wp[k]) && isfinite(Vp[k]) && Vp[k] > 0) ? Wp[k] / Vp[k] : NaN
            for k in eachindex(ρgrid)
    ]
    return (; psi_grid = collect(Float64, ρgrid), func_bin, den = Vp)
end

const _OPT_FIELDS = (:te, :ne, :ti, :den, :P, :E_par)

# Coefficient-row range per field: everything is evaluated at a plane (ζ=0, rows
# 1:20); the sole exception is legacy `f`, whose ∂/∂φ (δB/B) needs rows 21:40 →
# read 1:40. Hoisted to a named function so the per-slice read passes a constant
# (no per-iteration closure).
_export_slice_rows(fld::Symbol) = fld === :f ? (1:40) : (1:20)

# All export FEM evaluation is at a plane (ζ=0), which touches only rows 1:20;
# the sole exception is legacy-`f` δB/B, whose ∂/∂φ is rows 21:40. Reading just
# that leading block (HDF5 hyperslab) cuts the read volume and the Float32→
# Float64 conversion (~3× faster / 4× less memory per field).
function _try_field(file, ts, fld; rows = 1:20)
    try
        return read_field(file, ts, fld; rows = rows)
    catch err
        @debug "export_imas: field $fld unavailable for slice $ts" exception = err
        return nothing
    end
end

# COCOS 3 → COCOS 11 conversion of one per-slice result.
#
# **M3D-C1's native data is COCOS 3, not COCOS 1.** The distinction is σ_Bp:
# M3D-C1 writes B = +∇ψ×∇φ + F∇φ (⇒ B_R = −ψ_Z/R, B_Z = +ψ_R/R), while COCOS
# places σ_Bp in front of the *reversed* cross product — OMAS `omas_physics.py`
# and MXHEquilibrium `fields.jl` both spell it B_R = +σ_RφZ·σ_Bp·(∂ψ/∂Z)/((2π)^e_Bp R).
# So M3D-C1 is σ_Bp = −1, and operationally sign(ψ_edge − ψ_axis) = −sign(Ip) on
# every run. Confirmed three independent ways: OMAS `identify_cocos` on native
# (b0, Ip, q, ψ) returns [4, 14, 3, 13] for JET / DIII-D / ITER cases alike; a
# pointwise check of the *previously exported* file found its ψ map and its own
# `b_field_r`/`b_field_z` arrays anti-parallel (median ratio −0.9998,
# correlation −0.99999988 over 30 476 grid points); and M3D-C1's own
# `iflip_j` ("flip equilibrium toroidal current density") is implemented as
# `mult(psi_field(0), -1.)` — reversing Ip *is* negating ψ.
#
# The transform is therefore `cocos_transform(3, 11)` applied verbatim:
#     PSI              ×(−2π)      ψ decreases outward for Ip > 0 in COCOS 11
#     Q                ×(−1)       σ_ρθφ flips, giving sign(q) = sign(Ip·B0)
#     F_FPRIME/PPRIME  ×(−1/2π)    one inverse power of ψ ⇒ ψ's sign and scale
#     IP, BT, F, TOR   ×(+1)       ip, b0, F, Φ, ρ_tor are untouched
# `pressure`, the B maps, ⟨|B|⟩, E∥ and all geometry are frame quantities or
# invariants and do not transform. Partial fixes are invalid: flipping only q
# lands on COCOS 7/17, flipping only ψ lands on COCOS 5/15.
#
# `q1d` is flipped HERE and not upstream: `phi1d` is built from `q1d` as
# Φ = 2π∫q dψ in the native frame and must keep its native value (Φ is a TOR
# quantity, ×(+1)). The identity survives the relabelling exactly, because ψ
# flips with q: dΦ/dψ₁₁ = (dΦ/dψ_nat)·(−1/2π) = −q_nat = q₁₁.
function _to_cocos11_slice(s::NamedTuple)
    s = merge(
        s, (;
            psi1d = s.psi1d .* (-2π), psi_rz = s.psi_rz .* (-2π),
            psi_axis = s.psi_axis * (-2π), psi_boundary = s.psi_boundary * (-2π),
        )
    )
    for (k, f) in ((:ffprime1d, -2π), (:pprime1d, -2π), (:q1d, -1.0))
        v = get(s, k, nothing)
        v === nothing || (s = merge(s, NamedTuple{(k,)}((v ./ f,))))
    end
    return s
end

# One-line provenance note naming the convention a file was written in.
_cocos_comment(cc) =
    cc == 11 ?
    "M3DC1Reader FSA export; COCOS-11 via cocos_transform(3,11): psi = -2pi*psi_M3DC1 [Wb], q *= -1, f_df_dpsi/dpressure_dpsi /= -2pi; ip = M3D-C1 toroidal_current unmodified; b0 signed like F" :
    cc === :mhdsimdb ?
    "M3DC1Reader FSA export; MHDsimDB drop-in layout (equilibrium psi per-radian = M3D-C1 native COCOS-3, sigma_Bp=-1; core_profiles/disruption grid.psi = 2pi*(psi-psi_axis) NIMROD-style; ip = toroidal_current unmodified)" :
    "M3DC1Reader FSA export; raw M3D-C1 per-radian convention = COCOS 3 (sigma_Bp = -1), not COCOS-11"

# MHDsimDB drop-in layout (deliberately NOT a single COCOS — it mirrors the
# NIMROD-derived files, reverse-engineered in docs/cocos_conventions.md):
# the equilibrium IDS keeps M3D-C1's native per-radian ψ — which is a valid,
# self-consistent **COCOS 3** (σ_Bp = −1; its sign already agrees with the
# `b_field_r`/`b_field_z` arrays written beside it), so it is deliberately NOT
# given the COCOS-11 sign flips. The core_profiles/disruption grids carry the
# NIMROD-native 2π·(ψ − ψ_axis) — total-flux scale, zero at the axis.
# NOTE (unresolved): with σ_Bp = −1 that grid DECREASES for Ip > 0 (measured
# 0 → −1.7445 on C1_154127). Whether the NIMROD reference files run the other
# way has to be settled by reading one, not by argument; if so this needs
# 2π·(ψ_axis − ψ). `rho_pol_norm` is written alongside and is unaffected.
_to_mhdsimdb_slice(s::NamedTuple) =
    merge(s, (; grid_psi1d = 2π .* (s.psi1d .- s.psi_axis)))

"""
    export_imas(file, out_path; slices=list_timeslices(file), nbins=128,
                ngrid=200, adj=:linear, fsa_method=:bin, fsa_window=4.0, comment="",
                recompute_ne=false, cocos=11, pulse=nothing,
                ascot5=false, ascot5_nphi=0, ascot5_dir="",
                cocos11_path="", verbose=false) -> out_path

Compute FSA 1D profiles (Te, ne, Ti, ni, pressure) + the 2D ψ map for `slices`
and write an OMAS-compatible IMAS HDF5 file at `out_path`. When the KPRAD model
is active, also exports impurity charge-state densities (neutral + ion[1..Z]) and
the radiated-power-density profile (`disruption` IDS). Set `recompute_ne=true` to
overwrite the electron density with the quasi-neutral sum n_main + Σ_iz iz·n_imp,iz.

`fsa_method` picks the ratio-profile estimator: `:bin` (default, per-bin kernel
average) or `:cumulative` (smoother `W′/V′` cumulative estimator that removes the
bin-to-bin shot noise). See [`reduce_axisym_slice`](@ref) and [`fsa_imas`](@ref).

`ascot5=true` additionally writes one ASCOT5 input HDF5 per exported slice
(`ascot_input_<idx>.h5`, via [`write_ascot5`](@ref)), **reusing each slice's FSA**
— so the ASCOT5 `plasma_1D` background is identical to the IMAS `core_profiles`
and is not recomputed. Only the 3D `B_3DS`/`E_3D` fields are computed fresh (the
axisymmetric export never produces them). `ascot5_nphi` sets the field toroidal
resolution (`0` → `4·nplanes`); `ascot5_dir` sets the output directory (default:
the IMAS output's directory, created if new — the basename stays index-based).

Each `equilibrium…time_slice` carries, besides the ψ/q/pressure/Φ profiles and
the 2D maps: `profiles_1d.f_df_dpsi` and `profiles_1d.dpressure_dpsi` (FF′ and
p′, from the exact FEM ∇f·∇ψ/|∇ψ|² rather than a differentiated profile — see
[`reduce_axisym_slice`](@ref)), `profiles_1d.j_tor` (the IMAS
⟨jφ/R⟩/⟨1/R⟩, needs the `:jphi` field), `profiles_1d.rho_tor` /
`rho_tor_norm` (√(Φ/πb0), needs `phi1d` and `vacuum_toroidal_field.b0`), and
`boundary.outline.{r,z}` + `boundary.psi` — the LCFS traced on the FEM ψ by
[`trace_lcfs`](@ref) (grid-independent, private-flux region excluded by
construction). These are the paths the DREAM disruption-simulation setup reads
(alongside `boundary.x_point`).

Each `core_profiles.profiles_1d` also carries `e_field.parallel` — M3D-C1's own
`E_par` field, which `electric_field.f90` builds as E·B/|B| from the same ψ/I
quadrature arrays as everything else (Ohm's law there is E = +ηJ − v×B, and the
divisor `b2` expands term-by-term to B_R²+B_Z²+B_φ² for B_R=−ψ_Z/R, B_Z=+ψ_R/R,
B_φ=F/R). It is written with **no sign change**: dotting the stored
(E_R,E_PHI,E_Z) with that same B reproduces the stored `E_par` to 0.3%, and
against an Ampère-frame J it gives η = E∥/J∥ = 1.257e−7 Ω·m, matching M3D-C1's
own ⟨1/η⟩ to 0.3%. NOT E_φ, which mixes the inductive and v×B parts that nearly
cancel — R·E_φ was measured to vary 4–10% poloidally already at t = 0 and >100%
later, while E∥ stays smooth. `core_profiles.vacuum_toroidal_field.{r0,b0}` is
written alongside.

When the `:I` field is present, each `equilibrium…profiles_2d.0` also gets the
axisymmetric field maps `b_field_r/z/tor` and the toroidal-flux map `phi` [T,
Wb per-radian], and each `core_profiles.profiles_1d` gets the FSA
`custom.b_field_torus_average` = ⟨|B|⟩ [T] (the path the legacy fusion-io
mapper used for this ML feature).

`cocos = 11` (default) writes the file in the IMAS COCOS-11 convention.
**M3D-C1's native data is COCOS 3** (σ_Bp = −1 — see `_to_cocos11_slice` for the
evidence), so the conversion is `cocos_transform(3, 11)` applied verbatim:
every ψ (1D grids, 2D map, psi_axis/psi_boundary, boundary.psi) becomes
**−2π·ψ_M3D** [Wb], `q` flips sign, and the ψ-derivatives (`f_df_dpsi`,
`dpressure_dpsi`) are divided by **−2π**. `ip` is M3D-C1's `toroidal_current`
**unmodified** (IP is invariant under every 3→11 transform), and
`vacuum_toroidal_field.b0` is signed like F because the `bzero` attribute is an
unsigned magnitude. F, Φ, ρ_tor, p, E∥ and the B maps do not transform. The
result satisfies sign(q) = sign(Ip·B0) with the *physical* Ip, and its ψ map is
sign-consistent with the `b_field_r`/`b_field_z` arrays beside it.

`cocos = nothing` keeps the raw M3D-C1 per-radian convention — which is a
self-consistent **COCOS 3** file, not COCOS 1.

`cocos11_path` (when non-empty) additionally writes the **same export** in the
COCOS-11 convention to that second path — useful when the primary output has to
stay in the `:mhdsimdb` drop-in layout but a consumer wants the IMAS-standard
one. The per-slice FSA is convention-free, so the twin re-uses the same
reduction: it costs one transform + one write, not a second pass over the
multi-GB slice files. Returns `(out_path, twin_path)` in that case.

`cocos = :mhdsimdb` mirrors the **existing NIMROD-derived MHDsimDB files** so
the output is a drop-in row next to them (for the Python
`disruption_database_tools` consumers): the `equilibrium` IDS keeps M3D-C1's
native per-radian ψ/q/FF′/p′ (a valid COCOS 3 — no sign flips) while the
`core_profiles`/`disruption` `grid.psi` carries NIMROD-native 2π·(ψ − ψ_axis)
(total-flux scale, axis-zeroed); `b0` gets the same sign-from-F fix as
`cocos = 11` and `ip` is written unmodified. This layout is deliberately
**not one COCOS** — it reproduces the DB's mixed conventions (see
`docs/cocos_conventions.md`).

Machine-description / source IDS (all in the NIMROD-DB layout): the `wall`
IDS carries the M3D-C1 mesh boundary as the limiter outline
([`mesh_boundary_rz`](@ref) — usable as an ASCOT5 wall); when the run has a
pellet model (`/pellet` root group, [`read_pellets`](@ref)), the `pellets`
IDS gets per-slice, per-pellet `path_profiles.{ablation_rate,
ablated_particles, distance, position.{r,z,phi}}`, `shape.size`,
`velocity_initial`, and the
two-species list `species.0 = D` / `species.1 = KPRAD impurity` split by
`pellet_mix` (the NIMROD SPI layout); recomputed X-points
land in `equilibrium…boundary.x_point[*]`; `pulse` (when given) fills
`dataset_description.data_entry.pulse`.

`verbose = true` logs progress via `@info`: a header (file, slice count, mesh
and grid size, KPRAD/fprime/cocos), one line per slice as its FSA finishes
(slice index, `ntimestep`, physical time, wall time), and a final line with
the output path and total wall time.

0D global traces (`scalars/*`, sampled at each exported slice's `ntimestep`,
SI): `summary.global_quantities.{ip, energy_thermal, v_loop, power_radiated}
.value` from `toroidal_current` / `W_P` / `loop_voltage` / `radiation`, the
per-slice `equilibrium…global_quantities.ip`, and `disruption
.global_quantities.{power_radiated_total, power_ohm}` (P_Ω = Ip·V_loop). Traces
a file lacks are omitted. See `_GLOBAL_SCALARS` for why each dataset was chosen.

!!! warning "Radiation-sum definition (pending confirmation)"
    M3D-C1's `radiation` scalar is the electron-temperature *sink*
    Σ(line + brem + ion_loss + reck) — it counts ionization losses but not the
    recombination *potential* energy carried off as photons (`recp_rad`), and
    it is stored *negative* (a loss term; e.g. −2.5 GW at the SPI thermal
    quench). Pure photon power would be line + brem + recp instead. Which sum
    (and sign) the disruption database wants for
    `power_radiated`/`power_radiated_total` is not yet confirmed (ask Val);
    until then the `radiation` trace is written as-is. All six component
    traces (`radiation, line_rad, brem_rad, ion_loss, reck_rad, recp_rad`)
    exist in `scalars/*` via [`read_scalar`](@ref) (quantity `:power`), so the
    definition can be switched without re-running.
"""
function export_imas(
        file::M3DC1File, out_path::AbstractString;
        slices = list_timeslices(file), nbins::Integer = 128,
        ngrid::Integer = 200, adj::Symbol = :linear, fsa_method::Symbol = :bin,
        fsa_window::Real = 4.0,
        comment::AbstractString = "", recompute_ne::Bool = false,
        cocos::Union{Nothing, Integer, Symbol} = 11,
        pulse::Union{Nothing, Integer} = nothing,
        ascot5::Bool = false, ascot5_nphi::Integer = 0,
        ascot5_dir::AbstractString = "",
        cocos11_path::AbstractString = "",
        verbose::Bool = false
    )
    cocos === nothing || cocos == 11 || cocos === :mhdsimdb ||
        throw(
        ArgumentError(
            "export_imas: cocos must be 11 (IMAS), :mhdsimdb (NIMROD-DB drop-in layout), " *
                "or nothing (raw M3D-C1 per-radian), got $cocos"
        )
    )
    # validated up front: the slice loop below is the multi-GB read, and an
    # argument error must not cost it (nor leave a half-finished primary file)
    (!isempty(cocos11_path) && cocos == 11) &&
        throw(ArgumentError("cocos11_path is redundant when cocos = 11 (that IS the output)"))
    norm = normalization(file)
    ep = elems_plane(file)
    Rg_n = collect(range(extrema(ep[5, :])..., length = ngrid))
    Zg_n = collect(range(extrema(ep[6, :])..., length = ngrid))
    id_map = build_grid_to_element_map(Rg_n, Zg_n, ep)
    utime = unit_factor(norm, :time; system = :si)

    # KPRAD impurity / radiation fields to read (only when the model is active)
    kprad_z = _kprad_z(file)
    kfields = Symbol[]
    if kprad_z ≥ 0
        append!(kfields, (Symbol("kprad_n_" * lpad(iz, 2, '0')) for iz in 0:kprad_z))
        append!(kfields, (:kprad_rad, :kprad_brem, :kprad_reck, :kprad_recp))
    end

    lim = limiter_points(file)          # file-level LCFS limiter candidates
    fprime = _fprime_stored(file)       # modern files carry f′ as `fp`
    # :I → F+q+B; f′ (fp, or legacy f) → δB/B. This field set is fixed across the
    # loop (kprad_z / fprime are file-level), so ψ + all optional fields are read
    # into persistent buffers reused every slice — the ~28-MiB coefficient
    # matrices are allocated once, not per slice (see read_timeslice!).
    ffield = fprime ? :fp : :f
    opt_flds = (_OPT_FIELDS..., :I, :jphi, ffield, kfields...)
    fbuf = Dict{Symbol, Matrix{Float64}}()   # reused Float64 field buffers (= sl.fields)
    sbuf = Dict{Symbol, Matrix{Float32}}()   # reused Float32 hyperslab staging
    results = NamedTuple[]
    nsteps = Int[]                      # each slice's ntimestep (0D trace sampling)
    nsl = length(slices)
    # compact duration string for progress logs: "8.4s" or "2m05s"
    _fmt_dur(s) = s < 60 ? "$(round(s, digits = 1))s" :
        "$(floor(Int, s / 60))m$(lpad(round(Int, s % 60), 2, '0'))s"
    if verbose
        rlo, rhi = extrema(ep[5, :]);  zlo, zhi = extrema(ep[6, :])
        kp = kprad_z ≥ 0 ? "$(kprad_z + 1) charge states" : "off"
        @info "export_imas: $(basename(String(file.path)))" slices = nsl mesh = "$(size(ep, 2)) elems × $(file.nplanes) planes" grid = "$(ngrid)×$(ngrid)  R∈[$(round(rlo, digits = 3)), $(round(rhi, digits = 3))]  Z∈[$(round(zlo, digits = 3)), $(round(zhi, digits = 3))]" kprad = kp fprime = fprime cocos = cocos ascot5 = ascot5 out = out_path
        println(stderr)                 # blank line: detailed setup ┄ terse per-slice progress
    end
    t_start = time()
    t_warm = t_start                    # ETA baseline; reset after slice 1 (compile)
    kw = ndigits(nsl)                   # column widths (we know nsl and every ts up front)
    tw = isempty(slices) ? 1 : maximum(ndigits, slices)
    # ASCOT5 output dir: explicit ascot5_dir, else next to the IMAS output (which
    # for the default out_path is the run folder = next to C1.h5). Created if new.
    a5dir = !isempty(ascot5_dir) ? String(ascot5_dir) :
        let d = dirname(String(out_path))
            isempty(d) ? dirname(String(file.path)) : d
    end
    ascot5 && mkpath(a5dir)
    for (isl, ts) in enumerate(slices)
        t_sl = time()
        # One open per slice reads ψ + every optional field together into the
        # reused buffers. The C1.h5 slice group is an external link into a
        # multi-GB time_XXX.h5, so re-opening per field would re-resolve that
        # link each time; absent optional fields are skipped silently.
        sl = read_timeslice!(
            fbuf, sbuf, file, ts; required = (:psi,),
            optional = opt_flds, rows = 1:20, rows_of = _export_slice_rows
        )
        push!(nsteps, sl.nstep)
        fields = sl.fields
        res = reduce_axisym_slice(
            fields, ep, file.nplanes, norm,
            sl.psi_axis, sl.psi_lcfs, sl.xmag, sl.zmag,
            sl.time * utime, Rg_n, Zg_n, id_map;
            nbins = nbins, adj = adj, fsa_method = fsa_method,
            fsa_window = fsa_window, kprad_z = kprad_z,
            sl.xnull, sl.znull, sl.xnull2, sl.znull2,
            lim.xlim, lim.zlim, lim.xlim2, lim.zlim2
        )
        push!(results, res)
        # Optional ASCOT5 input per slice, reusing THIS slice's FSA (`res`) so the
        # plasma_1D background is byte-identical to the IMAS core_profiles. Only the
        # 3D B-field is computed fresh — the axisymmetric export never produces it,
        # so it cannot be reused (it dominates the per-slice cost either way).
        if ascot5
            bf = ascot5_bfield(file, ts; nphi = ascot5_nphi, efield = true)
            apath = _ascot5_default_path(file, ts; dir = a5dir)
            _write_ascot5_hdf5(
                apath, file, ep, norm, kprad_z, bf, res,
                _ascot5_default_desc(file, ts, bf)
            )
        end
        if verbose
            # one short, column-aligned line per slice: [k/n], ts, physical time,
            # the wall time this slice took, and a running ETA. lpad handles the
            # width-variable integers; @sprintf fixes the float decimals so the
            # columns line up. slice 1 carries the JIT cost, so the ETA baseline
            # (`t_warm`) starts from slice 2.
            # remaining-time estimate on every non-last slice. slice 1 has only its
            # own (JIT-heavy) time to go on, so its ETA is a rough upper bound; from
            # slice 2 the warm average (excluding slice 1) makes it accurate.
            per = isl == 1 ? (time() - t_sl) : (time() - t_warm) / (isl - 1)
            isl == 1 && (t_warm = time())
            eta = isl < nsl ? "   ETA $(_fmt_dur(per * (nsl - isl)))" : ""
            prog = "[$(lpad(isl, kw))/$nsl]"
            tms = @sprintf("%6.2f", sl.time * utime * 1.0e3)
            took = @sprintf("%5.1f", time() - t_sl)
            @info "  $prog ts=$(lpad(ts, tw))   t=$tms ms   took $(took)s$eta"
        end
    end
    verbose && @info "export_imas: $nsl slices done in $(_fmt_dur(time() - t_start)); writing $(basename(String(out_path)))…"

    cmt = isempty(comment) ? _cocos_comment(cocos) : comment
    imp_label, imp_a = _impurity_species(kprad_z)
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = cmt, ion_label = "D+", z_ion = 1.0, ion_a = file.ion_mass,
        imp_label = imp_label, imp_z = max(kprad_z, 0), imp_a = imp_a,
        recompute_ne = recompute_ne, r0 = NaN, b0 = NaN,
        globals = _read_globals(file, norm, nsteps),
    )
    # vacuum field constants if present on the file root (best-effort; NaN otherwise)
    meta = _with_vacuum_field(file, norm, meta)

    # The per-slice FSA (`results`) is convention-free — a convention is only a
    # relabelling applied on the way out. So each requested output re-uses the
    # SAME reduction: the second file costs a transform + a write, not a re-read
    # of the multi-GB slice files.
    mesh_zone_f = _try_field(file, first(slices), :mesh_zone; rows = 1:1)
    function _write_variant(cc, path)
        res = cc == 11 ? NamedTuple[_to_cocos11_slice(s) for s in results] :
            cc === :mhdsimdb ? NamedTuple[_to_mhdsimdb_slice(s) for s in results] :
            results
        # field-frame ip / signed-b0 fixes apply to both labelled conventions
        m = (cc === nothing) ? meta : _cocos11_meta(meta, res)
        # the primary keeps an explicit user comment verbatim; the twin gets the
        # convention appended, so a COCOS-11 file is never labelled as if it were
        # the (differently-scaled) primary
        m = merge(
            m, (;
                comment = isempty(comment) ? _cocos_comment(cc) :
                    cc == cocos ? comment : comment * " | " * _cocos_comment(cc),
            )
        )
        ir = assemble_ir(res, m)
        # machine-description / source IDS (static or trace-driven, not per-slice FSA)
        ts_times = Float64[s.time for s in res]
        isempty(ts_times) || _wall_ir!(ir, ep, norm, ts_times[1], mesh_zone_f)
        _pellets_ir!(ir, file, norm, nsteps, ts_times, m)
        pulse === nothing || (ir["dataset_description.data_entry.pulse"] = Int(pulse))
        return write_omas_h5(path, ir)
    end

    written = _write_variant(cocos, out_path)
    extra = String[]
    isempty(cocos11_path) || push!(extra, _write_variant(11, cocos11_path))
    verbose && @info "export_imas: done — $nsl slices in $(_fmt_dur(time() - t_start)) → $written" *
        (isempty(extra) ? "" : " (+ COCOS-11 twin: $(join(extra, ", ")))")
    return isempty(extra) ? written : (written, extra...)
end

# wall IDS: the M3D-C1 computational mesh boundary as the limiter outline
# (same layout as the NIMROD MHDsimDB files: description_2d.0.limiter with a
# single unit; wall.time is a single entry — the wall is static). ASCOT5's
# wall_2D input can be built from the same outline. When the run is
# multi-region (`imulti_region=1`: the resistive wall / vacuum are meshed as
# ZONE_CONDUCTOR=2 / ZONE_VACUUM=3), the plasma-facing limiter is the plasma
# zone's boundary and the conductor zone's boundary loop(s) — its inner and
# outer surfaces — are written as `description_2d.0.vessel` units.
function _wall_ir!(ir, ep, norm, t1::Real, mesh_zone = nothing)
    ulen = unit_factor(norm, :length; system = :si)
    # mesh_zone spans all planes (plane-stacked, identical); ep is one plane
    zones = mesh_zone === nothing ? nothing : elem_zones(mesh_zone)[1:size(ep, 2)]
    multi = zones !== nothing && any(>(1), zones)

    # limiter = plasma-facing boundary (the plasma zone if multi-region, else
    # the whole computational mesh boundary via the fast row-7 flags)
    bd = try
        multi ? (
                ls = mesh_zone_boundary_rz(ep, zones, 1);
                isempty(ls) ? mesh_boundary_rz(ep) : argmax(l -> length(l.R), ls)
            ) :
            mesh_boundary_rz(ep)
    catch err
        @warn "export_imas: mesh-boundary extraction failed; wall IDS omitted" exception = err
        return nothing
    end
    length(bd.R) ≥ 3 || return nothing
    ir["wall.time"] = [Float64(t1)]
    ir["wall.description_2d.0.limiter.type.index"] = 0
    ir["wall.description_2d.0.limiter.type.name"] = "first_wall"
    ir["wall.description_2d.0.limiter.type.description"] =
        multi ? "M3D-C1 plasma-region boundary" : "M3D-C1 computational mesh boundary"
    ir["wall.description_2d.0.limiter.unit.0.outline.r"] = bd.R .* ulen
    ir["wall.description_2d.0.limiter.unit.0.outline.z"] = bd.Z .* ulen

    # vessel = the meshed resistive-wall (conductor) region boundary loops
    if multi
        vloops = try
            mesh_zone_boundary_rz(ep, zones, 2)              # ZONE_CONDUCTOR
        catch err
            @warn "export_imas: conductor-zone boundary extraction failed; vessel omitted" exception = err
            NamedTuple[]
        end
        for (i, l) in enumerate(vloops)
            length(l.R) ≥ 3 || continue
            ir["wall.description_2d.0.vessel.unit.$(i - 1).annular.outline_inner.r"] = l.R .* ulen
            ir["wall.description_2d.0.vessel.unit.$(i - 1).annular.outline_inner.z"] = l.Z .* ulen
        end
        isempty(vloops) ||
            (ir["wall.description_2d.0.vessel.type.name"] = "resistive_wall")
    end
    return nothing
end

# pellets IDS from the root /pellet group (whole-pellet model; layout mirrors
# the NIMROD MHDsimDB files: time_slice[it].pellet[ip].{path_profiles.
# {ablation_rate, ablated_particles, distance}, shape.size, velocity_initial,
# species}) plus one extra field the DB omits: path_profiles.position.{r,z,phi},
# the pellet/SPI-fragment location at that slice's step (a valid IMAS rzphi1d
# node — additive, so DB-drop-in readers ignore it; the axisym viewer scatters
# it). Units: rate = n0·l0³ particles / t0 (the cloud is unit-integral,
# see `read_pellets`); ablated_particles = Σ rate·dt over ALL steps up to the
# slice's ntimestep; distance = cumulative cylindrical arc length of the
# trajectory; species mirror NIMROD's SPI two-species layout — species.0 = D
# (fraction = pellet_mix, the D2 mole fraction), species.1 = KPRAD impurity
# (fraction = 1 − mix).
function _pellets_ir!(
        ir, file, norm::M3DNormalization,
        nsteps::AbstractVector{<:Integer}, times, meta
    )
    pel = read_pellets(file)
    (pel === nothing || pel.rate === nothing || pel.npellets <= 0) && return nothing
    np, ntot = size(pel.rate)
    if maximum(nsteps) + 1 > ntot
        @warn "export_imas: /pellet traces shorter than the exported steps; pellets IDS omitted"
        return nothing
    end
    ulen = unit_factor(norm, :length; system = :si)
    uvel = unit_factor(norm, :velocity; system = :si)
    ucount = norm.n0 * norm.l0^3                     # particles per normalized count
    urate = ucount / norm.t0                        # particles / s

    dtv = try
        read_scalar(file, "dt")
    catch err
        @debug "export_imas: dt trace unavailable; ablated_particles omitted" exception = err
        nothing
    end
    cum = (dtv !== nothing && length(dtv) >= ntot) ?
        cumsum(pel.rate .* reshape(dtv[1:ntot], 1, ntot); dims = 2) : nothing

    dist = nothing
    if pel.r !== nothing && pel.z !== nothing && pel.phi !== nothing
        dr = diff(pel.r; dims = 2);  dz = diff(pel.z; dims = 2)
        dφ = diff(pel.phi; dims = 2)
        rm = (pel.r[:, 1:(end - 1)] .+ pel.r[:, 2:end]) ./ 2
        dist = hcat(zeros(np), cumsum(sqrt.(dr .^ 2 .+ (rm .* dφ) .^ 2 .+ dz .^ 2); dims = 2))
    end

    # initial speed: first step with any motion (velphi treated as a linear
    # velocity component, like velr/velz)
    vinit = fill(NaN, np)
    if pel.velr !== nothing
        for ip in 1:np, n in 1:ntot
            v = hypot(
                pel.velr[ip, n],
                pel.velphi === nothing ? 0.0 : pel.velphi[ip, n],
                pel.velz === nothing ? 0.0 : pel.velz[ip, n]
            )
            v > 0 && (vinit[ip] = v; break)
        end
    end

    _ids_props!(ir, "pellets", meta, times)
    kz = _kprad_z(file)
    lbl, aimp = _impurity_species(max(kz, 0))
    for (it, n) in enumerate(nsteps)
        col = n + 1
        for ip in 1:np
            base = "pellets.time_slice.$(it - 1).pellet.$(ip - 1)"
            ir["$base.path_profiles.ablation_rate"] = [pel.rate[ip, col] * urate]
            cum === nothing ||
                (ir["$base.path_profiles.ablated_particles"] = [cum[ip, col] * ucount])
            dist === nothing ||
                (ir["$base.path_profiles.distance"] = [dist[ip, col] * ulen])
            # (R,Z,φ) of the pellet at this step — IMAS path_profiles.position
            # (rzphi1d), one point per slice so the axisym viewer can scatter it.
            pel.r === nothing || (ir["$base.path_profiles.position.r"] = [pel.r[ip, col] * ulen])
            pel.z === nothing || (ir["$base.path_profiles.position.z"] = [pel.z[ip, col] * ulen])
            pel.phi === nothing || (ir["$base.path_profiles.position.phi"] = [pel.phi[ip, col]])
            pel.r_p === nothing || (ir["$base.shape.size"] = [pel.r_p[ip, col] * ulen])
            isfinite(vinit[ip]) && (ir["$base.velocity_initial"] = vinit[ip] * uvel)
            ir["$base.species.0.label"] = "D"
            ir["$base.species.0.z_n"] = 1.0
            ir["$base.species.0.a"] = 2.014
            pel.mix === nothing ||
                (ir["$base.species.0.fraction"] = pel.mix[ip, col])
            if kz >= 0
                ir["$base.species.1.label"] = lbl
                ir["$base.species.1.z_n"] = Float64(kz)
                ir["$base.species.1.a"] = aimp
                pel.mix === nothing ||
                    (ir["$base.species.1.fraction"] = 1.0 - pel.mix[ip, col])
            end
        end
    end
    return nothing
end

# Sign fix for the labelled conventions: `vacuum_toroidal_field.b0` is signed
# like F, because M3D-C1's `bzero` attribute is an unsigned magnitude.
#
# `ip` is written UNMODIFIED. It used to be flipped whenever
# sign(ip) != sign(ψ_bnd − ψ_axis), on the premise that M3D-C1 is σ_Bp = +1 and
# its `toroidal_current` diagnostic was the odd one out. That premise was wrong
# in both halves and the flip was a structural bug — see `_to_cocos11_slice`.
# Briefly: M3D-C1's B = +∇ψ×∇φ + F∇φ puts the sign in front of ∇ψ×∇φ, whereas
# COCOS puts σ_Bp in front of ∇φ×∇ψ (OMAS `omas_physics.py`; MXHEquilibrium
# `fields.jl`), so M3D-C1 is σ_Bp = **−1**. The flip condition is then σ_Bp
# itself — true on *every* M3D-C1 run — and flipping ip reverses sign(Ip·B0),
# the magnetic helicity, which no COCOS relabelling may do (IP = TOR =
# σ_RφZ,in·σ_RφZ,out = +1 for every 3→11 transform). It was self-certifying:
# because σ_Ip appears once in each of OMAS' σ_Bp and σ_ρθφ tests, flipping ip
# flips both together, so `identify_cocos` returned [1, 11] by construction.
# Measured: native data returns [4, 14, 3, 13] on all three machines tested.
function _cocos11_meta(meta, results)
    isempty(results) && return meta
    b0 = meta.b0
    F = first(results).F1d
    if isfinite(b0) && F !== nothing
        fs = filter(isfinite, F)
        isempty(fs) || (b0 = sign(last(fs)) * abs(b0))
    end
    return merge(meta, (; b0 = b0))
end

function _with_vacuum_field(file::M3DC1File, norm::M3DNormalization, meta)
    r0 = b0 = NaN
    try
        h5open(file.path, "r") do f
            haskey(attrs(f), "rzero") && (r0 = Float64(read_attribute(f, "rzero")))
            haskey(attrs(f), "bzero") && (b0 = Float64(read_attribute(f, "bzero")))
        end
    catch err
        @debug "export_imas: vacuum field attrs unavailable" exception = err
    end
    (isnan(r0) || isnan(b0)) &&
        @warn "export_imas: vacuum-field attrs (rzero/bzero) missing; equilibrium.vacuum_toroidal_field omitted"
    ulen = unit_factor(norm, :length; system = :si)
    ub = unit_factor(norm, :magnetic_field; system = :si)
    return merge(
        meta, (;
            r0 = isnan(r0) ? NaN : r0 * ulen,
            b0 = isnan(b0) ? NaN : b0 * ub,
        )
    )
end

# kprad_z (impurity atomic number) when the KPRAD model is active, else -1
function _kprad_z(file::M3DC1File)
    z = -1
    try
        h5open(file.path, "r") do f
            ik = haskey(attrs(f), "ikprad") ? Int(read_attribute(f, "ikprad")) : 0
            ik != 0 && haskey(attrs(f), "kprad_z") && (z = Int(read_attribute(f, "kprad_z")))
        end
    catch err
        @debug "export_imas: kprad attrs unavailable" exception = err
    end
    return z
end

# impurity label + atomic mass [amu] for a KPRAD atomic number (species KPRAD supports)
function _impurity_species(z::Integer)
    z == 1  && return ("H", 1.008)
    z == 2  && return ("He", 4.0026)
    z == 4  && return ("Be", 9.0122)
    z == 5  && return ("B", 10.811)
    z == 6  && return ("C", 12.011)
    z == 10 && return ("Ne", 20.1797)
    z == 18 && return ("Ar", 39.948)
    return ("imp", 2.0 * Float64(max(z, 1)))   # fallback (Z not in the table)
end
