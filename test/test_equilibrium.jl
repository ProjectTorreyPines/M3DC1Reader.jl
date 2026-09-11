@testitem "M3DAxisymField struct" begin
    eq = M3DAxisymField(;
        cocos = 1, time = 0.0, R = range(1.0, 2.0; length = 4),
        Z = range(-1.0, 1.0; length = 4), psi_rz = zeros(4, 4),
        psi1d = [0.0, 0.5, 1.0], F1d = [3.0, 3.0, 3.0], p1d = nothing,
        axis = (1.5, 0.0), psi_axis = 0.0, psi_boundary = 1.0, r0 = 1.7, b0 = 2.0
    )
    @test eq.cocos == 1
    @test eq.F1d == [3.0, 3.0, 3.0]
    @test eq.axis == (1.5, 0.0)
    @test eq.time == 0.0
    @test length(eq.psi1d) == length(eq.F1d)
end

@testitem "F1d via reduce_axisym_slice (constant I -> constant F)" begin
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    Icoef = zeros(80, 1); Icoef[1, 1] = 5.0          # constant I = 5.0 (M3D units)
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :I => Icoef)
    res = reduce_axisym_slice(
        fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0,
        Rg, Zg, id_map; nbins = 16
    )
    F = res.F1d
    f_F = unit_factor(norm, :length; system = :si) * unit_factor(norm, :magnetic_field; system = :si)
    @test length(F) == 16
    @test all(F[isfinite.(F)] .≈ 5.0 * f_F)          # FSA of a constant = the constant
    @test any(isfinite, F)
end

@testitem "axisym_field (synthetic C1)" begin
    using HDF5
    # minimal single-element C1.h5 with psi + I fields (mirrors test_export_helpers
    # _make_min_c1, extended with the scalars/* group that read_timeslice requires).
    #
    # read_timeslice(file, ts; fields) reads:
    #   nstep = Int(read_attribute(g, "ntimestep"))     — g = time_NNN group
    #   scl["psi0"][nstep+1], scl["psi_lcfs"][nstep+1], scl["xmag"][nstep+1],
    #   scl["zmag"][nstep+1], scl["time"][nstep+1]
    # NOT "psi_axis" — the scalar dataset is named "psi0". The time group also
    # needs an `ntimestep` attribute (here 0, so all scalars are indexed at [1]).
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1); psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0; psi[4, 1] = 1.0; psi[6, 1] = 1.0
    Icoef = zeros(80, 1); Icoef[1, 1] = 5.0
    p = tempname() * ".h5"
    h5open(p, "w") do f
        attrs(f)["b0_norm"] = 1.0e4; attrs(f)["n0_norm"] = 1.0e14
        attrs(f)["l0_norm"] = 100.0; attrs(f)["ion_mass"] = 2.0
        attrs(f)["rzero"] = 1.7; attrs(f)["bzero"] = 2.0
        g = create_group(f, "time_001"); m = create_group(g, "mesh")
        m["elements"] = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
        attrs(m)["nplanes"] = 1; attrs(m)["period"] = 2π; attrs(m)["phi"] = [0.0]
        attrs(g)["ntimestep"] = 0
        fg = create_group(g, "fields"); fg["psi"] = psi; fg["I"] = Icoef
        scl = create_group(f, "scalars")
        scl["psi0"] = [0.0]
        scl["psi_lcfs"] = [0.05]
        scl["xmag"] = [1.2]
        scl["zmag"] = [0.3]
        scl["time"] = [0.0]
    end
    file = M3DC1File(p)
    eq = axisym_field(file, 1; ngrid = 40, nbins = 16, cocos = 1)
    @test eq isa M3DAxisymField
    @test length(eq.psi1d) == 16
    @test length(eq.F1d) == 16
    @test size(eq.psi_rz) == (40, 40)
    @test eq.r0 == 1.7
    @test eq.time == 0.0          # slice time threaded from read_timeslice
    @test any(isfinite, eq.F1d)
    rm(p; force = true)
end

@testitem "to_cocos: native COCOS-3 → COCOS-11" begin
    eq = M3DAxisymField(;
        time = 1.5, R = range(1.0, 2.0; length = 4),   # default cocos=3
        Z = range(-1.0, 1.0; length = 4), psi_rz = fill(0.1, 4, 4),
        psi1d = [0.0, 0.5, 1.0], F1d = [-3.0, -3.0, -3.0],
        p1d = [2.0, 1.0, 0.5], q1d = [-1.0, -1.5, NaN],
        axis = (1.5, 0.0), psi_axis = 0.0, psi_boundary = 1.0,
        r0 = 1.7, b0 = 2.0, x_points = [(1.4, -0.9)]
    )
    eq11 = to_cocos(eq, 11)
    @test eq11.cocos == 11
    # cocos_transform(3,11): ψ ×(−2π) because M3D-C1 is σ_Bp = −1
    @test eq11.psi1d ≈ -2π .* eq.psi1d
    @test eq11.psi_rz ≈ -2π .* eq.psi_rz
    @test eq11.psi_axis == 0.0
    @test eq11.psi_boundary ≈ -2π
    # q flips with σ_ρθφ (Q = −1 for 3 → 11); NaNs survive
    @test isequal(eq11.q1d, [1.0, 1.5, NaN])
    # …and the pair together satisfies COCOS-11's sign(q) = sign(Ip·B0):
    # ψ now DECREASES outward, which is σ_Bp = +1 with Ip > 0
    @test sign(eq11.psi_boundary - eq11.psi_axis) < 0
    @test eq11.F1d == eq.F1d
    @test eq11.p1d == eq.p1d
    @test eq11.axis == eq.axis && eq11.x_points == eq.x_points
    @test eq11.time == eq.time && eq11.r0 == eq.r0
    # b0 signed like (edge) F: attr magnitude 2.0, F<0 → −2.0
    @test eq11.b0 == -2.0
    # same-cocos no-op; unsupported targets / sources error
    @test to_cocos(eq, 3) === eq
    @test_throws ArgumentError to_cocos(eq, 1)
    @test_throws ArgumentError to_cocos(eq11, 5)   # 11 → per-radian not implemented
    # only the native COCOS-3 label has an implemented 3 → 11 transform
    eq5 = M3DAxisymField(;
        cocos = 5, time = 0.0, R = eq.R, Z = eq.Z, psi_rz = eq.psi_rz,
        psi1d = eq.psi1d, F1d = eq.F1d, axis = eq.axis, psi_axis = 0.0,
        psi_boundary = 1.0, r0 = 1.7, b0 = 2.0
    )
    @test_throws ArgumentError to_cocos(eq5, 11)

    # the COCOS index is the TYPE parameter: visible in typeof, dispatchable
    @test eq isa M3DAxisymField{3}
    @test eq11 isa M3DAxisymField{11}
    convlabel(::M3DAxisymField{11}) = :imas
    convlabel(::M3DAxisymField) = :native
    @test convlabel(eq) === :native
    @test convlabel(eq11) === :imas
    @test :cocos in propertynames(eq)              # eq.cocos compat accessor
    # thin re-wrap: everything except the ψ arrays is shared, not copied
    @test eq11.F1d === eq.F1d
    @test eq11.x_points === eq.x_points
    @test eq11.psi1d !== eq.psi1d
    # type parameter must be an Int COCOS index
    @test_throws ArgumentError M3DAxisymField{:cocos11}(
        eq.time, eq.R, eq.Z,
        eq.psi_rz, eq.psi1d, eq.F1d, eq.p1d, eq.q1d, eq.axis,
        eq.psi_axis, eq.psi_boundary, eq.r0, eq.b0, eq.x_points
    )
    # adapters guard against non-native input
    @test M3DC1Reader._assert_native(eq) === nothing
    @test_throws ArgumentError M3DC1Reader._assert_native(eq11)
end

# real-data guarded: the M3DAxisymField axis / psi_axis / psi_boundary must equal
# the find_lcfs recomputation (n=0 O-point + X-point/limiter boundary), so every
# adapter (to_mxh / to_geqdsk / to_imas) inherits the self-consistent ψ_N.
@testitem "axisym_field uses recomputed O-point + LCFS" begin
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if isfile(c1)
        file = M3DC1File(c1); ts = last(list_timeslices(file))
        norm = normalization(file)
        uflux = unit_factor(norm, :magnetic_flux; system = :si)
        ulen = unit_factor(norm, :length; system = :si)
        r = find_lcfs(file, ts)
        eqf = axisym_field(file, ts; ngrid = 129, nbins = 64)
        @test eqf.psi_axis ≈ r.psi_axis * uflux rtol = 1.0e-10
        @test eqf.axis[1] ≈ r.axis.R * ulen rtol = 1.0e-10
        @test eqf.axis[2] ≈ r.axis.Z * ulen rtol = 1.0e-10
        if isfinite(r.psi_bound)
            @test eqf.psi_boundary ≈ r.psi_bound * uflux rtol = 1.0e-10
        end
        @test all(isfinite, eqf.F1d)          # weighted F(ψ) FSA still fills the grid
        # safety factor via the FSA identity: present, endpoint-NaN, sane, rising
        @test eqf.q1d !== nothing
        qv = eqf.q1d
        @test isfinite(qv[1]) && isnan(qv[end])   # axis from V-fit; separatrix NaN
        qi = filter(isfinite, qv)
        @test length(qi) ≥ length(qv) - 2
        @test all(x -> 0.3 < abs(x) < 60, qi)
        k50 = (length(qv) + 1) ÷ 2
        @test abs(qv[end - 2]) > abs(qv[k50]) > abs(qv[5])   # |q| rises outward
        @test !isempty(eqf.x_points)                        # diverted → X-point recorded
    else
        @info "skipping axisym_field O/X consistency test (no C1.h5 at $c1)"
    end
end
