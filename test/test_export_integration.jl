@testitem "export_imas cocos=11 vs raw (synthetic C1 file)" begin
    using HDF5
    # two-element synthetic C1: element 1 carries the bowl ψ (axis inside),
    # element 2 only widens the mesh-origin extents so export_imas' internal
    # grid (built from element origins) spans the bowl.
    ep1 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    ep2 = [1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 2)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    psi[1, 2] = 0.05                                  # flat ψ = ψ_bnd on the far element
    Ic = zeros(80, 2); Ic[1, 1] = 4.0; Ic[1, 2] = 4.0

    p = tempname() * ".h5"
    h5open(p, "w") do f
        attrs(f)["b0_norm"] = 1.0e4; attrs(f)["n0_norm"] = 1.0e14
        attrs(f)["l0_norm"] = 100.0; attrs(f)["ion_mass"] = 2.0
        g = create_group(f, "time_001"); attrs(g)["ntimestep"] = 0
        m = create_group(g, "mesh")
        m["elements"] = hcat(ep1, ep2)
        attrs(m)["nplanes"] = 1; attrs(m)["period"] = 2π; attrs(m)["phi"] = [0.0]
        fg = create_group(g, "fields"); fg["psi"] = psi; fg["I"] = Ic
        scl = create_group(f, "scalars")
        scl["psi0"] = [0.0]; scl["psi_lcfs"] = [0.05]
        scl["xmag"] = [1.2]; scl["zmag"] = [0.3]; scl["time"] = [0.0]
        scl["toroidal_current"] = [-0.5]      # sign OPPOSITE to the ψ-map orientation
        scl["W_P"] = [0.1]
    end
    file = M3DC1File(p)
    norm = normalization(file)
    uI = unit_factor(norm, :current; system = :si)

    out11 = tempname() * ".h5"; outraw = tempname() * ".h5"
    export_imas(file, out11; slices = [1], nbins = 16, ngrid = 40)             # default cocos=11
    export_imas(file, outraw; slices = [1], nbins = 16, ngrid = 40, cocos = nothing)

    h5open(out11, "r") do f11
        h5open(outraw, "r") do fr
            # ψ ×2π everywhere (grid, 2D map, boundary scalar)
            @test read(f11["equilibrium/time_slice/0/global_quantities/psi_boundary"]) ≈
                -2π * read(fr["equilibrium/time_slice/0/global_quantities/psi_boundary"])
            p11 = read(f11["core_profiles/profiles_1d/0/grid/psi"])
            praw = read(fr["core_profiles/profiles_1d/0/grid/psi"])
            @test p11 ≈ -2π .* praw
            m11 = read(f11["equilibrium/time_slice/0/profiles_2d/0/psi"])
            mraw = read(fr["equilibrium/time_slice/0/profiles_2d/0/psi"])
            @test isapprox(m11, -2π .* mraw; nans = true)
            # q flips with σ_ρθφ (Q = −1 for cocos_transform(3, 11))
            @test isapprox(
                read(f11["core_profiles/profiles_1d/0/q"]),
                -1 .* read(fr["core_profiles/profiles_1d/0/q"]); nans = true
            )
            # F is a frame quantity — identical in both conventions
            @test isequal(
                read(f11["equilibrium/time_slice/0/profiles_1d/f"]),
                read(fr["equilibrium/time_slice/0/profiles_1d/f"])
            )
            # ip is NEVER touched by a COCOS transform (IP = +1 for 3 → 11): both
            # files carry M3D-C1's `toroidal_current` verbatim. A flip here was a
            # structural bug — it reverses sign(Ip·B0), the magnetic helicity.
            @test read(fr["summary/global_quantities/ip/value"]) ≈ [-0.5 * uI]
            @test read(f11["summary/global_quantities/ip/value"]) ≈ [-0.5 * uI]
            @test read(f11["equilibrium/time_slice/0/global_quantities/ip"]) ≈ -0.5 * uI
            # the convention is recorded in the comment
            @test occursin("COCOS-11", read(f11["core_profiles/ids_properties/comment"]))
            @test occursin("COCOS 3", read(fr["core_profiles/ids_properties/comment"]))
        end
    end

    # :mhdsimdb — the NIMROD-DB drop-in layout (mixed on purpose): equilibrium
    # stays per-radian while cp/disruption grid.psi = 2π·(ψ−ψ_axis) axis-zeroed
    outdb = tempname() * ".h5"
    export_imas(file, outdb; slices = [1], nbins = 16, ngrid = 40, cocos = :mhdsimdb)
    h5open(outdb, "r") do fdb
        h5open(outraw, "r") do fr
            # equilibrium block: identical to the raw native-COCOS-3 output
            @test read(fdb["equilibrium/time_slice/0/global_quantities/psi_boundary"]) ≈
                read(fr["equilibrium/time_slice/0/global_quantities/psi_boundary"])
            @test read(fdb["equilibrium/time_slice/0/profiles_1d/psi"]) ≈
                read(fr["equilibrium/time_slice/0/profiles_1d/psi"])
            @test isequal(
                read(fdb["equilibrium/time_slice/0/profiles_1d/q"]),
                read(fr["equilibrium/time_slice/0/profiles_1d/q"])
            )
            # cp grid.psi: axis-zeroed total flux, 2π·(ψ−ψ_axis)
            praw = read(fr["core_profiles/profiles_1d/0/grid/psi"])
            pa = read(fr["equilibrium/time_slice/0/global_quantities/psi_axis"])
            pdb = read(fdb["core_profiles/profiles_1d/0/grid/psi"])
            @test pdb ≈ 2π .* (praw .- pa)
            @test pdb[1] ≈ 0.0 atol = 1.0e-12               # zero at the axis node
            # ip is unmodified here too
            @test read(fdb["summary/global_quantities/ip/value"]) ≈ [-0.5 * uI]
            @test occursin("MHDsimDB", read(fdb["core_profiles/ids_properties/comment"]))
        end
    end
    # `cocos11_path`: the same reduction written twice. The twin must be
    # byte-for-byte what a standalone cocos=11 run produces, so a consumer that
    # wants IMAS-standard psi never needs the (multi-GB) slice files re-read.
    outdb2 = tempname() * ".h5";  outtwin = tempname() * ".h5"
    ret = export_imas(
        file, outdb2; slices = [1], nbins = 16, ngrid = 40,
        cocos = :mhdsimdb, cocos11_path = outtwin
    )
    @test ret == (outdb2, outtwin)
    @test isfile(outtwin)
    h5open(outtwin, "r") do ft
        h5open(out11, "r") do f11
            for path in (
                    "equilibrium/time_slice/0/profiles_1d/psi",
                    "equilibrium/time_slice/0/profiles_1d/f",
                    "core_profiles/profiles_1d/0/grid/psi",
                    "equilibrium/time_slice/0/global_quantities/psi_axis",
                )
                @test isapprox(read(ft[path]), read(f11[path]); nans = true)
            end
            @test occursin("COCOS-11", read(ft["equilibrium/ids_properties/comment"]))
        end
        # …while the primary output stayed in the MHDsimDB layout
        h5open(outdb2, "r") do fd
            @test read(ft["equilibrium/time_slice/0/global_quantities/psi_axis"]) ≈
                -2π * read(fd["equilibrium/time_slice/0/global_quantities/psi_axis"])
            @test occursin("MHDsimDB", read(fd["equilibrium/ids_properties/comment"]))
        end
    end
    # redundant when the primary output already IS cocos=11
    @test_throws ArgumentError export_imas(
        file, tempname() * ".h5"; slices = [1], nbins = 16, ngrid = 40,
        cocos = 11, cocos11_path = tempname() * ".h5"
    )

    rm(p; force = true); rm(out11; force = true); rm(outraw; force = true)
    rm(outdb; force = true); rm(outdb2; force = true); rm(outtwin; force = true)
end

@testitem "export pipeline integration (compute→assemble→write, no data file)" begin
    using HDF5
    norm = M3DNormalization(b0 = 1.0e4, n0 = 1.0e14, l0 = 100.0, ion_mass = 2.0)
    ep = reshape([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 10, 1)
    ξ0, η0 = 0.2, 0.3
    psi = zeros(80, 1)
    psi[1, 1] = ξ0^2 + η0^2; psi[2, 1] = -2ξ0; psi[3, 1] = -2η0
    psi[4, 1] = 1.0;         psi[6, 1] = 1.0
    te = zeros(80, 1); te[1, 1] = 3.0
    fields = Dict{Symbol, Matrix{Float64}}(:psi => psi, :te => te)
    Rg = collect(range(1.05, 1.35, length = 6))
    Zg = collect(range(0.15, 0.45, length = 6))
    id_map = build_grid_to_element_map(Rg, Zg, ep)

    s0 = reduce_axisym_slice(fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 0.0, Rg, Zg, id_map; nbins = 16)
    s1 = reduce_axisym_slice(fields, ep, 1, norm, 0.0, 0.05, 1.2, 0.3, 1.0, Rg, Zg, id_map; nbins = 16)
    meta = (;
        source = "M3D-C1", provider = "M3DC1Reader.jl", code_name = "M3DC1Reader",
        comment = "integration test", ion_label = "D+", z_ion = 1.0, ion_a = 2.0,
        r0 = 1.7, b0 = 2.0,
    )
    ir = assemble_ir([s0, s1], meta)

    tmp = tempname() * ".h5"
    write_omas_h5(tmp, ir)
    h5open(tmp, "r") do f
        @test haskey(f, "core_profiles") && haskey(f, "equilibrium")
        @test read(f["core_profiles/ids_properties/homogeneous_time"]) === Int64(1)
        @test read(f["core_profiles/time"]) == [0.0, 1.0]
        te_out = read(f["core_profiles/profiles_1d/0/electrons/temperature"])
        @test length(te_out) == 16
        @test maximum(filter(isfinite, te_out)) ≈ 3.0 * unit_factor(norm, :temperature; system = :si)
        @test ndims(f["equilibrium/time_slice/0/profiles_2d/0/psi"]) == 2
        @test read(f["equilibrium/vacuum_toroidal_field/r0"]) == 1.7
    end
    rm(tmp; force = true)
end
