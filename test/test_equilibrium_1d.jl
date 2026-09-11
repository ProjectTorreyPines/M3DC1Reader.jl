# Tests for the equilibrium quantities derived from the FEM field rather than
# from a 1D profile: `boundary.outline` (trace_lcfs), `f_df_dpsi` /
# `dpressure_dpsi` (∇f·∇ψ/|∇ψ|², binned like every other profile), `j_tor`
# (⟨jφ/R⟩/⟨1/R⟩ from the R-weighted `jphi` field) and `rho_tor`.
#
# The synthetic items run everywhere; the two real-data items self-skip unless a
# C1.h5 (M3DC1_TEST_FILE) — and, for the EFIT cross-check, the run's own
# `geqdsk` next to it — are present.

@testitem "trace_lcfs (synthetic bowl+saddle)" setup = [LcfsFixture] begin
    # LcfsFixture: ψ = ξ² + (η−½)² − (η−½)³ on one element, O-point at
    # global (2, −½) with ψ=0, saddle at (2, 1/6) with ψ = 4/27.
    r = find_lcfs(
        lcoef, lelems; xmag = 2.05, zmag = -0.45,
        xnull = 2.0, znull = 0.1, xlim = 2.5, zlim = -0.8
    )
    xp = [(r.x1.R, r.x1.Z)]
    tr = trace_lcfs(
        lcoef, lelems, (r.axis.R, r.axis.Z), r.psi_axis, r.psi_bound;
        ntheta = 129, x_points = xp
    )
    @test tr !== nothing
    @test tr.R[1] == tr.R[end] && tr.Z[1] == tr.Z[end]        # closed polygon
    # every traced point sits on the boundary flux (that is the whole contract)
    ψ = [eval_axisym_at(lcoef, lelems, tr.R[k], tr.Z[k]) for k in eachindex(tr.R)]
    @test all(isfinite, ψ)
    @test maximum(abs, ψ .- r.psi_bound) < 1.0e-8 * max(abs(r.psi_bound), 1.0)
    # the surface encloses the axis and reaches the X-point
    @test minimum(tr.R) < r.axis.R < maximum(tr.R)
    @test minimum(tr.Z) < r.axis.Z < maximum(tr.Z)
    @test any(k -> hypot(tr.R[k] - 2.0, tr.Z[k] - 1 / 6) < 1.0e-8, eachindex(tr.R))

    # degenerate input is rejected rather than returning garbage
    @test trace_lcfs(lcoef, lelems, (2.0, -0.5), 0.0, 0.0) === nothing
    @test trace_lcfs(lcoef, lelems, (NaN, -0.5), 0.0, 4 / 27) === nothing
end

@testitem "trace_lcfs id_map hint matches the plain scan" setup = [LcfsFixture] begin
    r = find_lcfs(lcoef, lelems; xmag = 2.05, zmag = -0.45, xnull = 2.0, znull = 0.1)
    # element vertices: (0,−1), (4,−1), (2,1) — row 5/6 is only the origin
    Rg = collect(range(0.0, 4.0; length = 24))
    Zg = collect(range(-1.0, 1.0; length = 24))
    idm = build_grid_to_element_map(Rg, Zg, lelems)
    a = trace_lcfs(lcoef, lelems, (r.axis.R, r.axis.Z), r.psi_axis, r.psi_bound; ntheta = 65)
    b = trace_lcfs(
        lcoef, lelems, (r.axis.R, r.axis.Z), r.psi_axis, r.psi_bound;
        ntheta = 65, id_map = idm, R_grid = Rg, Z_grid = Zg
    )
    @test a !== nothing && b !== nothing
    @test a.nmiss == b.nmiss
    @test maximum(abs, a.R .- b.R) < 1.0e-12       # the hint must not change the answer
    @test maximum(abs, a.Z .- b.Z) < 1.0e-12
end

@testitem "_fill_axis_band! reproduces a smooth profile" begin
    f = M3DC1Reader._fill_axis_band!
    ρ = collect(range(0.0, 1.0; length = 128));  ψN = ρ .^ 2
    truth = [2.0 - 3.0 * x + 1.5 * x^2 - 0.4 * x^3 for x in ψN]   # cubic ⇒ exact fit
    v = copy(truth)
    v[ψN .< 0.01] .= NaN                                   # blank the axis region
    out = f(copy(v), ψN)
    @test all(isfinite, out)
    @test maximum(abs, out .- truth) < 1.0e-8
    # a biased (not just missing) axis region is overwritten too
    v2 = copy(truth);  v2[ψN .< 0.01] .+= 0.5
    @test maximum(abs, f(copy(v2), ψN) .- truth) < 1.0e-8
    # too few finite nodes in the fit band ⇒ left untouched, no throw
    v3 = fill(NaN, 128);  v3[1] = 7.0
    @test f(copy(v3), ψN)[1] == 7.0
    # only_nonfinite: fills the gaps, leaves finite (even biased) nodes alone
    v4 = copy(truth);  v4[ψN .< 0.01] .+= 0.5;  v4[1] = NaN
    o4 = f(copy(v4), ψN; only_nonfinite = true)
    @test isfinite(o4[1]) && abs(o4[1] - truth[1]) < 1.0e-8
    @test o4[2] == v4[2]                                    # biased node preserved
end

@testitem "export_imas equilibrium 1D derivatives + boundary (real C1.h5)" begin
    using HDF5
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    if !isfile(c1)
        @info "skipping equilibrium-1D export test (no C1.h5 at $c1)"
    else
        file = M3DC1File(c1)
        ts = first(list_timeslices(file))
        outs = Dict{Symbol, String}()
        for (tag, cc) in ((:raw, nothing), (:c11, 11), (:mhd, :mhdsimdb))
            o = tempname() * ".h5"
            export_imas(file, o; slices = [ts], ngrid = 100, nbins = 64, cocos = cc)
            outs[tag] = o
        end
        rd(tag, p) = h5open(f -> read(f["equilibrium/time_slice/0/$p"]), outs[tag], "r")

        # --- presence and shape
        for p in ("profiles_1d/f_df_dpsi", "profiles_1d/dpressure_dpsi",
                "profiles_1d/j_tor", "profiles_1d/rho_tor", "profiles_1d/rho_tor_norm",
                "boundary/outline/r", "boundary/outline/z", "boundary/psi")
            @test h5open(f -> haskey(f, "equilibrium/time_slice/0/$p"), outs[:raw], "r")
        end
        @test length(rd(:raw, "profiles_1d/f_df_dpsi")) == 64
        @test all(isfinite, rd(:raw, "profiles_1d/f_df_dpsi"))
        @test all(isfinite, rd(:raw, "profiles_1d/dpressure_dpsi"))

        # --- boundary.outline: closed, encloses the axis, sits on psi_boundary
        br = rd(:raw, "boundary/outline/r");  bz = rd(:raw, "boundary/outline/z")
        @test length(br) == length(bz) > 32
        @test br[1] == br[end] && bz[1] == bz[end]
        rax = rd(:raw, "global_quantities/magnetic_axis/r")
        zax = rd(:raw, "global_quantities/magnetic_axis/z")
        @test minimum(br) < rax < maximum(br)
        @test minimum(bz) < zax < maximum(bz)
        @test rd(:raw, "boundary/psi") ≈ rd(:raw, "global_quantities/psi_boundary")
        # strictly inside the computational wall
        wr = h5open(f -> read(f["wall/description_2d/0/limiter/unit/0/outline/r"]), outs[:raw], "r")
        wz = h5open(f -> read(f["wall/description_2d/0/limiter/unit/0/outline/z"]), outs[:raw], "r")
        @test minimum(wr) <= minimum(br) && maximum(br) <= maximum(wr)
        @test minimum(wz) <= minimum(bz) && maximum(bz) <= maximum(wz)

        # --- rho_tor: a length, monotone, normalized to 1 at the edge
        rt = rd(:raw, "profiles_1d/rho_tor");  rn = rd(:raw, "profiles_1d/rho_tor_norm")
        @test all(>=(0), rt) && issorted(rt)
        @test rt[1] ≈ 0 atol = 1.0e-12
            @test rn[end] ≈ 1
        # rho_tor = a·√κ for an elongated plasma, so it can exceed the geometric
        # minor radius — bound it by the wall span instead.
        @test 0.1 < rt[end] < maximum(wr) - minimum(wr)

        # --- j_tor: peaked, single-signed, same sign as the exported ip
        jt = filter(isfinite, rd(:raw, "profiles_1d/j_tor"))
        @test !isempty(jt)
        @test abs(jt[1]) > abs(jt[end])
        ip = rd(:c11, "global_quantities/ip")
        @test sign(jt[1]) == sign(ip)

        # --- COCOS 3 → 11: ψ ×(−2π), q ×(−1), ψ-derivatives ×(−1/2π);
        #     Ip / F / Φ / ρ_tor / geometry untouched (see cocos_transform(3,11))
        @test rd(:c11, "profiles_1d/psi") ≈ rd(:raw, "profiles_1d/psi") .* (-2π)
        @test rd(:c11, "boundary/psi") ≈ rd(:raw, "boundary/psi") * (-2π)
        @test rd(:c11, "profiles_1d/f_df_dpsi") ≈ rd(:raw, "profiles_1d/f_df_dpsi") ./ (-2π)
        @test rd(:c11, "profiles_1d/dpressure_dpsi") ≈
            rd(:raw, "profiles_1d/dpressure_dpsi") ./ (-2π)
        @test isapprox(
            rd(:c11, "profiles_1d/q"), -1 .* rd(:raw, "profiles_1d/q"); nans = true
        )
        @test rd(:c11, "profiles_1d/j_tor") ≈ rd(:raw, "profiles_1d/j_tor")
        @test rd(:c11, "profiles_1d/rho_tor") ≈ rd(:raw, "profiles_1d/rho_tor")
        @test rd(:c11, "boundary/outline/r") ≈ rd(:raw, "boundary/outline/r")
        @test rd(:c11, "global_quantities/ip") == rd(:raw, "global_quantities/ip")
        # mhdsimdb keeps the equilibrium IDS in M3D-C1's native COCOS 3
        @test rd(:mhd, "profiles_1d/f_df_dpsi") ≈ rd(:raw, "profiles_1d/f_df_dpsi")
        @test rd(:mhd, "global_quantities/ip") == rd(:raw, "global_quantities/ip")

        # ══ SIGN REGRESSION GUARDS ═══════════════════════════════════════════
        # A wrong σ_Bp once put an ip flip into `_cocos11_meta` that reversed the
        # current direction of every exported file, and survived because every
        # check in this area was magnitude-only. These six are sign-sensitive.

        # (i) ip is M3D-C1's `toroidal_current`, unmodified
        ipraw = h5open(c1, "r") do f
            read(f["scalars/toroidal_current"])[1]
        end
        ui = unit_factor(normalization(M3DC1File(c1)), :current; system = :si)
        @test sign(rd(:raw, "global_quantities/ip")) == sign(ipraw)
        @test rd(:raw, "global_quantities/ip") ≈ ipraw * ui rtol = 1.0e-6

        # (ii) native ψ is σ_Bp = −1: it runs OPPOSITE the current…
        spanraw = rd(:raw, "global_quantities/psi_boundary") -
            rd(:raw, "global_quantities/psi_axis")
        @test sign(spanraw) == -sign(rd(:raw, "global_quantities/ip"))
        # …and COCOS 11 (σ_Bp = +1) must therefore also give −sign(Ip), since the
        # ×(−2π) flips the span while ip stays put
        span11 = rd(:c11, "global_quantities/psi_boundary") -
            rd(:c11, "global_quantities/psi_axis")
        @test sign(span11) == -sign(spanraw)

        # (iii) COCOS-11 invariant with the PHYSICAL ip: sign(q) = sign(Ip·B0)
        b0 = h5open(f -> read(f["equilibrium/vacuum_toroidal_field/b0"])[1], outs[:c11], "r")
        q11 = filter(isfinite, rd(:c11, "profiles_1d/q"))
        @test sign(q11[1]) == sign(rd(:c11, "global_quantities/ip") * b0)

        # (iv) the 2D ψ map and the b_field arrays beside it must agree IN SIGN
        # under each file's declared convention: B_R = σ_Bp·(∂ψ/∂Z)/((2π)^e_Bp·R)
        for (tag, sgn, tp) in ((:raw, -1.0, 1.0), (:c11, +1.0, 2π))
            g2 = "equilibrium/time_slice/0/profiles_2d/0"
            Rg, Zg, ψ2, bR = h5open(outs[tag], "r") do f
                (
                    read(f["$g2/grid/dim1"]), read(f["$g2/grid/dim2"]),
                    read(f["$g2/psi"]), read(f["$g2/b_field_r"]),
                )
            end
            # NOTE the index order: `write_omas_h5` stores 2D maps in the
            # C/OMAS layout, so reading them back in (column-major) Julia yields
            # the TRANSPOSE — `ψ2[jZ, iR]`, not `ψ2[iR, jZ]`.
            # Sample the WHOLE map, not the midplane: B_R = σ_Bp·ψ_Z/R vanishes
            # there by up-down symmetry, so ratios taken near Z ≈ Z_axis are 0/0.
            acc = Float64[]
            for jj in 2:(length(Zg) - 1), i in 2:(length(Rg) - 1)
                abs(bR[jj, i]) > 0.05 || continue            # T — well off the null
                dψdZ = (ψ2[jj + 1, i] - ψ2[jj - 1, i]) / (Zg[jj + 1] - Zg[jj - 1])
                pred = sgn * dψdZ / (tp * Rg[i])
                (isfinite(pred) && isfinite(bR[jj, i])) || continue
                push!(acc, pred / bR[jj, i])
            end
            @test length(acc) > 500
            sort!(acc)
            med = acc[(length(acc) + 1) ÷ 2]
            @test 0.9 < med < 1.1                          # POSITIVE ⇒ same sign
        end

        # (v) sign(∫ j_tor dA) agrees with ip (j_tor ≈ ⟨jφ/R⟩/⟨1/R⟩, so the
        # sign of its core value is enough)
        @test sign(jt[1]) == sign(rd(:raw, "global_quantities/ip"))
        # ═════════════════════════════════════════════════════════════════════

        # --- self-consistency: ∫f_df_dpsi dψ must rebuild F²/2 from the file's
        # own `f`, in EVERY convention (the 2π cancels in the integral). This is
        # the check that catches an inverted COCOS rule on the derivatives.
        for tag in (:raw, :c11, :mhd)
            ψ = rd(tag, "profiles_1d/psi");  F = rd(tag, "profiles_1d/f")
            ffp = rd(tag, "profiles_1d/f_df_dpsi")
            cum = cumsum([0.0; 0.5 .* (ffp[2:end] .+ ffp[1:(end - 1)]) .* diff(ψ)])
            k0 = findfirst(isfinite, F)
            Fi = sign(F[k0]) .* sqrt.(max.(F[k0]^2 .+ 2 .* (cum .- cum[k0]), 0.0))
            ψN = (ψ .- ψ[1]) ./ (ψ[end] - ψ[1])
            m = [i for i in eachindex(F) if isfinite(F[i]) && ψN[i] <= 0.98]
            rel = sqrt(sum(((Fi[m] .- F[m]) ./ F[m]) .^ 2) / length(m))
            @test rel < 1.0e-3
        end

        foreach(rm, values(outs))
    end
end

# Cross-validation against the run's own EFIT input. Only slice 0 is comparable
# (M3D-C1 re-solves the Grad-Shafranov equation on its own mesh at startup, so
# even t=0 differs from the gEQDSK at the ~1% level in q95/pressure — hence the
# loose tolerances on everything except FF′, where the agreement is 0.03% of the
# ffprim scale). Doubly gated: needs a real C1.h5 AND a `geqdsk` beside it.
@testitem "f_df_dpsi / boundary vs the run's own gEQDSK" begin
    using HDF5, Statistics
    c1 = get(ENV, "M3DC1_TEST_FILE", "/scratch/gpfs/myoo/m3d_smoke/C1.h5")
    gpath = joinpath(dirname(c1), "geqdsk")
    if !isfile(c1) || !isfile(gpath)
        @info "skipping gEQDSK cross-check (need C1.h5 + geqdsk in $(dirname(c1)))"
    else
        # minimal gEQDSK reader (only the blocks this check needs)
        lines = readlines(gpath)
        toks = Float64[]
        for L in lines[2:end], m in eachmatch(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?", L)
            push!(toks, parse(Float64, replace(m.match, r"[Dd]" => "E")))
        end
        _hdr = split(lines[1])
        nw = parse(Int, _hdr[end - 1]);  nh = parse(Int, _hdr[end])
        k = Ref(0)
        take(n) = (v = toks[(k[] + 1):(k[] + n)]; k[] += n; v)
        _, _, _, _, _ = take(5)
        _, _, simag, sibry, _ = take(5)
        take(5); take(5)
        fpol = take(nw);  take(nw);  ffprim = take(nw)

        file = M3DC1File(c1)
        out = tempname() * ".h5"
        export_imas(file, out; slices = [first(list_timeslices(file))], ngrid = 200, cocos = nothing)
        eq(p) = h5open(f -> read(f["equilibrium/time_slice/0/$p"]), out, "r")
        ψ = eq("profiles_1d/psi")
        ψ0 = eq("global_quantities/psi_axis");  ψ1 = eq("global_quantities/psi_boundary")
        ψN = (ψ .- ψ0) ./ (ψ1 - ψ0)
        # our ψ may run opposite to the gEQDSK's (M3D-C1 flips it on some
        # devices); FF′ = F dF/dψ flips with ψ, |F| does not.
        sψ = sign(ψ1 - ψ0) * sign(sibry - simag)
        gN = collect(range(0.0, 1.0; length = nw))
        lin(y, x) = (i = searchsortedlast(gN, clamp(x, 0, 1));
            i >= nw ? y[nw] : y[i] + (y[i + 1] - y[i]) * (x - gN[i]) / (gN[i + 1] - gN[i]))

        # F(ψ): the profile M3D-C1 inherits essentially unchanged
        F = eq("profiles_1d/f")
        mid = [i for i in eachindex(ψN) if 0.02 <= ψN[i] <= 0.98 && isfinite(F[i])]
        @test maximum(abs, [F[i] / lin(fpol, ψN[i]) - 1 for i in mid]) < 5.0e-4

        # FF′: the point of the exercise. Compared only over ψ_N ∈ [0.02, 0.98]:
        # EFIT's own ffprim at the magnetic-axis node is an extrapolation of its
        # fit basis and is not a usable reference — on the KSTAR case it reads
        # 2.518 next to 2.064 at the very next node (a 22% jump into an
        # otherwise smooth profile), and our value continues the smooth trend.
        ffp = eq("profiles_1d/f_df_dpsi")
        sc = maximum(abs, ffprim)
        err = [abs(ffp[i] - sψ * lin(ffprim, ψN[i])) for i in mid] ./ sc
        @test sqrt(mean(err .^ 2)) < 0.01              # measured 0.0002 on JET SPI
        @test maximum(err) < 0.05

        # boundary.outline vs RBBBS/ZBBBS: mean radial deviation about the axis
        take(nw);  take(nw * nh);  take(nw)            # pprime, psirz, qpsi
        nb = Int(toks[k[] + 1]);  k[] += 2             # nbbbs, limitr
        bb = take(2nb);  rb = bb[1:2:end];  zb = bb[2:2:end]
        br = eq("boundary/outline/r");  bz = eq("boundary/outline/z")
        rax = eq("global_quantities/magnetic_axis/r")
        zax = eq("global_quantities/magnetic_axis/z")
        function radial(R, Z)
            θ = atan.(Z .- zax, R .- rax);  ρ = hypot.(R .- rax, Z .- zax)
            p = sortperm(θ);  θs = θ[p];  ρs = ρ[p]
            return [
                (
                    i = searchsortedlast(θs, t);
                    (i < 1 || i >= length(θs)) ? ρs[clamp(i, 1, length(θs))] :
                    ρs[i] + (ρs[i + 1] - ρs[i]) * (t - θs[i]) / (θs[i + 1] - θs[i])
                )
                    for t in range(-3.0, 3.0; length = 180)
            ]
        end
        a = radial(br, bz);  b = radial(rb, zb)
        @test mean(abs.(a .- b)) / mean(b) < 0.02      # measured 0.0009 on JET SPI
        rm(out)
    end
end
