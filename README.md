[![CI](https://github.com/ProjectTorreyPines/M3DC1Reader/actions/workflows/CI.yml/badge.svg)](https://github.com/ProjectTorreyPines/M3DC1Reader/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/projecttorreypines/m3dc1reader/graph/badge.svg?token=JDIwgzYKI2)](https://codecov.io/github/projecttorreypines/m3dc1reader)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

# M3DC1Reader.jl

Read M3D-C1 `C1.h5` output and work with its native C¹ reduced quintic Hermite
finite elements directly in Julia: read the mesh, fields and scalars; evaluate
fields and their exact derivatives at arbitrary points or grids; find critical
points and the LCFS; convert units; and export IMAS/OMAS HDF5.

Pure Julia, single core, no external field-interpolation library. All
evaluation is analytic on the finite elements — there is no intermediate
resampling.

## Quickstart

```julia
using M3DC1Reader

file  = M3DC1File("/path/to/C1.h5")          # caches mesh + normalization
slice = read_timeslice(file, 24; fields = (:psi, :te))

# (1) n = 0 toroidal average
psi_ax = average_toroidal_axisymmetric(slice.fields[:psi], file.nplanes)
te_ax  = average_toroidal_axisymmetric(slice.fields[:te] .* te_to_eV_factor(file),
                                       file.nplanes)

# self-consistent n = 0 magnetic axis
ep = elems_plane(file)
ax = find_axis_newton(psi_ax, ep, slice.xmag, slice.zmag)

# (2) 2D rectilinear field  (any AbstractVector works — ranges included)
Rg = range(extrema(ep[5, :])..., length = 100)
Zg = range(extrema(ep[6, :])..., length = 100)
id_map = build_grid_to_element_map(Rg, Zg, ep)
te_2d  = interpolate_axisym_to_grid(te_ax, ep, Rg, Zg; id_map = id_map)

# (3) 1D Te(ρ_pol)
psi_2d = interpolate_axisym_to_grid(psi_ax, ep, Rg, Zg; id_map = id_map)
ψn     = psi_to_psi_norm.(vec(psi_2d), ax.ψ, slice.psi_lcfs)
prof   = reduce_1d_psi_func(psi_n_to_rho_pol.(ψn), vec(te_2d);
                            n_bins = 60, psi_range = (0.0, 1.2), adj = :linear)
# prof.psi_grid, prof.func_bin
```

`id_map` depends only on the mesh and the grid, so build it once and reuse it
across every field and time slice.

## Pipeline

```
raw 3D mesh + 80-coef Hermite fields per element
  │  average_toroidal_axisymmetric      n = 0 toroidal Fourier mode
  ▼
axisymmetric coefficients per element (one R-Z plane)
  │  interpolate_axisym_to_grid         analytic FEM evaluation
  ▼
2D field on a rectilinear R-Z grid
  │  reduce_1d_psi_func                 adjoint scattered→grid projection
  ▼
1D profile vs ρ_pol = √ψ_N
```

The ψ_N reference axis comes from `find_axis_newton`, a reimplementation of
M3D-C1's internal `magaxis` Newton iteration on the FEM polynomial
(`diagnostics.f90:magaxis`, `imethod = 0`).

## IMAS export

`export_imas` runs the pipeline over a set of time slices and writes an
OMAS-style HDF5 file (`equilibrium`, `core_profiles`, `summary`, `wall`,
`pellets`, `disruption`), including flux-surface-averaged 1D profiles, the
traced LCFS outline, and 0D global traces from `scalars/*`.

```julia
export_imas(file, "M3DC1_axisym.h5"; cocos = 11, verbose = true)
```

```bash
scripts/export_run /path/to/run_dir           # one run
scripts/scan_export                           # many runs (SLURM-shardable)
```

The 2D R-Z maps can also be written as ASCOT5 input (`ascot5 = true`, or
`ascot5_bfield` / `write_ascot5` directly).

## Units

M3D-C1 stores fields in its own dimensionless normalization. A data-driven
registry converts any quantity to cgs or SI:

```julia
norm = normalization(file)                          # b0, n0, l0, ion_mass (+ V0, T0)

unit_factor(norm, :temperature)                     # eV per M3D unit (system = :si)
unit_factor(norm, :magnetic_field; system = :cgs)   # Gauss per M3D unit
to_units(norm, slice.fields[:te], :temperature)     # Te field → eV
unit_label(:magnetic_field; system = :si)           # "T"
available_quantities()                              # :density, :velocity, …
```

Systems: `:m3d` (factor 1), `:cgs`, `:si` (alias `:mks`).
`te_to_eV_factor(file)` is shorthand for `unit_factor(norm, :temperature)`.

## Conventions

- The 1D radial coordinate is **ρ_pol = √ψ_N**.
- M3D-C1's native ψ is **per radian** with σ_Bp = −1, i.e. a self-consistent
  **COCOS 3** file. `cocos = 11` applies the real 3→11 transform
  (ψ ×(−2π), q ×(−1), FF′ and p′ ÷(−2π)); the plasma current is never flipped.
- `psi_axis` / `psi_lcfs` in `scalars` are M3D-C1's **plane-1 (φ = 0)** values.
  For a toroidally averaged reference use `find_axis_newton` on the averaged ψ
  — the two differ by ~1% of |Δψ| during a disruption.

## Module layout

| File | Scope |
|------|-------|
| `src/io.jl` | `C1.h5` reader (`M3DC1File`, `read_timeslice`, …) |
| `src/elements.jl` | reduced quintic Hermite FEM math, element location |
| `src/reductions.jl` | toroidal average, 2D evaluation, 1D adjoint reduction |
| `src/magaxis.jl` | Newton O-point finder (`find_axis_newton`) |
| `src/find_critical_points.jl` | O/X-point finder, `trace_lcfs` |
| `src/units.jl` | per-quantity unit normalization (M3D / cgs / SI) |
| `src/normalization.jl` | flux coordinates ψ_N, ρ_pol |
| `src/export_imas.jl` | FSA export orchestration, IMAS IR assembly |
| `src/imas_writer.jl` | OMAS/IMAS HDF5 serializer (atomic writes) |
| `src/equilibrium.jl` | equilibrium interop (gEQDSK, MXH, IMAS, COCOS) |
| `src/ascot5.jl` | ASCOT5 input writer |

`io.jl` and `elements.jl` are the only M3D-C1-specific layers; the reduction,
axis and coordinate layers are format-independent.

## Tests

```julia
Pkg.test("M3DC1Reader")
```

Unit tests run on synthetic meshes and need no data file. If a `C1.h5` is
available (default `/scratch/gpfs/myoo/m3d_smoke/C1.h5`, or set
`ENV["M3DC1_TEST_FILE"]`), end-to-end slice and export tests also run.
