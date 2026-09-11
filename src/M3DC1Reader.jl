"""
    M3DC1Reader

Read M3D-C1 `C1.h5` simulation output and work with its native C¹ reduced
quintic Hermite finite-element fields directly in Julia — reading the mesh,
fields, and scalars; evaluating fields and their derivatives at arbitrary
points or grids; locating critical points; and applying unit normalizations.

It plays, in Julia, a role analogous to parts of `fusion-io` and of the
PPPL `visM3D` MATLAB tools (their non-visualization core), and serves as the
M3D-C1 counterpart to the NIMROD workflow in `disruption_database_tools`.

## Current capabilities
- **IO** (`io.jl`): open a file, list time slices, read fields/scalars/mesh.
- **FEM evaluation** (`elements.jl`): analytic field value and derivatives in
  element-local coordinates, global↔local mapping, element location.
- **Magnetic axis** (`magaxis.jl`): `find_axis_newton`, a reimplementation of
  M3D-C1's internal `magaxis` Newton iteration on the FEM polynomial.
- **Units** (`units.jl`): selectable per-quantity normalization
  (`unit_factor`, `to_units`) between M3D / cgs / SI systems.
- **Coordinates** (`normalization.jl`): ψ_N, ρ_pol.
- **Reductions** (`reductions.jl`): one end-to-end use case so far — reduce a
  3D field to a 2D axisymmetric (R-Z) field and a 1D radial profile:

      3D Hermite field  ──average_toroidal_axisymmetric──▶  n=0 axisymmetric
                        ──interpolate_axisym_to_grid──────▶  2D R-Z field
                        ──reduce_1d_psi_func──────────────▶  1D profile vs ρ_pol

More general functionality (e.g. additional field/coordinate utilities and a
full cgs/mks unit system, following `visM3D`) may be added over time.

## Design
The format-specific layers (`io.jl`, `elements.jl`) are kept separate from the
more general analysis layers (`reductions.jl`, `magaxis.jl`, `normalization.jl`)
so the latter can be lifted into a multi-code package later, with M3D-C1 as one
backend.
"""
module M3DC1Reader

using HDF5
using FastInterpolations
using AdaptiveArrayPools

include("elements.jl")        # quintic Hermite FEM math + element location
include("units.jl")          # M3D/cgs/SI unit registry
include("normalization.jl")   # ψ_N, ρ_pol coordinates
include("io.jl")              # C1.h5 reader
include("reductions.jl")      # toroidal avg, 2D eval, 1D adjoint
include("magaxis.jl")         # Newton O-point finder (plane-1, imethod=0)
include("find_critical_points.jl")  # damped-Newton O/X-point finder (∇ψ=0)
include("imas_writer.jl")     # generic OMAS/IMAS HDF5 serializer
include("export_imas.jl")     # FSA-export orchestration + IR assembly
include("equilibrium.jl")    # experimental equilibrium-interop path
include("ascot5.jl")          # ASCOT5 input-file writer (Pipeline ①)

# ---- IO / reader ----
export M3DC1File, elems_plane, list_timeslices, read_field, read_timeslice, read_timeslice!,
    normalization, te_to_eV_factor, limiter_points, scalar_names, read_scalar,
    read_pellets

# ---- reductions ----
export average_toroidal_axisymmetric, average_toroidal_axisymmetric!,
    build_grid_to_element_map, interpolate_axisym_to_grid,
    interpolate_axisym_gradient_to_grid,
    interpolate_at_zeta_to_grid, deltab_over_b_rz,
    ascot5_bfield, write_ascot5,
    reduce_1d_psi_func, pipeline_3d_to_1d

# ---- magnetic axis / critical points / LCFS ----
export mesh_zone_boundary_rz, elem_zones
export find_axis_newton, locate_element, trace_lcfs
export damped_newton_2d, critical_point_system, classify_critical,
    find_critical_point, find_o_point, find_x_point, find_lcfs

# ---- element-level FEM primitives ----
export global_to_local, is_in_element_local,
    eval_axisym_at_local, eval_axisym_at, eval_at_local, eval_psi_and_derivs,
    mesh_boundary_rz, MI, NI

# ---- units ----
export M3DNormalization, unit_factor, to_units, unit_label, available_quantities

# ---- flux coordinates ----
export psi_to_psi_norm, psi_n_to_rho_pol

# ---- IMAS export ----
export write_omas_h5, assemble_ir, reduce_axisym_slice, export_imas

# ---- equilibrium interop (experimental) ----
export M3DAxisymField, axisym_field, to_mxh, to_geqdsk, to_imas, to_cocos, fsa_imas

end # module
