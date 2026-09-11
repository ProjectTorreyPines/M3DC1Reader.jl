# imas_writer.jl — generic OMAS/IMAS HDF5 serializer.
#
# Writes a flat Dict of dotted IMAS paths into the OMAS nested-group HDF5
# layout (0-based integer subgroups, no attributes, fixed strings, Float64
# arrays, N-D arrays transposed for numpy/h5py row-major). Physics-agnostic.

"""
    write_omas_h5(path, ir::AbstractDict) -> path

Serialize `ir` (keys = dotted IMAS paths, e.g.
`"core_profiles.profiles_1d.0.electrons.temperature"`) into an OMAS-compatible
HDF5 file at `path`. Overwrites `path`.

The write is **atomic**: it goes to a unique temporary file in the same
directory and is `mv`d into place only after the HDF5 file is closed cleanly.
So `path` is either absent or a complete, readable file — never a half-written
one. This matters for batch scans, where an interrupted or duplicated writer
would otherwise leave a truncated file whose HDF5 object headers fail to
deserialize ("bad object header version number") and which a resume pass would
then happily *skip* as already done. The temp file is removed on failure.
"""
function write_omas_h5(path::AbstractString, ir::AbstractDict)
    out = String(path)
    dir = dirname(out);  isempty(dir) && (dir = ".")
    mkpath(dir)
    # same directory ⇒ the rename is atomic (a cross-filesystem mv would not be)
    tmp = joinpath(dir, "." * basename(out) * ".tmp-" * string(getpid()))
    try
        h5open(tmp, "w") do f
            for (key, val) in ir
                _write_omas_leaf!(f, String(key), val)
            end
        end
        mv(tmp, out; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return out
end

function _write_omas_leaf!(f::Union{HDF5.File, HDF5.Group}, key::AbstractString, val)
    segs = split(key, '.')
    parent = f
    @inbounds for s in @view segs[1:(end - 1)]
        sn = String(s)
        if haskey(parent, sn)
            parent = parent[sn]
            parent isa HDF5.Group ||
                error("write_omas_h5: path collision at '$sn' in key '$key' — a leaf was already written there")
        else
            parent = create_group(parent, sn)
        end
    end
    leaf = String(segs[end])
    (haskey(parent, leaf) && parent[leaf] isa HDF5.Group) &&
        error("write_omas_h5: path collision at '$leaf' in key '$key' — a group already exists there")
    return _write_omas_value!(parent, leaf, val)
end

# N-D arrays are stored transposed so the on-disk C/row-major order matches
# numpy's interpretation (a Julia (nR,nZ) array reads back as (nR,nZ) in h5py).
_omas_layout(A::AbstractArray) =
    ndims(A) ≥ 2 ? permutedims(A, reverse(ntuple(identity, ndims(A)))) : A

function _write_omas_value!(parent, name::AbstractString, val)
    if val isa AbstractString
        parent[name] = String(val)
    elseif val isa Integer
        parent[name] = Int64(val)
    elseif val isa Real
        parent[name] = Float64(val)
    elseif val isa AbstractArray{<:Real}
        parent[name] = _omas_layout(Array{Float64}(val))
    else
        error("write_omas_h5: unsupported value type $(typeof(val)) at leaf '$name'")
    end
    return nothing
end
