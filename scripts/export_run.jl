#!/usr/bin/env julia
# Batch runner: point at an M3D-C1 run folder (or a C1.h5 directly) and write the
# OMAS/IMAS HDF5 (and, by default, the ASCOT5 inputs) into an output folder. The
# multi-GB `time_NNN.h5` slice files are read transparently via C1.h5's external
# links (resolved relative to C1.h5), so only the run folder needs to be on disk.
#
# Usage:
#   julia --project=. scripts/export_run.jl <run_dir | C1.h5> [--key=value ...]
#
# The only positional argument is the input: a run folder containing C1.h5, or the
# C1.h5 path itself. Everything else is a --key=value option:
#
#   --outdir=<path>     output folder for ALL outputs (default: <run_dir>, created
#                       if missing): the axisymmetric OMAS file (M3DC1_axisym.h5)
#                       and, when --ascot=true, the ASCOT5 inputs (ascot/ascot_input_<idx>.h5).
#   --ascot=true        also write an ASCOT5 input per exported slice, reusing its
#                       FSA (default: true; --ascot=false writes IMAS only)
#   --nbins=128         FSA radial bins
#   --ngrid=200         R-Z evaluation grid (ngrid × ngrid)
#   --cocos=mhdsimdb    mhdsimdb (NIMROD-DB drop-in, default) | 11 (standard IMAS) | raw (per-radian)
#   --cocos11=true      also write M3DC1_axisym_cocos11.h5, the same export relabelled
#                       to standard IMAS COCOS-11 (default: true when --cocos=mhdsimdb).
#                       Shares the reduction, so it costs one extra write, not a re-run.
#   --fsa=cumulative    ratio-profile FSA estimator: cumulative (smooth, default) | bin (raw)
#   --fsa_window=4      cumulative regression window (smaller = follows pedestal tighter)
#   --pulse=<int>       dataset_description.data_entry.pulse
#   --slices=0,12,24    comma-separated timeslice indices (default: all)
#
# NOTE: this runner picks workflow-oriented defaults that differ from the library
# `export_imas` (which keeps standards-oriented defaults):
#   • --fsa=cumulative  (de-noised ne/Te/dB-over-B) vs library fsa_method=:bin
#   • --cocos=mhdsimdb  (NIMROD-DB drop-in layout)   vs library cocos=11 (IMAS)
# Pass --fsa=bin / --cocos=11 to reproduce the library defaults.
#
# Examples:
#   julia --project=. scripts/export_run.jl /scratch/run_042
#   julia --project=. scripts/export_run.jl /scratch/run_042 --outdir=/scratch/out
#   julia --project=. scripts/export_run.jl /scratch/run_042 --outdir=/scratch/out --slices=24
#   julia --project=. scripts/export_run.jl /scratch/run_042 --ascot=false
#   julia --project=. scripts/export_run.jl /scratch/run_042 --fsa=bin

using M3DC1Reader

function main(args)
    # One positional (the input); everything else is a --key=value option.
    input = ""
    opts = Dict{String, String}()
    for a in args
        if startswith(a, "--")
            kv = a[3:end]
            if occursin('=', kv)
                k, v = split(kv, '='; limit = 2)
                opts[String(k)] = String(v)
            else
                opts[kv] = "true"          # bare flag, e.g. --ascot
            end
        elseif isempty(input)
            input = a
        else
            error("unexpected positional '$a' — options use --key=value")
        end
    end
    isempty(input) && error(
        "usage: export_run <run_dir | C1.h5> [--outdir=<path>] [--ascot=true|false] " *
            "[--nbins= --ngrid= --cocos= --cocos11= --fsa= --fsa_window= --slices= --pulse=]"
    )

    # Resolve the C1.h5 path: accept either a folder or the file itself.
    c1 = isdir(input) ? joinpath(input, "C1.h5") : input
    isfile(c1) || error("no C1.h5 found at: $c1")
    rundir = dirname(abspath(c1))
    # Output folder holds the axisymmetric OMAS file; the ASCOT5 inputs go in an
    # `ascot/` subfolder to keep it tidy (default out folder: rundir).
    out_folder = get(opts, "outdir", rundir)
    mkpath(out_folder)
    # Axisymmetric OMAS/IMAS-DD output (the n=0 FSA reduction; the 3D fields go to
    # the ASCOT5 inputs). Named like the NIMROD DB entries' <CODE>_ prefix.
    out = joinpath(out_folder, "M3DC1_axisym.h5")

    # Parse options with defaults matching export_imas.
    nbins = parse(Int, get(opts, "nbins", "128"))
    ngrid = parse(Int, get(opts, "ngrid", "200"))
    pulse = haskey(opts, "pulse") ? parse(Int, opts["pulse"]) : nothing
    # This runner defaults to the MHDsimDB drop-in layout (the library default is
    # cocos=11), so exports land in the NIMROD-derived disruption-database format
    # by default; pass --cocos=11 for the standard IMAS layout, --cocos=raw for the
    # untouched M3D-C1 per-radian ψ.
    cocos = let c = get(opts, "cocos", "mhdsimdb")
        c == "11" ? 11 : c == "mhdsimdb" ? :mhdsimdb :
            (c == "raw" || c == "nothing") ? nothing :
            error("cocos must be 11 | mhdsimdb | raw, got $c")
    end
    # This runner defaults to the smooth cumulative estimator (the library default
    # is :bin); pass --fsa=bin to opt back into the raw per-bin profiles.
    fsa_method = let m = get(opts, "fsa", "cumulative")
        m == "cumulative" ? :cumulative : m == "bin" ? :bin :
            error("fsa must be cumulative | bin, got $m")
    end
    fsa_window = parse(Float64, get(opts, "fsa_window", "4"))
    # ASCOT5 emission (one input per exported slice, reusing each slice's FSA) is
    # ON by default; --ascot=false writes only the IMAS file.
    ascot5 = let a = get(opts, "ascot", "true")
        a in ("true", "on", "yes", "1") ? true :
            a in ("false", "off", "no", "0") ? false :
            error("--ascot must be true|false, got $a")
    end
    # Optional COCOS-11 twin of the same export. ON by default when the primary
    # output is the (mixed-convention) MHDsimDB layout: the reduction is shared,
    # so the twin costs one extra write, and it saves a full re-run if a
    # consumer turns out to want IMAS-standard psi.
    want_c11 = let a = get(opts, "cocos11", cocos === :mhdsimdb ? "true" : "false")
        a in ("true", "on", "yes", "1") ? true :
            a in ("false", "off", "no", "0") ? false :
            error("--cocos11 must be true|false, got $a")
    end
    cocos11_out = want_c11 ? joinpath(out_folder, "M3DC1_axisym_cocos11.h5") : ""

    file = M3DC1File(c1)
    all_slices = list_timeslices(file)
    slices = haskey(opts, "slices") ?
        parse.(Int, split(opts["slices"], ',')) : all_slices

    @info "M3DC1Reader export" run = rundir out_folder = out_folder slices = length(slices) nbins ngrid cocos pulse fsa_method fsa_window ascot5 cocos11 = want_c11

    export_imas(
        file, out;
        slices = slices, nbins = nbins, ngrid = ngrid,
        cocos = cocos, pulse = pulse,
        fsa_method = fsa_method, fsa_window = fsa_window,
        ascot5 = ascot5, ascot5_dir = joinpath(out_folder, "ascot"),
        cocos11_path = cocos11_out, verbose = true
    )

    return out
end

main(ARGS)
