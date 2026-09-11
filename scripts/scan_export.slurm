#!/bin/bash
# SLURM template: run scripts/scan_export in parallel across N ranks (one SLURM
# allocation, N tasks). Each rank processes its round-robin 1/N share of the run
# list; runs are independent (each writes its own $OUTBASE/<name>) so ranks never
# collide. Pass the run list as ARGUMENT(s) (or edit DEFAULT_LIST in-file):
#     sbatch scripts/scan_export.slurm /path/to/runs.txt   # pass the list (recommended)
#     sbatch scripts/scan_export.slurm /path/to/runs_dir   # or a parent dir to scan
#     sbatch scripts/scan_export.slurm                     # else uses DEFAULT_LIST
# (adjust --ntasks etc. with sbatch flags too, e.g.  sbatch --ntasks=16 … <list>)
#
# Re-run only ONE step without editing the file (env passes through to the ranks):
#     sbatch --export=ALL,FORCE=1 scripts/scan_export.slurm <list>                # redo everything
#     sbatch --export=ALL,DO_EXPORT=0,FORCE=1 scripts/scan_export.slurm <list>    # re-PLOT only (reuse exports)
#     sbatch --export=ALL,DO_PLOT=0,FORCE=1   scripts/scan_export.slurm <list>    # re-EXPORT only
# (plot-only is cheap — it reuses each run's M3DC1_axisym.h5 and just redraws the frames.)
#
#SBATCH --job-name=m3dc1_scan
#SBATCH --nodes=1
#SBATCH --ntasks=26              # = number of parallel ranks (NRANKS)
#SBATCH --cpus-per-task=1       # cores each export/plot gets
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:30:00
#SBATCH --output=scan_%j.log    # combined log (srun --label prefixes each rank)

set -euo pipefail

# Sanitize the HPC env before any Julia runs: libgomp (pulled in by Julia's native
# deps) aborts on a non-numeric OMP_NUM_THREADS, which some stellar modules leave
# set; keep GR headless too. (export_run/plot_equilibrium also self-guard.)
[[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]] || export OMP_NUM_THREADS=1
export GKSwstype="${GKSwstype:-100}"
# GPFS/Lustre: HDF5≥1.10 takes an flock() on every file it opens (even read-only);
# on parallel filesystems that lock can fail under many concurrent opens ("unable
# to open file"). No two ranks ever write the same file, so disabling it is safe.
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# ---- EDIT THESE (the run list is normally passed as sbatch argument(s)) ------
REPO=/scratch/gpfs/myoo/M3DC1Reader          # clone containing scripts/
export OUTBASE="${OUTBASE:-/scratch/gpfs/myoo/M3DC1_DB}"     # output root ($OUTBASE/<run name>/)
export EXPORT_OPTS=""                        # e.g. "--slices=0,24"; "" = all slices
export PLOT_OPTS=""                          # e.g. "--fps=12"
export FORCE="${FORCE:-0}"                   # 1 = overwrite existing outputs (else skip/resume);
                                             #   also settable per-submit: sbatch --export=ALL,FORCE=1 …
export DO_EXPORT="${DO_EXPORT:-1}"           # 0 = skip export step (e.g. plot-only re-run)
export DO_PLOT="${DO_PLOT:-1}"               # 0 = skip plot step   (e.g. export-only re-run)
DEFAULT_LIST=/home/myoo//m3dc1_spi_folder_list_on_Stellar.txt    # used only when no argument is given

# -----------------------------------------------------------------------------

# Inputs = the sbatch positional args (a list file / run folder / parent dir),
# else DEFAULT_LIST.  `sbatch scripts/scan_export.slurm <list>` → "$@" = <list>.
if [[ $# -gt 0 ]]; then inputs=("$@"); else inputs=("$DEFAULT_LIST"); fi
echo "scan_export.slurm: inputs=[${inputs[*]}]  OUTBASE=$OUTBASE  ntasks=${SLURM_NTASKS:-1}"
[[ -e "${inputs[0]}" ]] || { echo "ERROR: input '${inputs[0]}' not found" >&2; exit 1; }

# Julia on the compute node. `module` is not defined in a non-interactive batch
# shell until modules.sh is sourced, and `module load julia` alone fails with
# "No default version defined" — so source it and pin the version. JULIA_MODULE
# can override (must match what the depot in ~/.julia was precompiled against).
if ! command -v julia >/dev/null 2>&1; then
    [[ -r /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh
    module load "${JULIA_MODULE:-julia/1.12.1}" || {
        echo "ERROR: could not load ${JULIA_MODULE:-julia/1.12.1}" >&2; exit 1; }
fi
command -v julia >/dev/null 2>&1 || { echo "ERROR: julia not on PATH" >&2; exit 1; }
echo "julia: $(command -v julia)  ($(julia --version))"

# Warm the precompile caches ONCE so the N ranks don't race on first-time
# precompilation of the package / plotting deps (concurrent first compiles of the
# same package thrash the shared depot). Both envs export_run/plot use:
echo "warmup: precompiling Julia deps…"
julia --project="$REPO" -e 'using M3DC1Reader' >/dev/null
# plotting deps only when the plot step will actually run (DO_PLOT=0 skips them,
# and a missing Plots in the default env must not abort an export-only scan)
[[ "$DO_PLOT" == "1" ]] && { julia -e 'using Plots, HDF5' >/dev/null || {
    echo "WARN: plotting deps unavailable; plot step will fail per-run" >&2; }; }

# Launch NRANKS = $SLURM_NTASKS workers; scan_export auto-detects the rank from
# SLURM_PROCID/SLURM_NTASKS and takes its share. --label prefixes lines by task.
#   --wait=0            : do NOT kill the still-running ranks when the first
#                         (fastest) rank exits — shards finish at different times,
#                         so the srun default WaitTime would otherwise SIGKILL the
#                         slow ranks mid-run.
#   --kill-on-bad-exit=0: one rank's failed run must not abort the other ranks.
srun --label --wait=0 --kill-on-bad-exit=0 --ntasks="$SLURM_NTASKS" \
    "$REPO/scripts/scan_export" "${inputs[@]}"

# ---- Alternative (RECOMMENDED for many/uneven runs): a JOB ARRAY ------------
# A job array runs each shard as an INDEPENDENT job, so (a) a fast shard finishing
# never kills the others (the very problem --wait=0 works around above), and
# (b) SLURM schedules them dynamically under a concurrency cap — no rank sits idle
# waiting for the slowest. Best when runs vary a lot in size. Set the array size to
# the number of runs so each task handles exactly ONE (get the count from a
# dry-run:  scripts/scan_export -n <list>):
#     sbatch --array=0-26%10 scripts/scan_export.slurm <list>   # 27 runs, ≤10 at once
# then replace the `srun …` block above with just:
#     "$REPO/scripts/scan_export" "${inputs[@]}"
# scan_export reads SLURM_ARRAY_TASK_ID/_COUNT as RANK/NRANKS automatically.
# (Per-task logs: #SBATCH --output=scan_%A_%a.log)
