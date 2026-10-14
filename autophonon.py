#!/usr/bin/env python
"""Automated QE + phonopy phonon workflow.

Given a compound name, a crystal structure (CIF, POSCAR, pw.x input, ... -
anything ASE reads) and a directory of UPF pseudopotentials, this script runs

  1. ecutwfc convergence  (scf at fixed dual, energy/pressure/forces vs cutoff)
  2. ecutrho convergence  (scf at the converged ecutwfc, scanning the dual)
  3. vc-relax             (repeated until the final-scf pressure is small)
  4. phonopy displacements in a supercell (+ ph.x Born charges / epsilon for
     insulators, used for the LO-TO non-analytic correction)
  5. phonon dispersion + projected DOS plot

Each stage is submitted to Slurm and the driver waits for it. Progress is kept
in <workdir>/state.json and finished outputs are never recomputed, so a run
can be stopped and restarted at any point.

Usage (with the `autophonon` wrapper on your PATH; otherwise
`python autophonon.py ...`):

  autophonon start GaAs.cif                 structure file
  autophonon start GaAs:zincblende:5.653    or built: Formula:structure:a[:c]
  autophonon start Al:fcc:4.05 --metal      any option as a flag
        writes <name>.yaml (if missing) and submits the driver job
  autophonon status                         progress of the run in this folder
  autophonon start GaAs.yaml                resubmit after a change or failure

  Lower level: init (write a config only), submit <config>, run <config>
  [--dry-run] (foreground driver; --dry-run writes the next stage's inputs
  without submitting). Pseudopotentials default to $AUTOPHONON_PSEUDO_DIR,
  else ~/pseudo/dojo-pbe.
"""
import argparse
import glob
import itertools
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

import numpy as np

BOHR = 0.529177210903          # Angstrom
THZ_TO_CM = 33.35641
RY_PER_BOHR3_TO_KBAR = 147105.08

# Categorical series colors (fixed order) for the element-projected DOS
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Inputs: pseudopotentials and structure
# ----------------------------------------------------------------------------
def read_upf_header(path):
    """Return element, pseudo type and functional from a UPF (v1 or v2) file."""
    with open(path, errors="ignore") as f:
        head = f.read(20000)
    info = {"file": os.path.basename(path)}
    m = re.search(r'element\s*=\s*"\s*([A-Za-z]{1,2})\s*"', head)
    if m:                                   # UPF v2 attributes
        info["element"] = m.group(1)
        t = re.search(r'pseudo_type\s*=\s*"\s*(\w+)\s*"', head)
        info["type"] = t.group(1).upper() if t else "NC"
        fn = re.search(r'functional\s*=\s*"([^"]*)"', head)
        info["functional"] = fn.group(1).strip() if fn else "?"
        z = re.search(r'z_valence\s*=\s*"\s*([-+.\dEe]+)\s*"', head)
        info["zval"] = float(z.group(1)) if z else None
    else:                                   # UPF v1 <PP_HEADER> lines
        m = re.search(r"<PP_HEADER>\s*\n\s*\S+.*\n\s*([A-Za-z]{1,2})\s+Element", head)
        if not m:
            raise ValueError(f"Cannot find element in {path}")
        info["element"] = m.group(1)
        info["type"] = ("PAW" if "PAW" in head[:3000] else
                        "US" if re.search(r"\bUS\b", head[:3000]) else "NC")
        info["functional"] = "?"
        z = re.search(r"([-+.\dEe]+)\s+Z valence", head)
        info["zval"] = float(z.group(1)) if z else None
    info["element"] = info["element"].capitalize()
    if info["type"] in ("SL", "1/R"):
        info["type"] = "NC"
    return info


def read_pseudos(pseudo_dir):
    pseudos = {}
    files = sorted(glob.glob(os.path.join(pseudo_dir, "*.upf")) +
                   glob.glob(os.path.join(pseudo_dir, "*.UPF")))
    for path in files:
        info = read_upf_header(path)
        if info["element"] in pseudos:
            raise ValueError(f"Two pseudopotentials for {info['element']}: "
                             f"{pseudos[info['element']]['file']} and {info['file']}")
        pseudos[info["element"]] = info
    if not pseudos:
        raise ValueError(f"No .upf files in {pseudo_dir}")
    return pseudos


def standardize(lattice, positions, numbers, symprec):
    """Seekpath standardization: conventional + primitive cells and the k-path.

    Lattices are row vectors in Angstrom. The primitive cell is used for the
    convergence tests, the relaxation and the Born charges; the conventional
    cell is the phonopy unit cell, with primitive_matrix chosen so that
    phonopy's primitive cell is exactly seekpath's (so the q-path coordinates
    are in the right basis).
    """
    import seekpath
    sp = seekpath.get_path((lattice, positions, numbers), with_time_reversal=True,
                           symprec=symprec)
    return {
        "spacegroup": f"{sp['spacegroup_international']} ({sp['spacegroup_number']})",
        "conv_lattice": np.array(sp["conv_lattice"]).tolist(),
        "conv_positions": np.array(sp["conv_positions"]).tolist(),
        "conv_numbers": [int(n) for n in sp["conv_types"]],
        "prim_lattice": np.array(sp["primitive_lattice"]).tolist(),
        "prim_positions": np.array(sp["primitive_positions"]).tolist(),
        "prim_numbers": [int(n) for n in sp["primitive_types"]],
        # rows of primitive_lattice = P @ conv_lattice -> phonopy matrix is P^T
        "prim_matrix": np.array(sp["primitive_transformation_matrix"]).T.tolist(),
        "point_coords": {k: list(v) for k, v in sp["point_coords"].items()},
        "path": [list(p) for p in sp["path"]],
    }


def symbols_of(numbers):
    from ase.data import chemical_symbols
    return [chemical_symbols[n] for n in numbers]


def rattle(lattice, positions, amplitude, seed=7):
    """Fractional positions randomly displaced by up to `amplitude` Angstrom per
    Cartesian component (fixed seed, so every cutoff sees the same structure).
    Gives the convergence scans nonzero forces to compare."""
    lat = np.array(lattice, dtype=float)
    disp = np.random.default_rng(seed).uniform(-amplitude, amplitude, (len(positions), 3))
    return (np.array(positions, dtype=float) + disp @ np.linalg.inv(lat)).tolist()


def repeat_cell(lattice, positions, numbers, n):
    """n x n x n supercell (lattice rows in Angstrom, fractional positions)."""
    shifts = np.array(list(itertools.product(range(n), repeat=3)), dtype=float)
    pos = np.array(positions, dtype=float)
    new_pos = ((pos[:, None, :] + shifts[None, :, :]) / n).reshape(-1, 3)
    return ((np.array(lattice, dtype=float) * n).tolist(), new_pos.tolist(),
            [int(z) for z in numbers for _ in range(len(shifts))])


def kmesh(lattice, kspacing):
    """Monkhorst-Pack mesh with |b_i| / n_i <= kspacing (|b| includes 2*pi)."""
    rec = 2 * np.pi * np.linalg.inv(np.array(lattice)).T
    return [max(1, math.ceil(np.linalg.norm(b) / kspacing - 1e-6)) for b in rec]


def num_pools(lattice, positions, numbers, mesh, ntasks, max_pools):
    """Largest divisor of ntasks <= min(max_pools, irreducible k-points)."""
    import spglib
    mapping, _ = spglib.get_ir_reciprocal_mesh(
        mesh, (lattice, positions, numbers), is_shift=[0, 0, 0])
    nk = len(np.unique(mapping))
    cap = max(1, min(max_pools, nk))
    return max(d for d in range(1, cap + 1) if ntasks % d == 0)


# ----------------------------------------------------------------------------
# QE input writing and output parsing
# ----------------------------------------------------------------------------
def pw_input(calc, prefix, lattice, positions, symbols, ecutwfc, ecutrho, mesh,
             cfg, outdir, conv_thr=1e-10, extra_control="", extra_system="",
             extra_sections=""):
    """pw.x input with ibrav=0; lattice in Angstrom, positions fractional."""
    from ase.data import atomic_masses, atomic_numbers
    species = list(dict.fromkeys(symbols))
    occ = ("  occupations = 'smearing'\n  smearing    = 'mv'\n"
           f"  degauss     = {cfg['degauss']}\n" if cfg["metal"]
           else "  occupations = 'fixed'\n")
    s = (f"&CONTROL\n  calculation = '{calc}'\n  prefix      = '{prefix}'\n"
         f"  outdir      = '{outdir}'\n  pseudo_dir  = '{cfg['pseudo_dir']}'\n"
         f"  tprnfor     = .true.\n  tstress     = .true.\n{extra_control}/\n"
         f"&SYSTEM\n  ibrav       = 0\n  nat         = {len(symbols)}\n"
         f"  ntyp        = {len(species)}\n  ecutwfc     = {ecutwfc}\n"
         f"  ecutrho     = {ecutrho}\n{occ}{extra_system}/\n"
         f"&ELECTRONS\n  conv_thr    = {conv_thr:.1e}\n  mixing_beta = 0.7\n/\n"
         f"{extra_sections}")
    s += "CELL_PARAMETERS angstrom\n"
    for v in lattice:
        s += "  %16.10f %16.10f %16.10f\n" % tuple(v)
    s += "ATOMIC_SPECIES\n"
    for sym in species:
        s += f"  {sym:3s} {atomic_masses[atomic_numbers[sym]]:10.5f}  {cfg['pseudos'][sym]['file']}\n"
    s += "ATOMIC_POSITIONS crystal\n"
    for sym, p in zip(symbols, positions):
        s += f"  {sym:3s} %16.10f %16.10f %16.10f\n" % tuple(p)
    s += "K_POINTS automatic\n  %d %d %d 0 0 0\n" % tuple(mesh)
    return s


def parse_born(path):
    """(epsilon_inf eigen-ish diagonal, Z* array) from a phonopy BORN file.

    Format: optional factor/comment line, a line with the 9 components of
    epsilon_inf, then one line of 9 Z* components per independent atom."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split("#")[0].split()
            if len(parts) == 9:
                rows.append([float(x) for x in parts])
    if len(rows) < 2:
        raise ValueError(f"{path}: expected an epsilon line and at least one Z* line "
                         "with 9 numbers each (phonopy BORN format)")
    eps = np.array(rows[0]).reshape(3, 3)
    return np.diag(eps), np.array(rows[1:])


def born_check(scf_out, eps_diag, zstar, min_gap=0.5, max_eps=50.0):
    """Band gap from the Born scf and a warning when LO-TO can't be trusted."""
    gap = None
    if os.path.exists(scf_out):
        with open(scf_out, errors="ignore") as f:
            m = re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s+(\S+)\s+(\S+)",
                           f.read())
        if m:
            gap = float(m[-1][1]) - float(m[-1][0])
    problems = []
    if gap is not None and gap < min_gap:
        problems.append(f"band gap only {gap:.2f} eV")
    if eps_diag.max() > max_eps:
        problems.append(f"epsilon_inf = {eps_diag.max():.0f}")
    warning = None
    if problems:
        warning = (", ".join(problems) + ": the LO-TO splitting is not reliable "
                   "(DFT gap problem). Use born_volume: input to compute it at the input "
                   "(e.g. experimental) volume, or born_file: with literature values.")
    return {"gap_ev": gap, "eps_inf_max": float(eps_diag.max()),
            "zstar_max": float(np.abs(zstar).max()), "warning": warning}


def output_done(path):
    """True if a pw.x/ph.x output finished cleanly."""
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as f:
        text = f.read()
    return "JOB DONE." in text and "convergence NOT achieved" not in text


def parse_pw(path):
    with open(path, errors="ignore") as f:
        text = f.read()
    res = {}
    e = re.findall(r"^!\s+total energy\s+=\s+(\S+)\s+Ry", text, re.M)
    res["energy"] = float(e[-1]) if e else None
    p = re.findall(r"P=\s*(\S+)", text)
    res["pressure"] = float(p[-1]) if p else None
    blocks = text.split("Forces acting on atoms")
    if len(blocks) > 1:
        res["forces"] = [[float(x) for x in m] for m in re.findall(
            r"atom\s+\d+\s+type\s+\d+\s+force\s+=\s+(\S+)\s+(\S+)\s+(\S+)", blocks[-1])]
    res["nat"] = int(re.search(r"number of atoms/cell\s+=\s+(\d+)", text).group(1))
    if "Begin final coordinates" in text:
        res["final"] = parse_final_coordinates(
            text.split("Begin final coordinates")[1].split("End final coordinates")[0])
    res["bfgs_converged"] = "bfgs converged" in text
    return res


def parse_final_coordinates(block):
    lines = block.splitlines()
    lattice, positions, symbols = None, [], []
    for i, line in enumerate(lines):
        if line.startswith("CELL_PARAMETERS"):
            vec = np.array([[float(x) for x in lines[i + j].split()] for j in (1, 2, 3)])
            if "angstrom" in line:
                lattice = vec
            elif "bohr" in line:
                lattice = vec * BOHR
            else:
                alat = float(re.search(r"alat\s*=\s*([\d.]+)", line).group(1))
                lattice = vec * alat * BOHR
        if line.startswith("ATOMIC_POSITIONS"):
            if "crystal" not in line:
                raise ValueError("Expected crystal ATOMIC_POSITIONS in vc-relax output")
            for l2 in lines[i + 1:]:
                parts = l2.split()
                if len(parts) < 4 or not re.match(r"[A-Z]", parts[0]):
                    break
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
    return {"lattice": lattice.tolist(), "positions": positions, "symbols": symbols}


# ----------------------------------------------------------------------------
# Slurm
# ----------------------------------------------------------------------------
def walltime_seconds(text):
    """'D-HH:MM:SS', 'HH:MM:SS' or 'MM:SS' -> seconds."""
    days, _, hms = text.rpartition("-")
    parts = [int(x) for x in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


_QOS_MAXWALL = {}


def qos_maxwall(qos):
    """Walltime limit of a Slurm QOS in seconds (None if unlimited/unknown)."""
    if qos not in _QOS_MAXWALL:
        limit = None
        try:
            out = subprocess.run(["sacctmgr", "-nP", "show", "qos", qos, "format=maxwall"],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            if out:
                limit = walltime_seconds(out.splitlines()[0])
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        _QOS_MAXWALL[qos] = limit
    return _QOS_MAXWALL[qos]


def capped_walltime(walltime, qos):
    """Walltime shortened to the QOS limit (e.g. 4 h on standby)."""
    limit = qos_maxwall(qos)
    if limit and walltime_seconds(walltime) > limit:
        h, m = divmod(limit // 60, 60)
        return f"{h:02d}:{m:02d}:00"
    return walltime


def job_script(cfg, name, runs, walltime, outdir):
    """runs: list of (executable, npool, input, output) executed in order."""
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={name}",
             f"#SBATCH --account={cfg['account']}",
             f"#SBATCH --partition={cfg['partition']}",
             f"#SBATCH --qos={cfg['qos']}",
             "#SBATCH --nodes=1",
             f"#SBATCH --ntasks-per-node={cfg['ntasks']}",
             f"#SBATCH --time={walltime}",
             "#SBATCH --output=%x.%j.log",
             "#SBATCH --error=%x.%j.err",
             "",
             'cd "$SLURM_SUBMIT_DIR"',
             "module purge"]
    lines += [f"module load {m}" for m in cfg["modules"].split()]
    lines += ["export OMP_NUM_THREADS=1", f'mkdir -p "{outdir}"', ""]
    for exe, npool, inp, out in runs:
        lines.append(f'mpirun -np "$SLURM_NTASKS" {exe} -nk {npool} -in {inp} > {out}')
    return "\n".join(lines) + "\n"


def scratch_dir(cfg, directory):
    """QE outdir for a stage directory: the same relative path under cfg['scratch']."""
    return os.path.join(cfg["scratch"], os.path.relpath(directory, cfg["work"]))


class DryRunStop(Exception):
    """Raised once a dry run has written the job scripts of the next stage."""


class WalltimeSignal(Exception):
    """Slurm warned the driver job that its walltime is about to end."""


def submit(cfg, directory, name, runs, walltime):
    """Write <name>.slurm and sbatch it. Returns the job id (None in a dry run)."""
    path = os.path.join(directory, f"{name}.slurm")
    capped = capped_walltime(walltime, cfg["qos"])
    if capped != walltime:
        log(f"  {name}: walltime {walltime} capped to {capped} (qos {cfg['qos']} limit)")
    with open(path, "w") as f:
        f.write(job_script(cfg, name, runs, capped, scratch_dir(cfg, directory)))
    if cfg.get("dry_run"):
        log(f"  dry run: wrote {os.path.relpath(path, cfg['work'])} (not submitted)")
        return None
    out = subprocess.run(["sbatch", "--parsable", os.path.basename(path)],
                         cwd=directory, capture_output=True, text=True, check=True)
    jobid = out.stdout.strip().split(";")[0]
    log(f"  submitted {name} -> job {jobid}")
    return jobid


def job_active(jobid, retries=5):
    """True while the job is queued/running. squeue errors out for job ids that
    were already purged; any other failure (e.g. slurmctld timeout) is retried
    so a transient hiccup isn't mistaken for a finished job."""
    for attempt in range(retries):
        out = subprocess.run(["squeue", "-h", "-j", jobid, "-o", "%T"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip() != ""
        if "Invalid job id" in out.stderr:
            return False
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"squeue keeps failing for job {jobid}: {out.stderr.strip()}")


def job_state(jobid):
    """Final Slurm state of a job (e.g. COMPLETED, TIMEOUT, NODE_FAIL), or ''."""
    try:
        out = subprocess.run(["sacct", "-n", "-X", "-P", "-j", jobid, "-o", "State"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    return lines[-1].split()[0] if lines else ""


# Failures that are the cluster's fault, so resubmitting is worth it
RETRY_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REQUEUED"}


def default_scratch(requested, work):
    """QE outdir root. Supercell wavefunctions are large, so prefer the cluster
    scratch filesystem over the (quota-limited) workdir."""
    if requested:
        return os.path.abspath(requested)
    root = os.environ.get("RCAC_SCRATCH") or os.environ.get("SCRATCH")
    if root and os.path.isdir(root):
        return os.path.join(root, "autophonon", os.path.basename(work))
    return os.path.join(work, "tmp")


def preflight(cfg, pseudos, a):
    """Cheap checks before anything is submitted."""
    if not cfg["dry_run"] and not shutil.which("sbatch"):
        raise RuntimeError("sbatch not found; run on a cluster login node or in a Slurm job")
    try:
        os.makedirs(cfg["scratch"], exist_ok=True)
        probe = os.path.join(cfg["scratch"], f".write_test_{os.getpid()}")
        open(probe, "w").close()
        os.remove(probe)
    except OSError as exc:
        raise RuntimeError(f"Scratch directory {cfg['scratch']} is not writable ({exc}); "
                           "set scratch to a writable location") from None
    for line in quota_lines([cfg["work"], cfg["scratch"]]):
        log(f"  WARNING: {line}")


def quota_lines(paths, threshold=85.0):
    """Quotas above `threshold` percent for filesystems holding `paths`, from the
    RCAC `myquota` command when it exists (rows: type, location, size, limit, use%)."""
    if not shutil.which("myquota"):
        return []
    try:
        out = subprocess.run(["myquota"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    warn = []
    for line in out.splitlines():
        parts = line.split()
        pct = [float(x[:-1]) for x in parts if re.fullmatch(r"\d+(\.\d+)?%", x)]
        if len(parts) < 2 or not pct or pct[0] < threshold:
            continue
        roots = {"home": [os.path.expanduser("~")],
                 "scratch": [os.environ.get("RCAC_SCRATCH", "/scratch")],
                 "depot": [f"/depot/{parts[1]}"]}.get(parts[0], [])
        roots = [os.path.realpath(r) for r in roots]
        if any(os.path.realpath(p) == r or os.path.realpath(p).startswith(r + os.sep)
               for p in paths for r in roots):
            warn.append(f"{parts[0]} {parts[1]} quota at {pct[0]:.0f}% "
                        f"({parts[2]} of {parts[3]}); outputs may fail to write")
    return warn


def read_driver(work):
    try:
        with open(os.path.join(work, "driver.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_driver(work, info):
    path = os.path.join(work, "driver.json")
    with open(path + ".tmp", "w") as f:
        json.dump(info, f, indent=1)
    os.replace(path + ".tmp", path)


def live_driver(work, poll):
    """Description of another driver working on `work`, or None.

    A driver submitted as a Slurm job counts as live while the job is queued or
    running; one started by hand counts as live while its heartbeat is fresh."""
    info = read_driver(work)
    me = os.environ.get("SLURM_JOB_ID")
    jobid = info.get("jobid")
    if jobid and jobid != me and job_active(jobid):
        return f"driver job {jobid}"
    # A driver that ran as a Slurm job is dead once its job is gone; the
    # heartbeat only matters for drivers started by hand (no job id)
    if (info.get("pid") and not jobid and info.get("pid") != os.getpid()
            and time.time() - info.get("heartbeat", 0) < max(600, 5 * poll)):
        return f"driver pid {info['pid']} on {info.get('host', '?')}"
    return None


class Workflow:
    def __init__(self, args):
        self.args = args
        self.work = os.path.abspath(args.workdir or f"{args.name}_auto")
        os.makedirs(self.work, exist_ok=True)
        self.state_file = os.path.join(self.work, "state.json")
        self.state = {}
        if os.path.exists(self.state_file):
            with open(self.state_file) as f:
                self.state = json.load(f)

    def heartbeat(self):
        """Refresh <workdir>/driver.json so other invocations see a live driver."""
        if self.args.dry_run:
            return
        info = read_driver(self.work)
        info.update(pid=os.getpid(), host=socket.gethostname(), heartbeat=time.time(),
                    jobid=os.environ.get("SLURM_JOB_ID") or info.get("jobid"))
        write_driver(self.work, info)

    def save(self):
        with open(self.state_file + ".tmp", "w") as f:
            json.dump(self.state, f, indent=1)
        os.replace(self.state_file + ".tmp", self.state_file)

    def run_stage(self, key, directory, jobs):
        """Make sure every output in `jobs` exists and finished.

        jobs: list of (job_name, runs, walltime). A job is submitted only if
        one of its outputs is missing and no previously submitted job for it
        is still queued. Blocks until all outputs are complete.
        """
        os.makedirs(directory, exist_ok=True)
        ids = self.state.setdefault("jobs", {}).setdefault(key, {})
        for name, runs, walltime in jobs:
            outs = [os.path.join(directory, r[3]) for r in runs]
            if all(output_done(o) for o in outs):
                continue
            if name in ids and job_active(ids[name]):
                log(f"  {name}: job {ids[name]} still queued/running")
                continue
            todo = [r for r, o in zip(runs, outs) if not output_done(o)]
            jobid = submit(self.cfg, directory, name, todo, walltime)
            if jobid:
                ids[name] = jobid
                self.save()
        if self.cfg.get("dry_run"):
            raise DryRunStop(directory)
        while any(job_active(j) for j in ids.values()):
            self.heartbeat()
            time.sleep(self.args.poll)
        # Resubmit jobs lost to node failure/preemption (not ones that ran out
        # of time or crashed: rerunning those would fail the same way)
        retries = self.state.setdefault("retries", {})
        again = []
        for name, runs, walltime in jobs:
            if all(output_done(os.path.join(directory, r[3])) for r in runs) or name not in ids:
                continue
            state = job_state(ids[name])
            if state in RETRY_STATES and retries.get(name, 0) < 2:
                retries[name] = retries.get(name, 0) + 1
                log(f"  {name}: job {ids[name]} ended {state}; resubmitting "
                    f"(retry {retries[name]} of 2)")
                again.append((name, runs, walltime))
        if again:
            self.save()
            return self.run_stage(key, directory, jobs)
        bad = [os.path.join(directory, r[3]) for _, runs, _ in jobs for r in runs
               if not output_done(os.path.join(directory, r[3]))]
        if bad:
            raise RuntimeError(f"Stage {key} failed; incomplete outputs:\n  " +
                               "\n  ".join(bad) + f"\nCheck the .err/.log files in {directory}"
                               " and rerun the script to resubmit.")
        # Wavefunctions/charge are not reused between stages; free the space
        if not self.args.keep_tmp:
            shutil.rmtree(scratch_dir(self.cfg, directory), ignore_errors=True)

    # ------------------------------------------------------------------
    def setup(self):
        a = self.args
        pdir = os.path.join(self.work, "pseudo")
        os.makedirs(pdir, exist_ok=True)
        all_pseudos = read_pseudos(os.path.abspath(a.pseudo_dir))
        if "input" in self.state:
            elements = set(symbols_of(self.state["input"]["numbers"]))
        else:
            from ase.io import read
            if not os.path.exists(a.structure):
                raise FileNotFoundError(f"Structure file not found: {a.structure}")
            atoms = read(a.structure)
            elements = set(atoms.get_chemical_symbols())
        missing = elements - set(all_pseudos)
        if missing:
            raise ValueError(f"No pseudopotential for {sorted(missing)} in {a.pseudo_dir}")
        pseudos = {el: all_pseudos[el] for el in sorted(elements)}
        for info in pseudos.values():
            src = os.path.join(os.path.abspath(a.pseudo_dir), info["file"])
            dst = os.path.join(pdir, info["file"])
            if not os.path.exists(dst):
                shutil.copy(src, dst)
        self.cfg = {
            "pseudo_dir": pdir, "pseudos": pseudos, "metal": a.metal,
            "degauss": a.degauss, "account": a.account, "partition": a.partition,
            "qos": a.qos, "ntasks": a.ntasks, "modules": a.modules,
            "work": self.work,
            "scratch": default_scratch(a.scratch, self.work),
            "dry_run": a.dry_run,
        }
        preflight(self.cfg, pseudos, a)
        if "input" not in self.state:
            std = standardize(atoms.cell.array, atoms.get_scaled_positions(),
                              atoms.numbers, a.symprec)
            self.state["input"] = {"structure": os.path.abspath(a.structure),
                                   "spacegroup": std["spacegroup"],
                                   "lattice": std["prim_lattice"],
                                   "positions": std["prim_positions"],
                                   "numbers": std["prim_numbers"]}
            self.save()
        inp = self.state["input"]
        types = {p["type"] for p in pseudos.values()}
        log(f"{a.name}: space group {inp['spacegroup']}, {len(inp['numbers'])} atoms "
            f"in primitive cell; pseudos " +
            ", ".join(f"{k}={v['file']} ({v['type']}, {v['functional']})"
                      for k, v in pseudos.items()))
        if types != {"NC"} and a.scan_dual < 8:
            log("  WARNING: ultrasoft/PAW pseudos usually need ecutrho >= 8*ecutwfc; "
                "raise scan_dual")
        functionals = {p["functional"] for p in pseudos.values()} - {"?"}
        if len(functionals) > 1:
            log(f"  WARNING: pseudopotentials use different functionals: {sorted(functionals)}")
        self.ref_dual = a.scan_dual

    def _born_settings_changed(self, born_dir, born_path):
        """Move old Born outputs aside if born_volume changed since they were made."""
        want = {"born_volume": self.args.born_volume}
        have = self.state.get("born_settings")
        if have is None and os.path.exists(os.path.join(born_dir, "ph.out")):
            have = {"born_volume": "relaxed"}        # runs made before this option
        if have is not None and have != want:
            old = os.path.join(born_dir, f"previous_{have['born_volume']}")
            os.makedirs(old, exist_ok=True)
            for fn in ("scf.in", "scf.out", "ph.in", "ph.out", "BORN"):
                if os.path.exists(os.path.join(born_dir, fn)):
                    os.replace(os.path.join(born_dir, fn), os.path.join(old, fn))
            if os.path.exists(born_path):
                os.remove(born_path)
            self.state.get("jobs", {}).pop("born", None)
            log(f"  born_volume changed ({have['born_volume']} -> {want['born_volume']}); "
                f"old Born outputs moved to {os.path.relpath(old, self.work)}")
        self.state["born_settings"] = want
        self.save()

    def _npool(self, lattice, positions, numbers, mesh):
        return num_pools(lattice, positions, numbers, mesh, self.args.ntasks, self.args.max_pools)

    # ------------------------------------------------------------------
    def convergence(self, kind):
        """kind = 'ecutwfc' or 'ecutrho'. Returns the converged value."""
        a, key = self.args, f"conv_{kind}"
        if key in self.state.get("results", {}):
            return self.state["results"][key]
        inp = self.state["input"]
        # Rattled supercell: atoms move independently of their periodic images,
        # so the force check probes the same couplings the phonon runs need.
        lat, pos, num = repeat_cell(inp["lattice"], inp["positions"], inp["numbers"],
                                    a.conv_supercell)
        pos = rattle(lat, pos, a.conv_rattle)
        syms = symbols_of(num)
        mesh = kmesh(lat, a.kspacing)
        npool = self._npool(lat, pos, num, mesh)
        if kind == "ecutwfc":
            values = parse_list(a.ecutwfc_list)
            pairs = [(v, v * self.ref_dual) for v in values]
            directory = os.path.join(self.work, "01_ecutwfc")
        else:
            ecutwfc = self.state["results"]["conv_ecutwfc"]
            values = parse_list(a.dual_list)
            pairs = [(ecutwfc, ecutwfc * d) for d in values]
            directory = os.path.join(self.work, "02_ecutrho")
        log(f"Stage: {kind} convergence, values {values}, k-mesh {mesh}")
        os.makedirs(directory, exist_ok=True)
        runs = []
        for v, (ew, er) in zip(values, pairs):
            tag = f"{kind}_{v:g}"
            with open(os.path.join(directory, tag + ".in"), "w") as f:
                f.write(pw_input("scf", tag, lat, pos, syms, ew, er, mesh, self.cfg,
                                 scratch_dir(self.cfg, directory), extra_control="  disk_io     = 'none'\n"))
            runs.append(("pw.x", npool, tag + ".in", tag + ".out"))
        self.run_stage(key, directory, [(f"{a.name}_{kind}", runs, a.walltime)])

        data = [parse_pw(os.path.join(directory, r[3])) for r in runs]
        chosen, table = pick_converged(values, data, a.etol, a.ptol, a.ftol)
        write_conv_report(directory, kind, values, pairs, table, chosen, a)
        if chosen is None:
            raise RuntimeError(f"{kind} not converged within {values}; extend the list.")
        if chosen == values[-2]:
            log(f"  WARNING: {kind}={chosen} is the second-highest value tested; "
                "the range may be too small.")
        elif chosen == values[0]:
            log(f"  NOTE: {kind}={chosen} is the lowest value tested; a lower value "
                "might also pass (only matters for cost).")
        result = chosen if kind == "ecutwfc" else chosen * self.state["results"]["conv_ecutwfc"]
        log(f"  converged {kind} = {result:g} Ry")
        self.state.setdefault("results", {})[key] = result
        self.save()
        return result

    # ------------------------------------------------------------------
    def relax(self):
        a = self.args
        res = self.state.setdefault("results", {})
        if "relaxed" in res:
            return res["relaxed"]
        ew, er = res["conv_ecutwfc"], res["conv_ecutrho"]
        directory = os.path.join(self.work, "03_relax")
        inp = self.state["input"]
        lat, pos, syms = inp["lattice"], inp["positions"], symbols_of(inp["numbers"])
        for cycle in range(1, a.max_relax + 1):
            mesh = kmesh(lat, a.kspacing)
            npool = self._npool(lat, pos, inp["numbers"], mesh)
            tag = f"relax_{cycle}"
            log(f"Stage: vc-relax cycle {cycle} (ecutwfc={ew:g}, ecutrho={er:g}, k-mesh {mesh})")
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(os.path.join(directory, tag + ".in")):
                with open(os.path.join(directory, tag + ".in"), "w") as f:
                    f.write(pw_input(
                        "vc-relax", f"{a.name}_{tag}", lat, pos, syms, ew, er, mesh, self.cfg,
                        scratch_dir(self.cfg, directory),
                        extra_control="  etot_conv_thr = 1.0d-6\n  forc_conv_thr = 1.0d-4\n"
                                      "  disk_io     = 'none'\n",
                        extra_sections="&IONS\n  ion_dynamics = 'bfgs'\n/\n"
                                       "&CELL\n  cell_dynamics  = 'bfgs'\n  cell_dofree    = 'all'\n"
                                       "  press          = 0.0\n  press_conv_thr = 0.1\n/\n"))
            self.run_stage(f"relax_{cycle}", directory,
                           [(f"{a.name}_{tag}", [("pw.x", npool, tag + ".in", tag + ".out")],
                             a.walltime)])
            out = parse_pw(os.path.join(directory, tag + ".out"))
            if "final" not in out:
                raise RuntimeError(f"No final coordinates in {tag}.out (BFGS failed?)")
            lat, pos, syms = out["final"]["lattice"], out["final"]["positions"], out["final"]["symbols"]
            log(f"  final-scf pressure {out['pressure']:.2f} kbar, "
                f"volume {abs(np.linalg.det(lat)):.3f} A^3")
            if abs(out["pressure"]) <= a.relax_ptol:
                break
            log(f"  |P| > {a.relax_ptol} kbar (Pulay stress / basis change); relaxing again")
        else:
            log(f"  WARNING: pressure still {out['pressure']:.2f} kbar after {a.max_relax} cycles")
        from ase.data import atomic_numbers
        numbers = [atomic_numbers[s] for s in syms]
        std = standardize(lat, pos, numbers, a.symprec)
        if std["spacegroup"] != inp["spacegroup"]:
            log(f"  WARNING: symmetry changed during relaxation: "
                f"{inp['spacegroup']} -> {std['spacegroup']}")
        std["pressure"] = out["pressure"]
        res["relaxed"] = std
        self.save()
        c = np.array(std["conv_lattice"])
        log(f"  relaxed conventional cell: a,b,c = " +
            ", ".join(f"{np.linalg.norm(v):.4f}" for v in c) + " A")
        return std

    # ------------------------------------------------------------------
    def phonons(self):
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from phonopy.interface.qe import get_pwscf_structure
        a = self.args
        res = self.state["results"]
        std, ew, er = res["relaxed"], res["conv_ecutwfc"], res["conv_ecutrho"]
        directory = os.path.join(self.work, "04_phonopy")
        os.makedirs(directory, exist_ok=True)
        conv = np.array(std["conv_lattice"])
        n = [max(1, math.ceil(a.supercell_length / np.linalg.norm(v) - 1e-6)) for v in conv]
        yaml_path = os.path.join(directory, "phonopy_disp.yaml")
        pp = {k: v["file"] for k, v in self.cfg["pseudos"].items()}
        if not os.path.exists(yaml_path):
            unit = PhonopyAtoms(symbols=symbols_of(std["conv_numbers"]),
                                cell=conv / BOHR, scaled_positions=std["conv_positions"])
            ph = Phonopy(unit, supercell_matrix=np.diag(n),
                         primitive_matrix=std["prim_matrix"], calculator="qe")
            ph.generate_displacements(distance=a.displacement)
            ph.save(yaml_path)
        else:
            import phonopy
            ph = phonopy.load(yaml_path, produce_fc=False)
        sc = ph.supercell
        nat = len(sc)
        log(f"Stage: phonopy, supercell {n} of the conventional cell ({nat} atoms), "
            f"{len(ph.supercells_with_displacements)} displacement(s)")
        if nat > 300:
            log("  WARNING: large supercell; consider a smaller --supercell-length")

        sc_lat = sc.cell * BOHR
        mesh = kmesh(sc_lat, a.kspacing)
        npool = num_pools(sc_lat, sc.scaled_positions, sc.numbers, mesh, a.ntasks, a.max_pools)
        jobs = []
        for i, cell in enumerate(ph.supercells_with_displacements, 1):
            tag = f"disp-{i:03d}"
            path = os.path.join(directory, tag + ".in")
            if not output_done(os.path.join(directory, tag + ".out")):
                # disk_io='none': only the forces in the output are needed
                text = pw_input("scf", f"{a.name}_{tag}", cell.cell * BOHR, cell.scaled_positions,
                                list(cell.symbols), ew, er, mesh, self.cfg,
                                scratch_dir(self.cfg, directory), extra_control="  disk_io     = 'none'\n")
                with open(path, "w") as f:
                    f.write(text)
            jobs.append((f"{a.name}_{tag}", [("pw.x", npool, tag + ".in", tag + ".out")],
                         a.disp_walltime))

        born_dir = os.path.join(directory, "born")
        born_path = os.path.join(directory, "BORN")
        prim = ph.primitive
        compute_born = not a.metal and not a.born_file
        if a.metal and a.born_file:
            log("  NOTE: born_file is ignored for metals (no LO-TO correction)")
        if compute_born:
            os.makedirs(born_dir, exist_ok=True)
            p_lat = prim.cell * BOHR
            if a.born_volume == "input":
                # Same cell and atom order, isotropically rescaled to the input
                # structure's volume (e.g. the experimental lattice)
                v_in = abs(np.linalg.det(self.state["input"]["lattice"]))
                v_rel = abs(np.linalg.det(std["prim_lattice"]))
                p_lat = p_lat * (v_in / v_rel) ** (1 / 3)
            self._born_settings_changed(born_dir, born_path)
            p_mesh = kmesh(p_lat, a.kspacing)
            p_npool = num_pools(p_lat, prim.scaled_positions, prim.numbers, p_mesh,
                                a.ntasks, a.max_pools)
            # A few empty bands so pw.x reports the gap (checked below)
            zval = [self.cfg["pseudos"][el]["zval"] for el in prim.symbols]
            nbnd = (f"  nbnd        = {math.ceil(sum(zval) / 2) + 4}\n"
                    if None not in zval else "")
            # Always (re)written: phonopy-qe-born reads the structure from it
            with open(os.path.join(born_dir, "scf.in"), "w") as f:
                f.write(pw_input("scf", a.name, p_lat, prim.scaled_positions,
                                 list(prim.symbols), ew, er, p_mesh, self.cfg,
                                 scratch_dir(self.cfg, born_dir), conv_thr=1e-12,
                                 extra_system=nbnd))
        if compute_born and not output_done(os.path.join(born_dir, "ph.out")):
            with open(os.path.join(born_dir, "ph.in"), "w") as f:
                f.write(f"Born effective charges and dielectric tensor\n&INPUTPH\n"
                        f"  prefix = '{a.name}'\n  outdir = '{scratch_dir(self.cfg, born_dir)}'\n  tr2_ph = 1.0d-16\n"
                        f"  epsil  = .true.\n  trans  = .false.\n  fildyn = '{a.name}.dynG'\n/\n"
                        "0.0 0.0 0.0\n")
        # Submit the displacement jobs and the Born job together
        if compute_born:
            born_job = (f"{a.name}_born", [("pw.x", p_npool, "scf.in", "scf.out"),
                                           ("ph.x", p_npool, "ph.in", "ph.out")], a.walltime)
            # run_stage blocks, so start the Born job first if needed
            ids = self.state.setdefault("jobs", {}).setdefault("born", {})
            if not output_done(os.path.join(born_dir, "ph.out")) and not (
                    born_job[0] in ids and job_active(ids[born_job[0]])):
                jobid = submit(self.cfg, born_dir, born_job[0], born_job[1], born_job[2])
                if jobid:
                    ids[born_job[0]] = jobid
                    self.save()
        self.run_stage("disp", directory, jobs)
        if compute_born:
            self.run_stage("born", born_dir, [born_job])

        from phonopy.interface.qe import parse_set_of_forces
        from phonopy.file_IO import write_FORCE_SETS
        outs = [os.path.join(directory, f"disp-{i:03d}.out") for i in range(1, len(jobs) + 1)]
        ph.forces = parse_set_of_forces(nat, outs, verbose=False)
        write_FORCE_SETS(ph.dataset, filename=os.path.join(directory, "FORCE_SETS"))
        res.pop("born_check", None)
        if a.born_file and not a.metal:
            eps, zstar = parse_born(a.born_file)        # validates the format
            shutil.copy(a.born_file, born_path)
            res["born_check"] = {"source": os.path.abspath(a.born_file),
                                 "eps_inf_max": float(eps.max()),
                                 "zstar_max": float(np.abs(zstar).max())}
            log(f"  BORN copied from {a.born_file} (epsilon_inf {eps.max():.2f}, "
                f"max |Z*| {np.abs(zstar).max():.2f})")
        elif compute_born:
            bindir = os.path.dirname(sys.executable)
            r = subprocess.run([os.path.join(bindir, "phonopy-qe-born"), "scf.in", "ph.out"],
                               cwd=born_dir, check=True, capture_output=True, text=True)
            born_text = r.stdout
            with open(os.path.join(born_dir, "BORN"), "w") as f:
                f.write(born_text)
            eps, zstar = parse_born(os.path.join(born_dir, "BORN"))
            if np.abs(zstar).max() < 0.05:
                # all ~0 for non-polar crystals (Si): no LO-TO splitting
                if os.path.exists(born_path):
                    os.remove(born_path)
                log("  Born charges are ~0 (non-polar crystal): no LO-TO correction; "
                    "values kept in born/BORN")
            else:
                with open(born_path, "w") as f:
                    f.write(born_text)
                check = born_check(os.path.join(born_dir, "scf.out"), eps, zstar)
                check["source"] = f"computed ({a.born_volume} volume)"
                res["born_check"] = check
                gap = f"{check['gap_ev']:.2f} eV" if check["gap_ev"] is not None else "n/a"
                log(f"  BORN written (gap {gap}, epsilon_inf {check['eps_inf_max']:.2f}, "
                    f"max |Z*| {check['zstar_max']:.2f})")
                if check["warning"]:
                    log(f"  WARNING: {check['warning']}")
        self.save()
        log("  FORCE_SETS written")
        return directory

    # ------------------------------------------------------------------
    def plot(self, directory):
        import phonopy
        a = self.args
        std = self.state["results"]["relaxed"]
        born = os.path.join(directory, "BORN")
        has_nac = os.path.exists(born) and not a.metal
        ph = phonopy.load(os.path.join(directory, "phonopy_disp.yaml"),
                          force_sets_filename=os.path.join(directory, "FORCE_SETS"),
                          born_filename=born if has_nac else None, is_nac=has_nac)
        ph.save(os.path.join(directory, "phonopy_params.yaml"))
        ph_bare = None
        if has_nac:
            ph_bare = phonopy.load(os.path.join(directory, "phonopy_disp.yaml"),
                                   force_sets_filename=os.path.join(directory, "FORCE_SETS"),
                                   is_nac=False)
        bands, labels, connections = band_path(std, ph.primitive.cell, a.npoints)
        out_png = os.path.join(self.work, f"{a.name}_phonons.png")
        summary = plot_phonons(ph, ph_bare, bands, labels, connections, out_png, a)
        ph.write_yaml_band_structure(filename=os.path.join(directory, "band.yaml"))
        log(f"Plot written: {out_png}")
        return summary


# ----------------------------------------------------------------------------
# Convergence analysis
# ----------------------------------------------------------------------------
def parse_list(text):
    """'30:120:10' or '4,6,8' -> list of floats."""
    if ":" in text:
        lo, hi, step = (float(x) for x in text.split(":"))
        return [round(lo + i * step, 6) for i in range(int(round((hi - lo) / step)) + 1)]
    return [float(x) for x in text.split(",")]


def pick_converged(values, data, etol, ptol, ftol):
    """Smallest value whose results, and those of every larger value, agree with
    the largest value within etol (mRy/atom), ptol (kbar), ftol (Ry/bohr)."""
    ref = data[-1]
    table, ok = [], []
    for v, d in zip(values, data):
        dE = (d["energy"] - ref["energy"]) / d["nat"] * 1000
        dP = d["pressure"] - ref["pressure"]
        dF = (np.abs(np.array(d["forces"]) - np.array(ref["forces"])).max()
              if d.get("forces") and ref.get("forces") else 0.0)
        good = abs(dE) <= etol and abs(dP) <= ptol and dF <= ftol
        table.append((v, d["energy"], dE, d["pressure"], dP, dF, good))
        ok.append(good)
    chosen = None
    for i in range(len(values) - 2, -1, -1):     # the reference itself doesn't count
        if not ok[i]:
            break
        chosen = values[i]
    return chosen, table


def write_conv_report(directory, kind, values, pairs, table, chosen, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines = [f"# {kind} convergence (reference = last row); tolerances: "
             f"{a.etol} mRy/atom, {a.ptol} kbar, {a.ftol} Ry/bohr",
             f"{'ecutwfc':>8} {'ecutrho':>8} {'E (Ry)':>18} {'dE (mRy/at)':>12} "
             f"{'P (kbar)':>10} {'dP (kbar)':>10} {'dF':>10}  ok"]
    for (ew, er), (v, E, dE, P, dP, dF, good) in zip(pairs, table):
        lines.append(f"{ew:8g} {er:8g} {E:18.8f} {dE:12.4f} {P:10.2f} {dP:10.2f} "
                     f"{dF:10.2e}  {'yes' if good else 'no'}")
    lines.append(f"# chosen: {chosen}")
    text = "\n".join(lines)
    with open(os.path.join(directory, "convergence.txt"), "w") as f:
        f.write(text + "\n")
    print(text, flush=True)

    x = [ew if kind == "ecutwfc" else er for ew, er in pairs][:-1]
    dE = [abs(t[2]) for t in table][:-1]
    dP = [abs(t[4]) for t in table][:-1]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.0), constrained_layout=True)
    for ax, y, tol, lab in [(axs[0], dE, a.etol, "|ΔE| (mRy/atom)"),
                            (axs[1], dP, a.ptol, "|ΔP| (kbar)")]:
        ax.semilogy(x, np.maximum(y, 1e-6), "o-", color=SERIES[0], lw=2, ms=6)
        ax.axhline(tol, color="0.5", ls="--", lw=1)
        ax.text(x[0], tol * 1.15, "tolerance", color="0.35", fontsize=8)
        if chosen is not None:
            xc = chosen if kind == "ecutwfc" else chosen * pairs[0][0]
            ax.axvline(xc, color="0.7", lw=1)
        ax.set_xlabel(f"{kind} (Ry)")
        ax.set_ylabel(lab)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"{a.name}: {kind} convergence vs. {pairs[-1][0 if kind == 'ecutwfc' else 1]:g} Ry")
    fig.savefig(os.path.join(directory, "convergence.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Band path and plotting
# ----------------------------------------------------------------------------
def band_path(std, prim_cell_bohr, npoints):
    """Seekpath q-path converted into the basis of phonopy's primitive cell."""
    L_sp = np.array(std["prim_lattice"]) / BOHR
    L_ph = np.array(prim_cell_bohr)
    # q_frac_ph = q_frac_sp . inv(L_sp)^T . L_ph^T (same Cartesian frame)
    T = np.linalg.inv(L_sp).T @ L_ph.T
    coords = {k: np.array(v) @ T for k, v in std["point_coords"].items()}
    bands, labels, connections = [], [], []
    for i, (s, e) in enumerate(std["path"]):
        q0, q1 = coords[s], coords[e]
        bands.append(np.array([q0 + (q1 - q0) * t for t in np.linspace(0, 1, npoints)]))
        labels.append((s, e))
        nxt = std["path"][i + 1][0] if i + 1 < len(std["path"]) else None
        connections.append(nxt == e)
    return bands, labels, connections


def pretty(label):
    label = label.replace("GAMMA", r"\Gamma")
    if "_" in label:
        base, sub = label.split("_", 1)
        label = f"{base}_{{{sub}}}"
    return rf"$\mathrm{{{label}}}$"


def plot_phonons(ph, ph_bare, bands, labels, connections, out_png, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def run(p):
        p.run_band_structure(bands, path_connections=connections, labels=None)
        return p.band_structure.distances, [f * THZ_TO_CM for f in p.band_structure.frequencies]

    dist, freqs = run(ph)
    bare = run(ph_bare)[1] if ph_bare is not None else None

    ph.run_mesh(a.dos_mesh, with_eigenvectors=True, is_mesh_symmetry=False)
    ph.run_total_dos(sigma=a.dos_sigma)
    ph.run_projected_dos(sigma=a.dos_sigma)
    e_dos = ph.total_dos.frequency_points * THZ_TO_CM
    tdos = ph.total_dos.dos / THZ_TO_CM
    syms = list(ph.primitive.symbols)
    elements = list(dict.fromkeys(syms))
    pdos = {el: sum(ph.projected_dos.projected_dos[i] for i, s in enumerate(syms) if s == el)
            / THZ_TO_CM for el in elements}

    fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.5, 4.5), sharey=True,
                                  gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    for i, d in enumerate(dist):
        if bare is not None:
            lb = ax.plot(d, bare[i], color="0.6", lw=1.0, ls="--")
            if i == 0:
                lb[0].set_label("without NAC")
        ln = ax.plot(d, freqs[i], color="k", lw=1.2)
        if i == 0 and bare is not None:
            ln[0].set_label("with NAC (LO–TO)")
    # Tick at every segment start, merging labels where the path jumps
    ticks, ticklabels = [dist[0][0]], [pretty(labels[0][0])]
    for i, d in enumerate(dist):
        end = pretty(labels[i][1])
        if i + 1 < len(dist) and not connections[i]:
            end = f"{end}|{pretty(labels[i + 1][0])}"
        ticks.append(d[-1])
        ticklabels.append(end)
    for t in ticks[1:-1]:
        ax.axvline(t, color="gray", lw=0.6)
    ax.axhline(0, color="0.75", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels)
    ax.set_xlim(ticks[0], ticks[-1])
    allf = np.concatenate([f.ravel() for f in freqs + (bare or [])])
    ax.set_ylim(min(0, allf.min() * 1.05), allf.max() * 1.05)
    ax.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax.set_title(f"{a.name} phonon dispersion", pad=22 if bare is not None else 6)
    if bare is not None:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8,
                  frameon=False)

    axd.plot(tdos, e_dos, color="k", lw=1.2, label="total")
    if len(elements) > 1:          # one element: the projection equals the total
        for j, el in enumerate(elements):
            axd.plot(pdos[el], e_dos, color=SERIES[j % len(SERIES)], lw=1.2, label=el)
    axd.set_xlim(0, tdos.max() * 1.1)
    axd.set_xticks([])
    axd.set_xlabel("DOS")
    axd.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Gamma frequencies, approached along a segment leaving Gamma so the LO
    # mode carries its non-analytic shift
    q_g = np.zeros(3)
    if ph_bare is not None:
        i = next((i for i, l in enumerate(labels) if l[0] == "GAMMA"), None)
        if i is not None:
            d = bands[i][-1] - bands[i][0]
            q_g = d * 1e-4 / np.linalg.norm(d)
    ph.run_qpoints([q_g])
    return {"gamma_cm1": (ph.qpoints.frequencies[0] * THZ_TO_CM).round(2).tolist(),
            "min_freq_cm1": float(allf.min())}


# ----------------------------------------------------------------------------
# Command line and config files
# ----------------------------------------------------------------------------
COMMANDS = ("start", "run", "submit", "status", "init")
PATH_KEYS = ("structure", "pseudo_dir", "workdir", "scratch", "born_file")
REQUIRED = ("name", "structure", "pseudo_dir")
INTERNAL = {"help", "dry_run"}


def build_parser():
    p = argparse.ArgumentParser(prog="autophonon.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("required (here or in the config file)")
    g.add_argument("--name", help="compound name (used for prefixes/plots)")
    g.add_argument("--structure", help="CIF / POSCAR / pw.x input / ... (anything ASE reads)")
    g.add_argument("--pseudo-dir", help="directory with one UPF per element")
    g = p.add_argument_group("general")
    g.add_argument("--workdir", help="output directory (default: <name>_auto next to the "
                   "config file, or in the current directory)")
    g.add_argument("--dry-run", action="store_true",
                   help="write the next stage's inputs and job scripts, submit nothing")
    g = p.add_argument_group("convergence")
    g.add_argument("--ecutwfc-list", default="30:120:10", help="Ry, 'lo:hi:step' or 'a,b,c'")
    g.add_argument("--scan-dual", type=float, default=8,
                   help="ecutrho/ecutwfc used during the ecutwfc scan")
    g.add_argument("--dual-list", default="4,6,8,10,12", help="ecutrho/ecutwfc values to scan")
    g.add_argument("--etol", type=float, default=0.5, help="energy tolerance, mRy/atom")
    g.add_argument("--ptol", type=float, default=1.0, help="pressure tolerance, kbar")
    g.add_argument("--ftol", type=float, default=1e-4,
                   help="force tolerance, Ry/bohr (phonon forces are ~1e-4..1e-3)")
    g.add_argument("--conv-rattle", type=float, default=0.03,
                   help="random displacement (Angstrom) applied for the cutoff scans")
    g.add_argument("--conv-supercell", type=int, default=2,
                   help="n for the n x n x n primitive-cell supercell used in the cutoff scans")
    g.add_argument("--kspacing", type=float, default=0.25,
                   help="k-point spacing in 1/Angstrom (2*pi included)")
    g = p.add_argument_group("relaxation / phonons")
    g.add_argument("--max-relax", type=int, default=3, help="max vc-relax cycles")
    g.add_argument("--relax-ptol", type=float, default=0.5,
                   help="rerun vc-relax if final-scf |P| exceeds this (kbar)")
    g.add_argument("--symprec", type=float, default=1e-3, help="symmetry tolerance (Angstrom)")
    g.add_argument("--supercell-length", type=float, default=10.0,
                   help="min supercell edge (Angstrom); supercell = n x conventional cell")
    g.add_argument("--displacement", type=float, default=0.02, help="bohr")
    g.add_argument("--metal", action="store_true",
                   help="smearing occupations, no Born charges / LO-TO correction")
    g.add_argument("--degauss", type=float, default=0.02, help="Ry, with --metal")
    g.add_argument("--born-volume", choices=["relaxed", "input"], default="relaxed",
                   help="cell for epsilon/Z*: the relaxed one, or rescaled to the input "
                   "structure's volume (use when the DFT gap is too small at the relaxed volume)")
    g.add_argument("--born-file", help="phonopy BORN file to use instead of computing "
                   "epsilon/Z* (e.g. experimental values)")
    g = p.add_argument_group("plot")
    g.add_argument("--npoints", type=int, default=101, help="q-points per path segment")
    g.add_argument("--dos-mesh", type=float, default=200.0,
                   help="phonopy mesh length (bohr) for the DOS")
    g.add_argument("--dos-sigma", type=float, default=0.08, help="DOS smearing, THz")
    g = p.add_argument_group("slurm")
    g.add_argument("--account", default="strachan")
    g.add_argument("--partition", default="cpu")
    g.add_argument("--qos", default="normal",
                   help="QOS for the compute jobs; 'standby' uses idle nodes outside the "
                   "group allocation (4 h limit per job; walltimes are capped to it)")
    g.add_argument("--ntasks", type=int, default=32)
    g.add_argument("--max-pools", type=int, default=4)
    g.add_argument("--walltime", default="04:00:00", help="convergence/relax/Born jobs")
    g.add_argument("--disp-walltime", default="08:00:00", help="each displacement job")
    g.add_argument("--driver-qos", default="normal",
                   help="QOS for the 1-core driver job (keep a QOS without a short "
                   "walltime limit, even when qos: standby is used for the compute jobs)")
    g.add_argument("--driver-walltime", default="3-00:00:00",
                   help="driver job (it resubmits itself before this runs out)")
    g.add_argument("--modules", default="gcc/12.2.0 openmpi/4.1.4 quantum-espresso/7.1")
    g.add_argument("--scratch", help="QE outdir root (default: $RCAC_SCRATCH/autophonon/"
                   "<workdir name>, else <workdir>/tmp)")
    g.add_argument("--keep-tmp", action="store_true",
                   help="keep QE tmp/ (wavefunctions) after each stage")
    g.add_argument("--poll", type=int, default=60, help="seconds between queue checks")
    g.add_argument("--stop-after", choices=["ecutwfc", "ecutrho", "relax", "phonopy"],
                   help="stop after this stage")
    return p


def config_defaults(parser, path):
    """Read a YAML config into parser defaults. Paths are relative to the file."""
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        parser.error(f"{path}: expected 'key: value' lines")
    actions = {a.dest: a for a in parser._actions}
    base = os.path.dirname(os.path.abspath(path))
    out = {}
    for key, value in data.items():
        dest = str(key).replace("-", "_")
        if dest not in actions or dest in INTERNAL:
            valid = sorted(d for d in actions if d not in INTERNAL)
            parser.error(f"{path}: unknown option '{key}'. Valid options:\n  " + ", ".join(valid))
        act = actions[dest]
        if value is None:
            continue
        if isinstance(act, argparse._StoreTrueAction):
            value = bool(value)
        elif isinstance(value, list):
            value = ",".join(str(v) for v in value)
        elif act.type is not None:
            value = act.type(value)
        else:
            value = str(value)
        if act.choices and value not in act.choices:
            parser.error(f"{path}: {key} must be one of {list(act.choices)}")
        if dest in PATH_KEYS:
            value = os.path.join(base, os.path.expanduser(value))
        out[dest] = value
    if "workdir" not in out and out.get("name"):
        out["workdir"] = os.path.join(base, f"{out['name']}_auto")
    return out


def default_pseudo_dir():
    env = os.environ.get("AUTOPHONON_PSEUDO_DIR")
    if env:
        return os.path.expanduser(env)
    home = os.path.expanduser("~/pseudo/dojo-pbe")
    return home if os.path.isdir(home) else None


def parse_args(argv):
    """[config.yaml] [--flags] -> argparse namespace (flags override the file)."""
    parser = build_parser()
    config = None
    if argv and not argv[0].startswith("-"):
        config = os.path.abspath(argv.pop(0))
        if not os.path.isfile(config):
            parser.error(f"config file not found: {config}")
        parser.set_defaults(**config_defaults(parser, config))
    args = parser.parse_args(argv)
    if not args.pseudo_dir:
        args.pseudo_dir = default_pseudo_dir()
    missing = [k for k in REQUIRED if not getattr(args, k)]
    if missing:
        parser.error("missing " + ", ".join(m.replace("_", "-") for m in missing) +
                     " (give them in the config file or as --flags)")
    args.config = config
    return args


def resolved_config(args):
    """Every option with absolute paths, as written to <workdir>/autophonon.yaml."""
    out = {}
    for act in build_parser()._actions:
        if act.dest in INTERNAL:
            continue
        value = getattr(args, act.dest)
        if act.dest in PATH_KEYS and value:
            value = os.path.abspath(value)
        out[act.dest] = value
    out["workdir"] = os.path.abspath(args.workdir or f"{args.name}_auto")
    return out


def driver_script(args, work, config):
    script = os.path.abspath(__file__)
    lines = ["#!/bin/bash",
             f"#SBATCH --job-name={args.name}_driver",
             f"#SBATCH --account={args.account}",
             f"#SBATCH --partition={args.partition}",
             f"#SBATCH --qos={args.driver_qos}",
             "#SBATCH --nodes=1",
             "#SBATCH --ntasks=1",
             "#SBATCH --mem=1G",
             f"#SBATCH --time={args.driver_walltime}",
             # USR1 to the batch shell (exec'd into python) 10 min before the end
             "#SBATCH --signal=B:USR1@600",
             f"#SBATCH --output={os.path.join(work, 'driver.log')}",
             "#SBATCH --open-mode=append",
             "",
             f'cd "{work}"',
             "export PYTHONUNBUFFERED=1",
             f'exec "{sys.executable}" "{script}" run "{config}"']
    return "\n".join(lines) + "\n"


def cmd_submit(args):
    import yaml
    work = os.path.abspath(args.workdir or f"{args.name}_auto")
    os.makedirs(work, exist_ok=True)
    other = live_driver(work, args.poll)
    if other:
        sys.exit(f"{work} already has a {other}; see `autophonon status`")
    config = os.path.join(work, "autophonon.yaml")
    with open(config, "w") as f:
        f.write("# Resolved options for this run (written by `autophonon start/submit`)\n")
        yaml.safe_dump(resolved_config(args), f, sort_keys=False)
    path = os.path.join(work, "driver.slurm")
    with open(path, "w") as f:
        f.write(driver_script(args, work, config))
    if args.dry_run:
        print(f"Dry run: wrote {config} and {path}; not submitted.\n")
        print(open(path).read())
        return
    out = subprocess.run(["sbatch", "--parsable", path], cwd=work,
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"sbatch failed: {out.stderr.strip()}")
    jobid = out.stdout.strip().split(";")[0]
    write_driver(work, {"jobid": jobid, "submitted": time.time()})
    print(f"Submitted driver job {jobid} for {args.name}")
    print(f"  workdir: {work}")
    print(f"  log:     {os.path.join(work, 'driver.log')}")
    print(f"  check:   autophonon status   (in {os.path.dirname(work)})")


def resubmit_driver(work):
    path = os.path.join(work, "driver.slurm")
    out = subprocess.run(["sbatch", "--parsable", path], cwd=work,
                         capture_output=True, text=True)
    if out.returncode != 0:
        log(f"Could not resubmit the driver ({out.stderr.strip()}); "
            f"run `sbatch {path}` to continue")
        return
    jobid = out.stdout.strip().split(";")[0]
    write_driver(work, {"jobid": jobid, "submitted": time.time()})
    log(f"Driver walltime nearly used up; resubmitted as job {jobid}")


def cmd_run(args):
    wf = Workflow(args)
    if not args.dry_run:
        other = live_driver(wf.work, args.poll)
        if other:
            sys.exit(f"{wf.work} already has a {other}; see `autophonon status`")
        wf.heartbeat()
    in_slurm_driver = (os.environ.get("SLURM_JOB_ID")
                       and os.path.exists(os.path.join(wf.work, "driver.slurm")))
    if in_slurm_driver and not args.dry_run:
        def on_usr1(signum, frame):
            raise WalltimeSignal()
        signal.signal(signal.SIGUSR1, on_usr1)
    try:
        pipeline(wf, args)
    except DryRunStop as stop:
        log(f"Dry run finished: inputs and job scripts are in {stop.args[0]}")
    except WalltimeSignal:
        resubmit_driver(wf.work)
        return
    if not args.dry_run:
        info = read_driver(wf.work)
        info.update(heartbeat=0, finished=time.time())
        write_driver(wf.work, info)


def pipeline(wf, args):
    wf.setup()
    wf.convergence("ecutwfc")
    if args.stop_after == "ecutwfc":
        return
    wf.convergence("ecutrho")
    if args.stop_after == "ecutrho":
        return
    wf.relax()
    if args.stop_after == "relax":
        return
    directory = wf.phonons()
    if args.stop_after == "phonopy":
        return
    summary = wf.plot(directory)
    res = wf.state["results"]
    std = res["relaxed"]
    res["summary"] = summary
    wf.save()
    log("Done.")
    print(f"  ecutwfc = {res['conv_ecutwfc']:g} Ry, ecutrho = {res['conv_ecutrho']:g} Ry")
    print(f"  space group {std['spacegroup']}, conventional a,b,c = " +
          ", ".join(f"{np.linalg.norm(v):.4f}" for v in std["conv_lattice"]) + " A")
    print(f"  Gamma frequencies (cm^-1): {summary['gamma_cm1']}")
    bc = res.get("born_check")
    if bc:
        print(f"  Born charges: {bc['source']}, epsilon_inf {bc['eps_inf_max']:.2f}, "
              f"max |Z*| {bc['zstar_max']:.2f}")
        if bc.get("warning"):
            print(f"  WARNING: {bc['warning']}")
    if summary["min_freq_cm1"] < -5:
        print(f"  WARNING: imaginary modes down to {summary['min_freq_cm1']:.1f} cm^-1")


def find_run(directory="."):
    """The single run (workdir with state.json) in or below `directory`."""
    d = os.path.abspath(directory)
    marks = ("state.json", "driver.json")
    if any(os.path.exists(os.path.join(d, m)) for m in marks):
        return d
    found = sorted({os.path.dirname(f) for m in marks
                    for f in glob.glob(os.path.join(d, "*", m))})
    if len(found) == 1:
        return found[0]
    if not found:
        sys.exit("No run in this folder; give a config file or workdir, "
                 "e.g. `autophonon status GaAs.yaml`")
    sys.exit("Several runs here; pick one:\n  " + "\n  ".join(
        os.path.relpath(f) for f in found))


def cmd_status(argv):
    if not argv or argv[0].startswith("-"):
        argv = [find_run()]
    target = os.path.abspath(argv[0])
    work = target if os.path.isdir(target) else os.path.abspath(
        parse_args(list(argv)).workdir)
    state_file = os.path.join(work, "state.json")
    if not os.path.exists(state_file):
        info = read_driver(work)
        if info.get("jobid") and job_active(info["jobid"]):
            sys.exit(f"Workdir: {work}\nDriver:  Slurm job {info['jobid']} is queued; "
                     "nothing has run yet")
        sys.exit(f"No state.json in {work}; nothing has run there yet")
    with open(state_file) as f:
        state = json.load(f)
    res = state.get("results", {})
    print(f"Workdir: {work}")

    info = read_driver(work)
    if "summary" in res and not (info.get("jobid") and job_active(info["jobid"])):
        print("Driver:  finished")
    elif info.get("finished"):
        print(f"Driver:  finished {time.strftime('%Y-%m-%d %H:%M', time.localtime(info['finished']))}")
    elif info.get("jobid") and job_active(info["jobid"]):
        age = time.time() - info.get("heartbeat", 0)
        seen = f", last heartbeat {age / 60:.0f} min ago" if info.get("heartbeat") else ", queued"
        print(f"Driver:  Slurm job {info['jobid']}{seen}")
    elif info.get("pid") and time.time() - info.get("heartbeat", 0) < 600:
        print(f"Driver:  pid {info['pid']} on {info.get('host', '?')}")
    else:
        print("Driver:  not running (`autophonon start <name>.yaml` continues it)")

    inp = state.get("input", {})
    if inp:
        print(f"Input:   {inp['spacegroup']}, {len(inp['numbers'])} atoms in primitive cell")
    print(f"ecutwfc: {res['conv_ecutwfc']:g} Ry" if "conv_ecutwfc" in res else "ecutwfc: pending")
    print(f"ecutrho: {res['conv_ecutrho']:g} Ry" if "conv_ecutrho" in res else "ecutrho: pending")
    if "relaxed" in res:
        r = res["relaxed"]
        abc = ", ".join(f"{np.linalg.norm(v):.4f}" for v in r["conv_lattice"])
        print(f"relax:   {r['spacegroup']}, a,b,c = {abc} A, P = {r['pressure']:.2f} kbar")
    else:
        print("relax:   pending")
    pdir = os.path.join(work, "04_phonopy")
    disp = sorted(glob.glob(os.path.join(pdir, "disp-*.in")))
    if disp:
        done = sum(output_done(d[:-3] + ".out") for d in disp)
        born = os.path.join(pdir, "born", "ph.out")
        extra = (", Born charges " + ("done" if output_done(born) else "pending")
                 if os.path.exists(os.path.join(pdir, "born")) else "")
        print(f"phonons: {done}/{len(disp)} displacements done{extra}")
    else:
        print("phonons: pending")
    bc = res.get("born_check")
    if bc:
        gap = f", gap {bc['gap_ev']:.2f} eV" if bc.get("gap_ev") is not None else ""
        print(f"born:    {bc['source']}{gap}, epsilon_inf {bc['eps_inf_max']:.2f}, "
              f"max |Z*| {bc['zstar_max']:.2f}")
        if bc.get("warning"):
            print(f"         WARNING: {bc['warning']}")
    if "summary" in res:
        sm = res["summary"]
        print(f"result:  Gamma (cm^-1) {sm['gamma_cm1']}; lowest frequency "
              f"{sm['min_freq_cm1']:.1f} cm^-1")
        pngs = glob.glob(os.path.join(work, "*_phonons.png"))
        if pngs:
            print(f"plot:    {pngs[0]}")

    ids = sorted({j for stage in state.get("jobs", {}).values() for j in stage.values()})
    if ids:
        out = subprocess.run(["squeue", "-h", "-j", ",".join(ids), "-o", "%i  %j  %T  %M"],
                             capture_output=True, text=True)
        rows = [l for l in out.stdout.splitlines() if l.strip()]
        if rows:
            print("Jobs in queue:")
            for row in rows:
                print("  " + row)
    log_file = os.path.join(work, "driver.log")
    if os.path.exists(log_file):
        with open(log_file, errors="ignore") as f:
            tail = f.read().splitlines()[-5:]
        print("Recent driver.log:")
        for line in tail:
            print("  " + line)


def build_structure(spec):
    """'GaAs:zincblende:5.653' or 'GaN:wurtzite:3.19:5.19' -> CIF path."""
    from ase.build import bulk
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        sys.exit(f"'{spec}' is neither a file nor Formula:structure:a[:c] "
                 "(e.g. GaAs:zincblende:5.653, Al:fcc:4.05, GaN:wurtzite:3.19:5.19)")
    formula, kind = parts[0], parts[1]
    try:
        lengths = [float(x) for x in parts[2:]]
        atoms = bulk(formula, kind, a=lengths[0], c=lengths[1] if len(lengths) > 1 else None)
    except Exception as exc:
        sys.exit(f"Could not build {spec}: {exc}\nStructures ASE knows: sc, fcc, bcc, "
                 "hcp, diamond, zincblende, rocksalt, cesiumchloride, fluorite, wurtzite")
    path = os.path.abspath(f"{formula}.cif")
    if os.path.exists(path):
        sys.exit(f"{path} already exists; use `autophonon start {formula}.cif`")
    atoms.write(path)
    print(f"Built {kind} {formula} -> {os.path.basename(path)}")
    return path


def cmd_start(argv):
    """One step: structure (file or spec) or config -> config + submitted driver."""
    if not argv or argv[0].startswith("-"):
        sys.exit("usage: autophonon start <structure file | Formula:structure:a | config.yaml> "
                 "[--option value ...]")
    target, flags = argv[0], argv[1:]
    if target.endswith((".yaml", ".yml")):
        return cmd_submit(parse_args([target] + flags))
    if not os.path.exists(target):
        formula = target.split(":")[0]
        if ":" in target and os.path.exists(f"{formula}.yaml"):
            print(f"Using existing {formula}.yaml (edit it to change options)")
            return cmd_submit(parse_args([f"{formula}.yaml"] + flags))
        target = build_structure(target)
    name = os.path.splitext(os.path.basename(target))[0]
    if "--name" in flags:
        name = flags[flags.index("--name") + 1]
    config = os.path.abspath(f"{name}.yaml")
    if os.path.exists(config):
        print(f"Using existing {os.path.basename(config)} (edit it to change options)")
        return cmd_submit(parse_args([config] + flags))
    init_flags = [f for f in flags if f != "--dry-run"]
    if "--name" not in init_flags:
        init_flags += ["--name", name]
    cmd_init(init_flags + ["--structure", target, "-o", config])
    return cmd_submit(parse_args([config] + (["--dry-run"] if "--dry-run" in flags else [])))


def cmd_init(argv):
    import yaml
    parser = build_parser()
    parser.add_argument("-o", "--output", help="config file to write (default: <name>.yaml)")
    args = parser.parse_args(argv)
    if not args.pseudo_dir:
        args.pseudo_dir = default_pseudo_dir()
    missing = [k for k in REQUIRED if not getattr(args, k)]
    if missing:
        parser.error("init needs " + ", ".join("--" + m.replace("_", "-") for m in REQUIRED))
    out = os.path.abspath(args.output or f"{args.name}.yaml")
    if os.path.exists(out):
        sys.exit(f"{out} exists; remove it or pass -o")
    base = os.path.dirname(out)

    def rel(path):
        r = os.path.relpath(os.path.abspath(path), base)
        return os.path.abspath(path) if r.startswith("..") else r

    def show(v):
        return yaml.safe_dump(v, default_flow_style=True).strip().removesuffix("...").strip()

    lines = [f"# autophonon configuration for {args.name}",
             "# Paths are relative to this file. Uncomment a line to change its default.",
             f"# Start or resume:  autophonon start {os.path.basename(out)}",
             "",
             f"name: {show(args.name)}",
             f"structure: {show(rel(args.structure))}",
             f"pseudo_dir: {show(rel(args.pseudo_dir))}"]
    for group in parser._action_groups:
        acts = [a for a in group._group_actions
                if a.dest not in INTERNAL and a.dest not in REQUIRED and a.dest != "output"]
        if not acts:
            continue
        lines += ["", f"# --- {group.title} ---"]
        for a in acts:
            value = getattr(args, a.dest)
            key = f"{a.dest}: {show(value)}"
            comment = f"  # {a.help}" if a.help else ""
            lines.append(("" if value != a.default else "# ") + key + comment)
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {os.path.relpath(out)}")


def main():
    argv = sys.argv[1:]
    cmd = argv.pop(0) if argv and argv[0] in COMMANDS else "run"
    if not sys.argv[1:]:
        print("Usage" + __doc__.split("Usage", 1)[1])
        return
    if cmd == "start":
        return cmd_start(argv)
    if cmd == "status":
        return cmd_status(argv)
    if cmd == "init":
        return cmd_init(argv)
    args = parse_args(argv)
    if cmd == "submit":
        return cmd_submit(args)
    return cmd_run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log(f"FAILED: {exc}")
        sys.exit(1)
