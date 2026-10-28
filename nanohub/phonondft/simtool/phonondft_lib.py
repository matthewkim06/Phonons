"""Calculation code for the phonondft Sim2L.

Finite-displacement phonons (phonopy + Quantum ESPRESSO pw.x) for a structure
relaxed with the cellrelaxdft Sim2L, plus Born charges / epsilon_inf (ph.x)
for the LO-TO correction of insulators.

Adapted from autophonon.py (Negishi workflow). Differences: no cutoff scan
(the cutoff is an input with a guessed default), jobs go through nanoHUB
`submit` (or a local command), and the relaxed structure comes from a
cellrelaxdft squid.
"""
import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

BOHR = 0.529177210903          # Angstrom
THZ_TO_CM = 33.35641


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Reading the cellrelaxdft run
# ----------------------------------------------------------------------------
# Output names are matched loosely because they may change between
# cellrelaxdft revisions; check_cellrelax_outputs() prints what a run has.
STRUCTURE_HINTS = ("relaxed", "final", "structure", "poscar", "cif", "atoms", "cell")
CUTOFF_HINTS = {"ecutwfc": ("ecutwfc", "ecut_wfc", "wfc_cutoff", "cutoff_wfc", "ecut"),
                "ecutrho": ("ecutrho", "ecut_rho", "rho_cutoff", "cutoff_rho")}


def load_cellrelax(squid_id):
    """Outputs of a finished cellrelaxdft run, as {name: value}."""
    from simtool import searchForSimTool, CachedRun
    parts = squid_id.strip().split("/")
    if len(parts) != 3:
        raise ValueError(f"squid_id should look like cellrelaxdft/r101/<hash>, got {squid_id!r}")
    location = searchForSimTool(parts[0], parts[1])
    run = CachedRun(location, squidId=squid_id.strip())
    if not getattr(run, "cached", False) or run.db is None:
        raise RuntimeError(f"No cached result for {squid_id}; check the squid_id")
    names = list(run.db.nb.scraps.keys())
    values = {}
    for name in names:
        try:
            values[name] = run.read(name)
        except Exception as exc:          # one unreadable output shouldn't stop us
            values[name] = None
            log(f"  could not read output {name}: {exc}")
    return values


def check_cellrelax_outputs(squid_id):
    """Print every output of a cellrelaxdft run (to check the name matching)."""
    for name, value in load_cellrelax(squid_id).items():
        text = repr(value)
        print(f"{name:30s} {type(value).__name__:10s} {text[:100]}")


def _to_atoms(value):
    """ASE Atoms from whatever form a structure output takes, else None."""
    from ase import Atoms
    from ase.io import read
    if value is None:
        return None
    if isinstance(value, Atoms):
        return value
    if isinstance(value, dict):
        lat = value.get("lattice") or value.get("cell")
        pos = value.get("positions") or value.get("scaled_positions")
        sym = value.get("symbols") or value.get("species") or value.get("numbers")
        if lat is not None and pos is not None and sym is not None:
            scaled = "scaled_positions" in value or value.get("coords_are_cartesian") is False
            return (Atoms(sym, cell=lat, scaled_positions=pos, pbc=True) if scaled
                    else Atoms(sym, cell=lat, positions=pos, pbc=True))
        return None
    if not isinstance(value, str):
        return None
    if os.path.isfile(value):
        try:
            return read(value)
        except Exception:
            return None
    text = value
    # pw.x output with final coordinates, pw.x input, CIF, POSCAR, extxyz
    fmts = []
    if "Begin final coordinates" in text or "Program PWSCF" in text:
        fmts.append("espresso-out")
    if "ATOMIC_POSITIONS" in text and "CELL_PARAMETERS" in text:
        fmts.append("espresso-in")
    if "_cell_length_a" in text:
        fmts.append("cif")
    if "Lattice=" in text:
        fmts.append("extxyz")
    fmts.append("vasp")
    for fmt in fmts:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write(text)
        try:
            atoms = read(fh.name, format=fmt, index=-1)
            if len(atoms) and atoms.cell.volume > 1e-6:
                return atoms
        except Exception:
            pass
        finally:
            os.remove(fh.name)
    return None


def relaxed_structure(outputs):
    """(Atoms, output name) of the relaxed structure in the cellrelaxdft outputs."""
    ranked = sorted(outputs, key=lambda n: -sum(h in n.lower() for h in STRUCTURE_HINTS))
    for name in ranked:
        atoms = _to_atoms(outputs[name])
        if atoms is not None:
            return atoms, name
    raise RuntimeError("Could not find a relaxed structure in the cellrelaxdft outputs: "
                       + ", ".join(outputs) + ". Run check_cellrelax_outputs(squid_id) "
                       "and add the right name to STRUCTURE_HINTS.")


def relax_setting(outputs, kind):
    """A number such as ecutwfc used by the cellrelaxdft run, or None."""
    for name, value in outputs.items():
        low = name.lower()
        if any(h == low or low.endswith(h) for h in CUTOFF_HINTS[kind]):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    # also look inside dictionaries of DFT parameters
    for value in outputs.values():
        if isinstance(value, dict):
            for key, v in value.items():
                if str(key).lower() in CUTOFF_HINTS[kind]:
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
    return None


def pseudo_dir_from(outputs):
    """A directory with UPF files referenced by the cellrelaxdft outputs, or None."""
    for value in outputs.values():
        cands = value if isinstance(value, (list, tuple)) else [value]
        for v in cands:
            if not isinstance(v, str) or len(v) > 4096:
                continue
            if os.path.isdir(v) and glob.glob(os.path.join(v, "*.[uU][pP][fF]")):
                return v
            if v.lower().endswith(".upf") and os.path.isfile(v):
                return os.path.dirname(os.path.abspath(v))
    return None


# ----------------------------------------------------------------------------
# Pseudopotentials and cutoffs
# ----------------------------------------------------------------------------
def read_upf_header(path):
    with open(path, errors="ignore") as f:
        head = f.read(20000)
    info = {"file": os.path.basename(path)}
    m = re.search(r'element\s*=\s*"\s*([A-Za-z]{1,2})\s*"', head)
    if m:
        info["element"] = m.group(1).capitalize()
        t = re.search(r'pseudo_type\s*=\s*"\s*(\w+)\s*"', head)
        info["type"] = t.group(1).upper() if t else "NC"
        z = re.search(r'z_valence\s*=\s*"\s*([-+.\dEe]+)\s*"', head)
        info["zval"] = float(z.group(1)) if z else None
    else:
        m = re.search(r"<PP_HEADER>\s*\n\s*\S+.*\n\s*([A-Za-z]{1,2})\s+Element", head)
        if not m:
            raise ValueError(f"Cannot find the element in {path}")
        info["element"] = m.group(1).capitalize()
        info["type"] = ("PAW" if "PAW" in head[:3000] else
                        "US" if re.search(r"\bUS\b", head[:3000]) else "NC")
        z = re.search(r"([-+.\dEe]+)\s+Z valence", head)
        info["zval"] = float(z.group(1)) if z else None
    if info["type"] in ("SL", "1/R"):
        info["type"] = "NC"
    return info


def find_pseudos(pseudo_dir, elements):
    """{element: header info} for the elements, from a directory of UPF files."""
    found = {}
    for path in sorted(glob.glob(os.path.join(pseudo_dir, "*.[uU][pP][fF]"))):
        try:
            info = read_upf_header(path)
        except ValueError:
            continue
        if info["element"] in elements and info["element"] not in found:
            found[info["element"]] = info
    missing = sorted(set(elements) - set(found))
    if missing:
        raise RuntimeError(f"No UPF file for {missing} in {pseudo_dir}")
    return found


# Rule-of-thumb cutoffs (Ry) used when neither the user nor the relaxation
# gives one. They are a starting point, not a convergence test.
# The dual is 8 even for norm-conserving files: 4 is formally enough, but
# semicore states (e.g. Na 2s2p) gave wrong force constants at 4 for NaCl.
GUESS_ECUTWFC = {"NC": 80.0, "US": 50.0, "PAW": 50.0}
GUESS_DUAL = {"NC": 8.0, "US": 8.0, "PAW": 8.0}


def choose_cutoffs(ecutwfc, dual, pseudos, outputs):
    """(ecutwfc, ecutrho, note) from the inputs, the relaxation or a guess."""
    types = {p["type"] for p in pseudos.values()}
    kind = "NC" if types == {"NC"} else ("PAW" if "PAW" in types else "US")
    note = []
    if ecutwfc and ecutwfc > 0:
        note.append(f"ecutwfc {ecutwfc:g} Ry (given)")
    else:
        relax = relax_setting(outputs, "ecutwfc")
        if relax:
            ecutwfc = relax
            note.append(f"ecutwfc {ecutwfc:g} Ry (same as the relaxation)")
        else:
            ecutwfc = GUESS_ECUTWFC[kind]
            note.append(f"ecutwfc {ecutwfc:g} Ry (guess for {kind} pseudopotentials; edit "
                        "the ecutwfc input to change it)")
    if dual and dual > 0:
        ecutrho = ecutwfc * dual
        note.append(f"ecutrho {ecutrho:g} Ry (dual {dual:g}, given)")
    else:
        relax = relax_setting(outputs, "ecutrho")
        ecutrho = relax if relax and relax >= 4 * ecutwfc else ecutwfc * GUESS_DUAL[kind]
        note.append(f"ecutrho {ecutrho:g} Ry")
    return ecutwfc, ecutrho, "; ".join(note)


# ----------------------------------------------------------------------------
# Structure handling and QE inputs
# ----------------------------------------------------------------------------
def standardize(atoms, symprec):
    """Seekpath conventional/primitive cells and q-path for the band plot."""
    import seekpath
    sp = seekpath.get_path((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers),
                           with_time_reversal=True, symprec=symprec)
    return {
        "spacegroup": f"{sp['spacegroup_international']} ({sp['spacegroup_number']})",
        "conv_lattice": np.array(sp["conv_lattice"]),
        "conv_positions": np.array(sp["conv_positions"]),
        "conv_numbers": [int(n) for n in sp["conv_types"]],
        "prim_lattice": np.array(sp["primitive_lattice"]),
        "prim_matrix": np.array(sp["primitive_transformation_matrix"]).T,
        "point_coords": {k: np.array(v) for k, v in sp["point_coords"].items()},
        "path": [list(p) for p in sp["path"]],
    }


def formula(atoms):
    """Reduced formula in order of first appearance, e.g. NaCl, GaN, Zr."""
    from math import gcd
    from functools import reduce
    syms = atoms.get_chemical_symbols()
    order = list(dict.fromkeys(syms))
    counts = [syms.count(e) for e in order]
    g = reduce(gcd, counts)
    return "".join(e + (str(c // g) if c // g > 1 else "") for e, c in zip(order, counts))


def symbols_of(numbers):
    from ase.data import chemical_symbols
    return [chemical_symbols[n] for n in numbers]


def kmesh(lattice, kspacing):
    """Monkhorst-Pack mesh with |b_i| / n_i <= kspacing (|b| includes 2*pi)."""
    rec = 2 * np.pi * np.linalg.inv(np.array(lattice)).T
    return [max(1, math.ceil(np.linalg.norm(b) / kspacing - 1e-6)) for b in rec]


def pw_input(prefix, lattice, positions, symbols, ecutwfc, ecutrho, mesh, pseudos,
             pseudo_dir, metal, degauss, conv_thr=1e-10, nbnd=None):
    from ase.data import atomic_masses, atomic_numbers
    species = list(dict.fromkeys(symbols))
    occ = (f"  occupations = 'smearing'\n  smearing    = 'mv'\n  degauss     = {degauss}\n"
           if metal else "  occupations = 'fixed'\n")
    if nbnd:
        occ += f"  nbnd        = {nbnd}\n"
    s = (f"&CONTROL\n  calculation = 'scf'\n  prefix      = '{prefix}'\n"
         f"  outdir      = './tmp_{prefix}'\n  pseudo_dir  = '{pseudo_dir}'\n"
         "  tprnfor     = .true.\n/\n"
         f"&SYSTEM\n  ibrav       = 0\n  nat         = {len(symbols)}\n"
         f"  ntyp        = {len(species)}\n  ecutwfc     = {ecutwfc}\n"
         f"  ecutrho     = {ecutrho}\n{occ}/\n"
         f"&ELECTRONS\n  conv_thr    = {conv_thr:.1e}\n  mixing_beta = 0.7\n/\n")
    s += "CELL_PARAMETERS angstrom\n"
    for v in lattice:
        s += "  %16.10f %16.10f %16.10f\n" % tuple(v)
    s += "ATOMIC_SPECIES\n"
    for sym in species:
        s += f"  {sym:3s} {atomic_masses[atomic_numbers[sym]]:10.5f}  {pseudos[sym]['file']}\n"
    s += "ATOMIC_POSITIONS crystal\n"
    for sym, p in zip(symbols, positions):
        s += f"  {sym:3s} %16.10f %16.10f %16.10f\n" % tuple(p)
    s += "K_POINTS automatic\n  %d %d %d 0 0 0\n" % tuple(mesh)
    return s


def output_done(path):
    if not os.path.exists(path):
        return False
    with open(path, errors="ignore") as f:
        text = f.read()
    return "JOB DONE." in text and "convergence NOT achieved" not in text


# ----------------------------------------------------------------------------
# Running QE
# ----------------------------------------------------------------------------
def run_commands(jobs, command, ncpus, walltime, max_parallel, poll=10):
    """Run QE jobs, at most max_parallel at a time.

    jobs: list of lists of (executable, input, output) run in order in one
    directory (a Born job is pw.x then ph.x). command is a template with
    {ncpus}, {walltime}, {exe}, {input}; its stdout goes to the output file.
    """
    pending = [j for j in jobs if not all(output_done(o) for _, _, o in j)]
    running = []                         # (process, job, step index)

    def start(job, step):
        exe, inp, out = job[step]
        cmd = command.format(ncpus=ncpus, walltime=walltime, exe=exe, input=inp)
        fh = open(out, "w")
        proc = subprocess.Popen(shlex.split(cmd), stdout=fh, stderr=subprocess.STDOUT,
                                cwd=os.path.dirname(os.path.abspath(out)) or ".")
        proc._out = fh
        return proc

    log(f"Running {len(pending)} job(s), up to {max_parallel} at a time")
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            running.append((start(job, 0), job, 0))
        time.sleep(poll)
        still = []
        for proc, job, step in running:
            if proc.poll() is None:
                still.append((proc, job, step))
                continue
            proc._out.close()
            if step + 1 < len(job) and output_done(job[step][2]):
                still.append((start(job, step + 1), job, step + 1))
            else:
                log(f"  finished {os.path.basename(job[-1][2])}")
        running = still
    bad = [o for j in jobs for _, _, o in j if not output_done(o)]
    if bad:
        raise RuntimeError("These QE runs did not finish cleanly:\n  " + "\n  ".join(bad))


# ----------------------------------------------------------------------------
# Phonons
# ----------------------------------------------------------------------------
def setup_phonopy(std, supercell_length, displacement_bohr):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    conv = std["conv_lattice"]
    n = [max(1, math.ceil(supercell_length / np.linalg.norm(v) - 1e-6)) for v in conv]
    unit = PhonopyAtoms(symbols=symbols_of(std["conv_numbers"]), cell=conv / BOHR,
                        scaled_positions=std["conv_positions"])
    ph = Phonopy(unit, supercell_matrix=np.diag(n), primitive_matrix=std["prim_matrix"],
                 calculator="qe")
    ph.generate_displacements(distance=displacement_bohr)
    return ph, n


def parse_born(text):
    rows = []
    for line in text.splitlines():
        parts = line.split("#")[0].split()
        if len(parts) == 9:
            rows.append([float(x) for x in parts])
    if len(rows) < 2:
        raise ValueError("BORN text has no epsilon/Z* lines")
    return np.array(rows[0]).reshape(3, 3), np.array(rows[1:]).reshape(-1, 3, 3)


def band_gap(scf_out):
    with open(scf_out, errors="ignore") as f:
        m = re.findall(r"highest occupied, lowest unoccupied level \(ev\):\s+(\S+)\s+(\S+)",
                       f.read())
    return float(m[-1][1]) - float(m[-1][0]) if m else None


def band_path(std, prim_cell_bohr, npoints):
    L_sp = std["prim_lattice"] / BOHR
    T = np.linalg.inv(L_sp).T @ np.array(prim_cell_bohr).T
    coords = {k: v @ T for k, v in std["point_coords"].items()}
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


def dispersion(ph, ph_bare, std, npoints, title, out_png, dos, dos_mesh=200.0, dos_sigma=0.08):
    """Plot bands (+ DOS); return distances, frequencies (cm^-1), tick data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bands, labels, connections = band_path(std, ph.primitive.cell, npoints)

    def run(p):
        p.run_band_structure(bands, path_connections=connections, labels=None)
        return p.band_structure.distances, [f * THZ_TO_CM for f in p.band_structure.frequencies]

    dist, freqs = run(ph)
    bare = run(ph_bare)[1] if ph_bare is not None else None
    if dos:
        ph.run_mesh(dos_mesh)
        ph.run_total_dos(sigma=dos_sigma)
        fig, (ax, axd) = plt.subplots(1, 2, figsize=(7.5, 4.5), sharey=True,
                                      gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05})
    else:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for i, d in enumerate(dist):
        if bare is not None:
            lb = ax.plot(d, bare[i], color="0.6", lw=1.0, ls="--")
            if i == 0:
                lb[0].set_label("without NAC")
        ln = ax.plot(d, freqs[i], color="k", lw=1.2)
        if i == 0 and bare is not None:
            ln[0].set_label("with NAC (LO–TO)")
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
    ax.set_title(f"{title} phonon dispersion", pad=22 if bare is not None else 6)
    if bare is not None:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8,
                  frameon=False)
    if dos:
        e = ph.total_dos.frequency_points * THZ_TO_CM
        g = ph.total_dos.dos / THZ_TO_CM
        axd.plot(g, e, color="k", lw=1.2)
        axd.set_xlim(0, g.max() * 1.1)
        axd.set_xticks([])
        axd.set_xlabel("DOS")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Gamma frequencies approached along a path leaving Gamma (LO carries its shift)
    q = np.zeros(3)
    if ph_bare is not None:
        i = next((i for i, l in enumerate(labels) if l[0] == "GAMMA"), None)
        if i is not None:
            d = bands[i][-1] - bands[i][0]
            q = d * 1e-4 / np.linalg.norm(d)
    ph.run_qpoints([q])
    return {"distances": [d.tolist() for d in dist],
            "frequencies_cm1": [f.tolist() for f in freqs],
            "labels": [list(l) for l in labels],
            "gamma_cm1": (ph.qpoints.frequencies[0] * THZ_TO_CM).round(2).tolist(),
            "min_cm1": float(allf.min())}


def run_phonons(atoms, pseudos, pseudo_dir, ecutwfc, ecutrho, *, name="material",
                metal=False, degauss=0.02, kspacing=0.25, supercell_length=10.0,
                displacement=0.02, symprec=1e-3, compute_born=True, npoints=101,
                dos=True, command="submit -n {ncpus} -w {walltime} {exe} -in {input}",
                pw_exe="pw.x", ph_exe="ph.x", ncpus=16, walltime=120, max_parallel=8,
                workdir="phonon_run"):
    """The whole calculation; returns a dict of results (see the Sim2L OUTPUTS)."""
    import phonopy
    from phonopy.interface.qe import parse_set_of_forces
    os.makedirs(workdir, exist_ok=True)
    pseudo_dir = os.path.abspath(pseudo_dir)
    std = standardize(atoms, symprec)
    ph, n = setup_phonopy(std, supercell_length, displacement)
    sc = ph.supercell
    log(f"{name}: {std['spacegroup']}, supercell {n} of the conventional cell "
        f"({len(sc)} atoms), {len(ph.supercells_with_displacements)} displacement(s)")
    common = dict(ecutwfc=ecutwfc, ecutrho=ecutrho, pseudos=pseudos, pseudo_dir=pseudo_dir,
                  metal=metal, degauss=degauss)

    jobs = []
    mesh = kmesh(sc.cell * BOHR, kspacing)
    for i, cell in enumerate(ph.supercells_with_displacements, 1):
        tag = f"disp-{i:03d}"
        with open(os.path.join(workdir, tag + ".in"), "w") as f:
            f.write(pw_input(tag, cell.cell * BOHR, cell.scaled_positions, list(cell.symbols),
                             mesh=mesh, **common))
        jobs.append([(pw_exe, tag + ".in", os.path.join(workdir, tag + ".out"))])

    do_born = compute_born and not metal
    if do_born:
        prim = ph.primitive
        p_lat = prim.cell * BOHR
        zval = [pseudos[el]["zval"] for el in prim.symbols]
        nbnd = math.ceil(sum(zval) / 2) + 4 if None not in zval else None
        with open(os.path.join(workdir, "born_scf.in"), "w") as f:
            f.write(pw_input("born", p_lat, prim.scaled_positions, list(prim.symbols),
                             mesh=kmesh(p_lat, kspacing), conv_thr=1e-12, nbnd=nbnd, **common))
        with open(os.path.join(workdir, "born_ph.in"), "w") as f:
            f.write("Born effective charges and dielectric tensor\n&INPUTPH\n"
                    "  prefix = 'born'\n  outdir = './tmp_born'\n  tr2_ph = 1.0d-16\n"
                    "  epsil  = .true.\n  trans  = .false.\n  fildyn = 'born.dynG'\n/\n"
                    "0.0 0.0 0.0\n")
        jobs.append([(pw_exe, "born_scf.in", os.path.join(workdir, "born_scf.out")),
                     (ph_exe, "born_ph.in", os.path.join(workdir, "born_ph.out"))])

    run_commands(jobs, command, ncpus, walltime, max_parallel)

    outs = [os.path.join(workdir, f"disp-{i:03d}.out")
            for i in range(1, len(ph.supercells_with_displacements) + 1)]
    ph.forces = parse_set_of_forces(len(sc), outs, verbose=False)
    raw = os.path.join(workdir, "phonopy_forces.yaml")
    ph.save(raw)                         # structure, displacements and forces
    results = {"spacegroup": std["spacegroup"], "supercell": n, "natoms_supercell": len(sc),
               "n_displacements": len(outs), "eps_inf": None, "born_charges": None,
               "band_gap_ev": None, "warnings": []}

    born_file = None
    if do_born:
        script = os.path.join(os.path.dirname(sys.executable), "phonopy-qe-born")
        if not os.path.exists(script):
            script = shutil.which("phonopy-qe-born") or script
        born_text = subprocess.run([script, "born_scf.in", "born_ph.out"], cwd=workdir,
                                   check=True, capture_output=True, text=True).stdout
        eps, zstar = parse_born(born_text)
        gap = band_gap(os.path.join(workdir, "born_scf.out"))
        results.update(eps_inf=eps.tolist(), born_charges=zstar.tolist(), band_gap_ev=gap)
        if np.abs(zstar).max() < 0.05:
            log("  Born charges ~0 (non-polar): no LO-TO correction")
        else:
            born_file = os.path.join(workdir, "BORN")
            with open(born_file, "w") as f:
                f.write(born_text)
            if (gap is not None and gap < 1.0) or np.diag(eps).max() > 50:
                gap_txt = f"{gap:.2f} eV" if gap is not None else "unknown"
                results["warnings"].append(
                    f"Small DFT band gap ({gap_txt}), epsilon_inf {np.diag(eps).max():.1f}: "
                    "the LO-TO splitting is likely too small (DFT gap problem).")

    # With and without the non-analytic (LO-TO) correction
    ph = phonopy.load(raw, born_filename=born_file, is_nac=born_file is not None)
    ph_bare = phonopy.load(raw, is_nac=False) if born_file else None

    png = os.path.join(workdir, f"{name}_phonons.png")
    disp = dispersion(ph, ph_bare, std, npoints, name, png, dos)
    if disp["min_cm1"] < -5:
        results["warnings"].append(
            f"Imaginary modes down to {disp['min_cm1']:.1f} cm^-1 (dynamical instability, "
            "or unconverged cutoff / k-points / supercell)")
    params = os.path.join(workdir, "phonopy_params.yaml")
    ph.save(params)
    results.update(gamma_cm1=disp["gamma_cm1"], min_cm1=disp["min_cm1"],
                   band_structure={k: disp[k] for k in ("distances", "frequencies_cm1", "labels")},
                   plot=png, phonopy_params=params)
    for w in results["warnings"]:
        log(f"  WARNING: {w}")
    return results
