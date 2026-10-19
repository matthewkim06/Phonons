# autophonon

Phonon dispersions from a crystal structure with one command, using Quantum
ESPRESSO (pw.x, ph.x) and phonopy on Negishi. Give it a structure file and
pseudopotentials. It finds converged cutoffs, relaxes the cell, runs the
finite-displacement supercells (plus Born charges for insulators) through
Slurm, and plots the dispersion and projected DOS.

```
01_ecutwfc   wavefunction cutoff scan        (rattled 2x2x2 supercell)
02_ecutrho   charge-density cutoff scan      (same structure)
03_relax     vc-relax, repeated until |P| < 0.5 kbar
04_phonopy   displaced supercells + born/ (ph.x: Z*, epsilon)
<name>_phonons.png   dispersion (with/without LO-TO) + projected DOS
```

Every stage is resumable. If anything stops, run the same command again and
finished calculations are kept.

## Setup (once)

```bash
git clone git@github.com:matthewkim06/Phonons.git ~/Phonons
module load anaconda
conda env create -f ~/Phonons/environment.yml      # the "autophonon" env
mkdir -p ~/bin && ln -s ~/Phonons/autophonon ~/bin/  # puts `autophonon` on your PATH

# Pseudopotentials: PseudoDojo NC, PBE, scalar-relativistic, standard, v0.4
mkdir -p ~/pseudo/dojo-pbe && cd ~/pseudo/dojo-pbe
curl -LO http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz
tar xzf nc-sr-04_pbe_standard_upf.tgz && rm nc-sr-04_pbe_standard_upf.tgz
```

After this you never need `module load` or `conda activate`: the `autophonon`
command finds the environment itself. `~/pseudo/dojo-pbe` is the default
pseudopotential directory; to use another set, `export
AUTOPHONON_PSEUDO_DIR=/path/to/upfs` (e.g. in `~/.bashrc`) or pass
`--pseudo-dir`. Only the elements in your structure are used. Ultrasoft/PAW
files work too, but add `--scan-dual 12` since they need a larger
ecutrho/ecutwfc ratio.

Quantum ESPRESSO comes from the cluster module (`gcc/12.2.0 openmpi/4.1.4
quantum-espresso/7.1` by default; `--modules` to change).

## Running a calculation

```bash
mkdir -p ~/calcs/GaAs && cd ~/calcs/GaAs
autophonon start GaAs:zincblende:5.653     # or: autophonon start GaAs.cif
autophonon status                          # any time, in this folder
```

`start` builds the structure (or reads your CIF/POSCAR/any ASE format),
writes `GaAs.yaml` with every option and its default, and submits a 1-core
driver job that runs all stages through Slurm. You can log out. Results go to
`GaAs_auto/`; the log is `GaAs_auto/driver.log`.

* Built structures: `Formula:structure:a[:c]` with sc, fcc, bcc, hcp, diamond,
  zincblende, rocksalt, cesiumchloride, fluorite or wurtzite, e.g.
  `Al:fcc:4.05`, `MgO:rocksalt:4.21`, `GaN:wurtzite:3.19:5.19`. Use the
  experimental lattice constant; it is relaxed anyway.
* Options as flags on the first `start` (they're saved in the YAML), e.g.
  metals: `autophonon start Al:fcc:4.05 --metal` (Marzari–Vanderbilt
  smearing, no Born charges). `autophonon run -h` lists all options.
* To change options later, edit `GaAs.yaml` and run `autophonon start GaAs.yaml`.
  The same command resumes after a failure; finished work is kept.
* `dos: false` in the YAML (or `--dos false`) draws the dispersion without the
  DOS panel; rerunning `autophonon run <name>.yaml` on a finished run just
  redraws the plot.
* `--dry-run` on any `start` writes the inputs and job scripts without
  submitting.
* The structure is symmetrized and standardized first, so conventional or
  primitive cells are both fine.

**Small-gap semiconductors (GaAs, InAs, InSb, Ge, ...):** PBE nearly closes
their band gap at the relaxed lattice constant, so the computed dielectric
constant blows up and the LO–TO splitting comes out wrong (the dispersion
itself is fine). `status` warns when the gap is below 0.5 eV or epsilon_inf is
above 50. Fix it by adding one line to the YAML and running `start` again
(only the small Born step is redone):

```yaml
born_volume: input          # compute epsilon/Z* at the input (experimental) volume
born_file: GaAs_exp.BORN    # or use your own epsilon_inf and Z* (phonopy BORN format)
```

A BORN file has the 9 components of epsilon_inf on one line, then one line of
9 Z* components per symmetry-independent atom (`#` starts a comment):

```
# GaAs, experimental
10.9 0 0 0 10.9 0 0 0 10.9
2.07 0 0 0 2.07 0 0 0 2.07
-2.07 0 0 0 -2.07 0 0 0 -2.07
```

## Reading the results

* `NaCl_auto/NaCl_phonons.png`: dispersion along the standard
  (seekpath) path plus element-projected DOS. For insulators the dashed grey
  curves are without the non-analytic correction, so the LO–TO splitting at
  Γ is the difference between grey and black.
* `01_ecutwfc/convergence.txt`, `02_ecutrho/convergence.txt`: energy,
  pressure and force differences against the highest cutoff. The chosen value
  is the lowest one where it and every higher value pass all three tolerances.
* `04_phonopy/phonopy_params.yaml`: everything phonopy needs (structure,
  forces, Born charges). Load it for thermal properties, other paths, etc.:
  ```python
  import phonopy
  ph = phonopy.load("NaCl_auto/04_phonopy/phonopy_params.yaml")
  ph.run_mesh([20, 20, 20]); ph.run_thermal_properties(t_max=1000)
  ```
* `autophonon status` prints the converged cutoffs, relaxed lattice, Γ frequencies and
  lowest frequency.

### Sanity checks before using the numbers

* **Imaginary modes** (negative frequencies) below about −5 cm⁻¹ mean either a
  real instability or an unconverged calculation; the summary warns about
  them.
* **Wiggly bands** (oscillations, branches crossing at odd places) usually
  mean noisy forces from too-low cutoffs, or a supercell that is too small.
* **Supercell size** is set by `supercell_length` (default: every edge at
  least 10 Å). For publishable numbers, rerun with a larger value in a new
  `workdir` and check the frequencies change by only a few cm⁻¹.
* The cutoff scan compares forces in a randomly perturbed ("rattled") 2x2x2
  supercell with a 1e-4 Ry/bohr tolerance. This is deliberate: in NaCl,
  ecutrho = 320 Ry converged the energy and pressure but gave Na–Na force
  constants with the wrong sign. Don't loosen `ftol` without checking the bands.

## When something goes wrong

| Symptom | What to do |
|---|---|
| `status` says the driver is not running, stages pending | Look at the end of `driver.log`, fix the cause, then `autophonon start <name>.yaml`; it resumes. |
| `Stage ... failed; incomplete outputs` | Read the `.out` and `.err` files in that stage directory (walltime, memory, SCF not converging). Raise `walltime` / `disp_walltime` in the YAML if needed, then `autophonon start <name>.yaml`. |
| `not converged within [...]; extend the list` | Raise the top of `ecutwfc_list` (e.g. `30:150:10`) or `dual_list`. |
| `Disk quota exceeded` | QE scratch goes to `$RCAC_SCRATCH/autophonon/` and is deleted after each stage, so check what else is filling home/depot (`myquota`). |
| `already has a driver job ...` | A driver is already working in that directory. Use `autophonon status`, or `scancel` that job first. |
| Want to redo one stage | Delete that stage's directory and its entry under `results` in `state.json` (and everything after it), then `autophonon start <name>.yaml`. Simplest is a new `workdir`. |

The driver job requests 3 days and resubmits itself shortly before the
limit, so long phonon runs need no babysitting.

## Validated systems

| System | Notes | Status |
|---|---|---|
| NaCl | ionic, LO–TO; ecutwfc 80 / ecutrho 640 Ry, a = 5.688 Å, TO 159 / LO 251 cm⁻¹ at Γ (PBE) | done |
| Si | covalent, non-polar; 30 / 120 Ry, a = 5.468 Å, Γ optical 504 cm⁻¹, TA(X) 147, TA(L) 112 cm⁻¹ | done |
| Al | metal; 40 / 160 Ry, a = 4.038 Å, X: LA 323, TA 191 cm⁻¹ (expt. 323, 193) | done |
| GaAs | small PBE gap (0.15 eV at the relaxed a = 5.747 Å): computed epsilon_inf = 546 and LO–TO only 1 cm⁻¹, flagged by the gap check. With `born_file` (expt. epsilon_inf 10.9, Z* 2.07): TO 252 / LO 274 cm⁻¹ (expt. 268 / 292) | done |
| GaN (wurtzite) | polar, hexagonal; 80 / 320 Ry, a = 3.219, c = 5.244 Å, Z*(Ga) = 2.65 (⊥c) / 2.79 (∥c), E2(low) 136, A1(TO) 503, E1(TO) 526 cm⁻¹; ~3 h per 128-atom displacement job | done |

All PBE with the PseudoDojo set above. Inputs and outputs are on Negishi in
`~kim5249/NaCL_Phonons/validation/runs/` (Si, Al, GaN) and
`~kim5249/NaCL_Phonons/auto_test/NaCl_auto_v2/` (NaCl).

## Changing the code

Work on a branch and open a pull request. If you touch the convergence or
phonon stages, rerun NaCl and compare against the table above.
