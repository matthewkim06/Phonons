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
git clone git@github.com:matthewkim06/Phonons.git
cd Phonons
module load anaconda
conda env create -f environment.yml      # creates the "autophonon" env
```

Quantum ESPRESSO is not installed by conda. The Slurm jobs load the cluster
module (`gcc/12.2.0 openmpi/4.1.4 quantum-espresso/7.1` by default; change
`modules` in your config if you need another build).

**Pseudopotentials.** Use one UPF file per element in a directory. We use
[PseudoDojo](http://www.pseudo-dojo.org) norm-conserving, PBE, scalar-relativistic,
"standard" accuracy, v0.4 (UPF format):

```bash
mkdir -p ~/pseudo/dojo-pbe && cd ~/pseudo/dojo-pbe
curl -LO http://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz
tar xzf nc-sr-04_pbe_standard_upf.tgz && rm nc-sr-04_pbe_standard_upf.tgz
```

The script reads the element from each file's header, so file names don't
matter. A directory may hold the whole table; only the elements in your
structure are used. Ultrasoft/PAW files work too, but set `scan_dual: 12` or
higher, since they need a larger ecutrho/ecutwfc ratio.

## Running a calculation

```bash
conda activate autophonon
AP=~/Phonons/autophonon.py             # wherever you cloned it

# 1. Write a config file (every option listed with its default, commented out)
python $AP init --name NaCl --structure NaCl.cif --pseudo-dir ~/pseudo/dojo-pbe

# 2. Optional: see exactly what will be submitted
python $AP run NaCl.yaml --dry-run

# 3. Start it. This submits a small 1-core "driver" job that submits and
#    waits for the real QE jobs. You can log out.
python $AP submit NaCl.yaml

# 4. Check on it at any time
python $AP status NaCl.yaml
```

Outputs go to `NaCl_auto/` next to the config file. The driver's log is
`NaCl_auto/driver.log`.

The structure can be anything ASE reads (CIF, POSCAR, pw.x input, ...). It is
symmetrized and standardized first, so a conventional or primitive cell are
both fine.

**Metals:** add `metal: true` (Marzari–Vanderbilt smearing, `degauss` 0.02 Ry
by default; no Born charges).

**Changing options:** uncomment lines in the YAML, or pass `--flags` on the
command line (flags override the file). `python $AP run -h` lists everything.

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
* `status` prints the converged cutoffs, relaxed lattice, Γ frequencies and
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
| `status` says the driver is not running, stages pending | Look at the end of `driver.log`, fix the cause, then `submit` again; it resumes. |
| `Stage ... failed; incomplete outputs` | Read the `.out` and `.err` files in that stage directory (walltime, memory, SCF not converging). Raise `walltime` / `disp_walltime` if needed, then `submit` again. |
| `not converged within [...]; extend the list` | Raise the top of `ecutwfc_list` (e.g. `30:150:10`) or `dual_list`. |
| `Disk quota exceeded` | QE scratch goes to `$RCAC_SCRATCH/autophonon/` and is deleted after each stage, so check what else is filling home/depot (`myquota`). |
| `already has a driver job ...` | A driver is already working in that directory. Use `status`, or `scancel` that job first. |
| Want to redo one stage | Delete that stage's directory and its entry under `results` in `state.json` (and everything after it), then `submit` again. Simplest is a new `workdir`. |

The driver job requests 3 days and resubmits itself shortly before the
limit, so long phonon runs need no babysitting.

## Validated systems

| System | Notes | Status |
|---|---|---|
| NaCl | ionic, LO–TO; ecutwfc 80 / ecutrho 640 Ry, a = 5.688 Å, TO 159 / LO 251 cm⁻¹ at Γ (PBE) | done |
| Si | covalent, non-polar; 30 / 120 Ry, a = 5.468 Å, Γ optical 504 cm⁻¹, TA(X) 147, TA(L) 112 cm⁻¹ | done |
| Al | metal; 40 / 160 Ry, a = 4.038 Å, X: LA 323, TA 191 cm⁻¹ (expt. 323, 193) | done |
| GaN (wurtzite) | polar, hexagonal; 80 / 320 Ry, a = 3.219, c = 5.244 Å, Z*(Ga) = 2.65 (⊥c) / 2.79 (∥c), E2(low) 136, A1(TO) 503, E1(TO) 526 cm⁻¹; ~3 h per 128-atom displacement job | done |

All PBE with the PseudoDojo set above. Inputs and outputs are on Negishi in
`~kim5249/NaCL_Phonons/validation/runs/` (Si, Al, GaN) and
`~kim5249/NaCL_Phonons/auto_test/NaCl_auto_v2/` (NaCl).

## Changing the code

Work on a branch and open a pull request. If you touch the convergence or
phonon stages, rerun NaCl and compare against the table above.
