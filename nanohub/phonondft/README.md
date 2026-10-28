# phonondft (nanoHUB Sim2L, draft)

Phonon dispersion of a structure relaxed with
[CellRelaxDFT](https://nanohub.org/tools/cellrelaxdft): finite displacements
(phonopy + Quantum ESPRESSO pw.x) and, for insulators, Born charges and
epsilon_inf from DFPT (ph.x) for the LO-TO correction. Modelled on the
`piezodft` Sim2L: give it the squid of a cellrelaxdft run.

```
run_phonondft.ipynb         driver: paste a squid, edit inputs, Run (like piezodft's workflow)
simtool/phonondft.ipynb     the Sim2L (INPUTS / OUTPUTS, reads the cellrelaxdft run by squid)
simtool/phonondft_lib.py    calculation code (EXTRA_FILES of the Sim2L)
```

No cutoff convergence scan: `ecutwfc` is an input (default 80 Ry, a guess
for norm-conserving pseudopotentials; 0 = reuse the relaxation's cutoff) and
`ecutrho = ecutrho_dual x ecutwfc` (default 8).

## Status

Tested on Negishi with stand-ins for the nanoHUB parts: it reproduces the
NaCl reference from the Negishi workflow (Gamma TO 158.75 / LO 250.77 cm^-1,
Z* = 1.106, epsilon_inf = 2.49). **Not yet run on nanoHUB.** To check there:

1. How cellrelaxdft names its outputs. The relaxed structure is found by
   trying every output as CIF / POSCAR / pw.x text / file / dict;
   `phonondft_lib.check_cellrelax_outputs(squid)` lists them.
2. Where cellrelaxdft keeps its pseudopotentials (else set `pseudo_dir`).
3. The `submit_command`, `pw_executable`, `ph_executable` inputs for nanoHUB's
   QE installation (piezodft's source has the working ones).
4. That phonopy, seekpath and ase are in the Sim2L's Python environment.
