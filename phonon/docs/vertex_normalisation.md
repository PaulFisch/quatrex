# The three-phonon vertex against phono3py, element by element

Step 1 of the 2026-08-15 convergence plan. Script:
`phonon/studies/_vertex_element_check.py`. Bed: the checked-in Si primitive
reap `phonon/reaps/si_primitive_work/` (2-atom primitive, 2x2x2 FC supercell,
phono3py 3.29.0). No cluster time; no production path changed.

## Why an element comparison and not a linewidth

Every previous attempt to pin the absolute strength of the 3-phonon self-energy
went through a linewidth: ours against phono3py's golden-rule `gamma`. That
route is structurally confounded. `lab_notebook_archive.md` F28 measured the
ratio mode by mode and found it neither mode-independent nor mesh-converged
(spread 138% at 2^3, 172% at 4^3), because the low-frequency acoustic modes are
joint-DOS starved: phono3py's `gamma -> 0` while the NEGF Lorentzian still picks
up background. F28's own verdict was that the linewidth route "cannot determine
the absolute prefactor -- final verdict on this method". A finer mesh does not
repair a sampling problem.

`Interaction.interaction_strength` is the same physics one step earlier:
`|Phi_{lambda lambda' lambda''}|^2` for individual triplets, with no delta
function, no Brillouin-zone sum, no broadening and no mesh convergence in it.
Everything the comparison needs is checked in.

## What was measured

Five stages, each isolating one thing. Both sides of every comparison use the
*same* eigenvectors, so eigenvector gauge and degenerate-subspace rotation never
enter.

| stage | question | result |
| --- | --- | --- |
| S0 | does phono3py's C kernel agree with its own python reference? | 3.4e-16 at `make_r0_average=False` |
| S1 | does our contraction reproduce `interaction_strength`? | 2.2e-16 |
| S2 | **is the code's FC3 normalisation right?** | ratio = `CONVERSION_FC3_THZ^2`, worst deviation **1.1e-13** |
| S3 | **is the code's Fourier/gauge convention right?** | **0.0e+00** at commensurate q, after the documented A -> B gauge |
| S4 | the same at the level of `D_B(q)`, eigenvector-free | 1.2e-16 commensurate; 25-46% at q = 1/4 |
| S5 | which periodic image does each side use? | see below |

### S2 -- the normalisation is exact, not approximate

The code's mass-weighted real-space vertex
(`phonon_inputs/separable.py::build_realspace_fc3_matrices`) is phono3py's
raw FC3 divided by `sqrt(m_i m_j m_k)` and multiplied by
`CONVERSION_FC3_THZ = 2.3926935378e+19`. Pushing it through phono3py's own
Fourier transform and phono3py's own eigenvectors, the ratio of `|Phi|^2` to
phono3py's is that constant squared for **every triplet and every band triplet**,
to 1.1e-13.

This closes Workstream A in the "is there a missing factor" sense, and closes it
far harder than F28 could: F28's area-integrated ratio `R/(2pi)^2 ~ 1.06` put
native within ~15% with `div4` and `x4` excluded by an order of magnitude. The
element comparison says the vertex is not within 15% of phono3py's -- it *is*
phono3py's.

Caveat on what Si can and cannot test: all masses are equal, so `1/sqrt(m_i m_j
m_k)` is a global constant here and the per-atom mass *indexing* is not
exercised. A two-species bed (MoS2, TiS3) would test that; this bed cannot.

### S3/S4 -- the convention is right, and `convention.py` is why

Read naively, the code's Fourier fold looks wrong by a median factor 5.1 with a
seven-decade spread. It is not. `phonon_inputs/convention.py` documents the two
conventions --

    A (phonopy):  phase = exp(2 pi i q . (R + tau' - tau))
    B (quatrex):  phase = exp(2 pi i q . R)

-- and gives the gauge `D_B = P D_A P^dagger`, `P = diag(exp(2 pi i q . tau))`.
Applying that gauge, the code's cell-translation fold reproduces phono3py's
reciprocal-space FC3 to **0.0e+00** at q commensurate with the FC supercell. The
factor 5.1 was the gauge and nothing else.

The device stack does apply it: `get_btd_blocks_folded` calls
`gauge_transform_A_to_B` before the transport IDFT, so H and Phi live in the same
convention. S4 confirms it at matrix level -- the code's convention-B `D(q)`
equals phonopy's gauge-transformed `D` to 1.2e-16 at commensurate q.

### S5 -- what is *not* shared: the choice of periodic image

At q **not** commensurate with the FC supercell the two sides part company, and
the reason is not the gauge:

* H comes from phonopy's dynamical matrix, whose Fourier sum uses the shortest
  vectors `R + tau_kappa - tau_i`, **averaged over ties**;
* the vertex fold (`solver/se_q.py::_qfold_device_blocks`,
  `separable.build_gathering_matrix`) uses **one wrapped cell index** per atom,
  with no basis offset and no tie average.

On this 2x2x2 bed that is worth 25-46% of `||D_B||` at q = 1/4, and 2.1-4.5 THz
in the phonon frequencies on a 15.4 THz spectrum. 62.5% of (supercell atom,
reference atom) pairs have transversely-degenerate images here; a further 6.2%
have an unambiguous image that the wrap places elsewhere.

The code's own comment in `build_supercell_mapping` anticipates the aliasing and
applies a transverse minimum-image wrap. The measurement says the wrap is
necessary but not sufficient: it fixes the index ambiguity, not the tie average,
and not the basis-offset contribution to which image is shortest.

phono3py's `make_r0_average` is the same kind of interpolation choice one level
up -- it is worth 2.05% in `|Phi|^2` at incommensurate q and exactly zero at
commensurate q. The code does not apply it.

## What this does and does not say about production

**Does not** say production is wrong. The production Si film uses
`reaps/si_big_hiphive` (5x5x5 supercell, verified on the cluster) with an odd
`nk = 9`, and the CNT beds are transversely finite (`nk = 1`, Gamma only), where
the question does not arise at all.

**Does** say the size of the effect on the film is unmeasured. Counting pairs on
the film geometry -- rebuilt from the reap's `phono3py.yaml` (fcc primitive,
a = 2.734 A, 2 atoms, 5x5x5, transport x); `get_smallest_vectors` needs the
geometry only, not the force constants -- 22.8% of pairs are
transversely degenerate and 4.0% are unambiguous-but-placed-elsewhere. Those are
*pair counts*, an upper bound on what can differ, not an error -- the weight
those pairs actually carry is set by the hiphive FC3 cutoff, which is well
inside the 5x5x5 box. Quoting 22.8% as an error would be exactly the
fill-fraction mistake catalogued in `bubble_positivity.md` §6.10.

The measurement that would settle it is the FC-weighted one, i.e. the S4 column
computed on `si_big_hiphive` rather than on this 2x2x2 bed: `rel |dD_B|` between
the code's cell fold and phonopy's shortest-vector fold, at the nine production
`q = k/9`. It needs `fc2.hdf5` from the cluster reap and no compute.

## Regression

`python -m phonon.studies._vertex_element_check --mesh 2 2 2 --grid-point 1 --gate`
exits non-zero unless S2 holds everywhere (< 1e-12) and S3 holds at the
commensurate mesh (< 1e-10). Not a pytest module: `tests/quatrex/phonon/`
carries an `__init__.py` while `tests/quatrex/` does not, so inside the suite the
top-level name `phonon` resolves to the *test* package and shadows the repo's
`phonon/` namespace package.

## Two dead registrations, cleared

`phonon/studies/__main__.py` advertised `sse_verify` and `bte_linewidths`. Neither
file has a blob anywhere in history, so both verbs raised `ModuleNotFoundError`
on dispatch; both are removed. `phonon/scripts/verify/d5a_gamma_anh.py` imports
`bte_linewidths._bte_machinery` and its docstring advertised that machinery as
"Si-validated ... vertex 3% vs phono3py on Si". The code behind that number is
absent, so the claim is withdrawn in place, with a pointer to the measurement
that replaces it -- the vertex is not 3% off, it is exact.
