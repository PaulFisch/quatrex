# Si-film SCBA run census

**Census date:** 29 August 2026  
**Machine-readable ledger:**
[`si_film_run_census.csv`](../scripts/data/si_film_run_census.csv)  
**Regenerator:** [`si_run_census.py`](../scripts/si_run_census.py)

## Scope and method

This ledger separates historical evidence from the support-complete Si-film
campaign.  The regenerator searches the local campaign mirror for Si-film
configuration files, surviving result arrays and Slurm logs.  It gives the
effective `QX_*` values printed by a run priority over the TOML base
configuration.  Byte-identical result mirrors are merged by SHA-256; metadata
records without a surviving result are merged by Slurm job identifier.  An
alias column retains every merged location.

The frozen historical subset has 61 unique records.  Of these, 29 have a
surviving hashed result, 33 have at least one job identifier, and none of the
surviving logs records a source commit.  The live ledger additionally ingests
new `probe*` and `conv*` campaign artifacts as they are pulled from Daint, so
its total grows during certification.  The missing commit is retained as an
empty field rather than inferred from file dates.

The ledger regenerated after the ballistic linear-response test contains 132
unique records: 40 `analysis-only`, 43 `frequency-truncated`, 44 `superseded`
and five `divergent`.  The ten new zero-broadening ballistic frequency,
q-mesh and temperature-drop records are intentionally `analysis-only`,
because they validate the harmonic solver and contain no interacting SCBA
fixed point.  The two converged auxiliary-grid continuation points and the
stopped \(s=0.9609375\) and \(s=0.96875\) stability probes are `superseded`,
because their vertex scales remain below the physical value one.  These labels prevent either
class from entering the final interacting length curve while retaining both
as numerical evidence.

The classification is deliberately strict.  A trustworthy interacting run
must have a causal FFT retarded reconstruction, zero artificial broadening, a
frequency interval covering the supplied Si spectrum, complete self-energy
support, a stored converged fixed point and an internal current spread no
larger than (10^{-3}).  A run is not promoted because its log contains the
word `converged` when one of these physical gates fails.

## What survives

The historical length distribution is dominated by the three-cell film:

| Primitive cells | Records |
|---:|---:|
| 3 | 40 |
| 4 | 6 |
| 5 | 3 |
| 8 | 4 |
| 10 | 7 |
| 12 | 1 |

Fifty-seven records use the (1\times9\times9) mesh, two use
(1\times5\times5), and two use (1\times13\times13).  The recorded upper
frequency boundaries are 15 THz for 36 runs, 20 THz for 17 runs and 28 THz
for eight runs.  No historical artifact reaches the 31.5, 35 or 40 THz
extent required by the present convergence test.  Fifty-five records use the
old half-retarded approximation and six use causal FFT reconstruction.  Only
one record uses a nonuniform primary grid.

Twenty-four artifacts say that their algebraic SCBA iteration converged.
They are not certified physical results.  The census assigns 36 records to
`frequency-truncated`, 20 to `analysis-only`, and five to `divergent`; the
intersection of the support, spectral, causal and conservation gates is
empty.  The previous 3/5/8-cell length comparison therefore cannot establish
a length trend for the cubic-bubble model.

The L5 and L10 evidence is particularly limited.  All three L5 records and
all seven L10 records stop at 15 THz with 0.125 THz spacing and use the old
retarded approximation.  The interacting records also precede the primitive
microblock output and consequently do not certify the generated support law.
The apparent availability of rank-8 through rank-128 L10 variants is a factor
study on this truncated legacy problem, not a dense-vertex accuracy gate for
the new calculation.

The L8 reblocking result is nevertheless a genuine algebraic convergence
record.  Job 4553056 used four two-cell groups, q=9, zero broadening, the
half-retarded rule and a 0--15 THz grid with 0.125 THz spacing.  It converged
after 17 evaluations to a retarded residual of (9.5112\times10^{-4}), lead
balance (5.85\times10^{-6}) and internal spread
(2.829\times10^{-3}).  Its current was written before the frequency-cell
measure correction and must be multiplied by 0.125 for comparison with new
outputs.  The row establishes that two-cell reblocking helped the historical
functional.  Its short window, half rule, legacy output support and failed
(10^{-3}) internal-spread gate prevent its use as the new length baseline.

## Consequence for the new campaign

No old row is used as a converged baseline.  Historical results remain useful
for failure diagnosis, restart provenance and rough cost estimates.  New L5
and L10 certification points are added only after their complete metadata and
fixed-point arrays have been pulled from Daint.  A failed gate remains in the
ledger as `superseded`, `divergent`, `frequency-truncated` or
`spatially-pinned`; it is never replaced by a more favourable transient.

The present campaign begins with one grouped Dyson block for L5 while the FC3
contraction remains on five primitive six-DOF cells.  It compares a batched
dense q-folded vertex with independently fitted rank-64 and rank-128 factors
on the same frozen Green functions.  Subsequent spatial, frequency, q-mesh,
temperature and residual refinements use only a factor route that passes that
dense gate.  This ordering prevents the historical rank comparisons from
being mistaken for evidence that the factor error is smaller than the target
0.2 per cent conductance tolerance.

Regenerate the table with

```text
python phonon/scripts/si_run_census.py
```

and verify that a committed ledger is current with

```text
python phonon/scripts/si_run_census.py --check
```
