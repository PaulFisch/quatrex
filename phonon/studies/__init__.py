"""Phonon-transport investigations on the production quatrex solver.

One module per investigation, each exposing ``run(args)`` and ``plot(args)``:

- ``conservation`` -- vertex S3 gate + discrete bubble energy-balance replica.
- ``linewidths``   -- phono3py vs NEGF single-shot vs SCBA linewidths.
- ``ballistic``    -- ballistic conductance / transmission + eta study.
- ``convergence``  -- strong-coupling SCBA convergence (mixing, vertex-scale
  continuation, annealing).
- ``transport``    -- production T-sweeps / length ladders, spectral current,
  summaries and document figures.

Run from the repo root::

    python -m phonon.studies <investigation> {run,plot} [options]

All plots go through :mod:`phonon.studies.style` (uniform style, png+pdf),
all solver cells through :mod:`phonon.studies.pipeline` (node hygiene,
single-threaded BLAS, full npz snapshot per run).
"""
