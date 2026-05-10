"""Per-system validation pipeline for finite (no-q) phonon NEGF inputs.

Produces a uniform report covering FC2/FC3 quality and convergence,
sparsity structure (1D decay, 2D heatmap, 3D scatter), tensor-decomposition
trade-offs, physical invariants (ASR, permutational symmetry, Hermiticity),
phonon-phonon SSE sparsity (synthetic + quatrex), and the cutoff hierarchy
sensitivity (diagonal-G, NN-only, distance/magnitude FC3 thresholds).

The driver is system-agnostic; each example structure is described by the
same YAML schema as the existing ``phonon_inputs`` pipeline.
"""

from .loader import SystemBundle, load_system

__all__ = ["SystemBundle", "load_system"]
