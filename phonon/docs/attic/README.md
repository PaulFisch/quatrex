# Attic

Superseded working notes, kept because they are the only record of a design or
of an investigation, and moved out of `phonon/docs/` so that what remains there
is current.

| file | why it is here | what replaced it |
|---|---|---|
| `pole_subtracted_modal_scba.md` (3526 lines) | The design note for the pole-subtracted modal SCBA. Its own companion, `pole_scba_implemented.md`, states that it records the formulas *as built* and that "where the two disagree, the design note is the older document". Its Part II (the spatial/modal half) was never implemented and is the only place that design is written down | `pole_scba_implemented.md` for the frequency half, which is what shipped; the report's "Resolving sharp lines off the grid" for the construction and its conditions |
| `quatrex_phph_implementation_notes.md` (169 lines) | A May-June 2026 consolidation of the dense-reference investigation, written to guide the production solver. Its "Status of the production quatrex phph -- KNOWN DEFECT" section describes a defect fixed long since, and its truncation ranking is superseded by the measured spatial support law | The report's "What may be discarded" and its results counterpart; `spatial_truncation_derivation.md` for the index algebra |

Nothing outside this directory should cite these files. Everything still cited
from code lives in `phonon/docs/`.
