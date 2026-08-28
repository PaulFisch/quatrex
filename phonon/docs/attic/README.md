# Attic

Superseded working notes, kept because they are the only record of a design or
of an investigation, and moved out of `phonon/docs/` so that what remains there
is current.

| file | why it is here | what replaced it |
|---|---|---|
| `pole_subtracted_modal_scba.md` (3526 lines) | The design note for the pole-subtracted modal SCBA. Its own companion, `pole_scba_implemented.md`, states that it records the formulas *as built* and that "where the two disagree, the design note is the older document". Its Part II (the spatial/modal half) was never implemented and is the only place that design is written down | `pole_scba_implemented.md` for the frequency half, which is what shipped; the report's "Resolving sharp lines off the grid" for the construction and its conditions |
| `quatrex_phph_implementation_notes.md` (169 lines) | A May-June 2026 consolidation of the dense-reference investigation, written to guide the production solver. Its "Status of the production quatrex phph -- KNOWN DEFECT" section describes a defect fixed long since, and its truncation ranking is superseded by the measured spatial support law | The report's "What may be discarded" and its results counterpart; `spatial_representation.md` Sec. 0.1-0.2 for the index algebra |
| `spatial_truncation_derivation.md` (320 lines) | The index algebra of the ring's three spatial truncations, the support law, what the output pin costs, and the error trail behind them. Correct and not superseded -- absorbed whole rather than retired, because the programme it underpins now has a verdict and the record should be one document | `spatial_representation.md` Part 0, Secs. 0.1-0.3 and the error trail at 0.7 |
| `spatial_band_range.md` (329 lines) | How far a damped mode travels, measured on Si and CNT, plus the modal reconstruction and the mask-PSD bound. Carried a Status header withdrawing its own central inference, which is exactly the kind of reader-beware that consolidation removes | `spatial_representation.md` Secs. 0.4-0.6; the withdrawal is stated in 0.4 |

Nothing outside this directory should cite these files. Everything still cited
from code lives in `phonon/docs/`.
