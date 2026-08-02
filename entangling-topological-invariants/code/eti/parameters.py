"""Named parameter choices used in the manuscript figures.

Only values that define the displayed deformation paths are kept here.  Grid
sizes, convergence orders, and statistical budgets remain next to the
calculations that use them.
"""

from __future__ import annotations

# Mixed-label path, Eq. (S24): (11, 10, 7, 4) / 20.
LABEL_MIXING_DIRECTION = (11 / 20, 10 / 20, 7 / 20, 4 / 20)
LABEL_MIXING_ENDPOINT = 6 / 5
LABEL_MIXING_SCAN_MAX = 8 / 5

# Endpoint of the interblock-coupling path in the S4 construction.
S4_COUPLING_ENDPOINT = 5 / 4

# Initial flux used for the finite-cylinder cycle.  The offset is selected from
# the plateau in Fig. S2(d), away from the finite-size edge anticrossing.
PUMP_PHASE_OFFSET = 0.37
