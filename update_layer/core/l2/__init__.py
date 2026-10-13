"""The decision core of the TSDF version (session_core @192c1cf, L2_FINAL2), ported line by line.

evidence.py  <- src/evidence/projected_physical_evidence.cpp (observed absence, eq. 7; P21/P22/P25)
state.py     <- src/state/persistent_object_state.cpp (identity -> fragment states, M1c-M1l, M2a, S/M/U)
replay_absence.py, replay_state.py: consistency tests against the L2_FINAL2 khronos.log
Nothing is re-derived or tuned; every constant and comment that justifies it is the original's.
"""
