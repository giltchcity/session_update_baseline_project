"""[O2] The t_L (kTrackFirstSeenDetail) of a state formed at a new site is the first frame of ITS sightings.

Real C 2026-10-08 (records 13:20): the layer's running first sighting of an identity (38.1 s, the round after a
short state's death) was carried into every later observation, so the observed_new slot that became the suitcase's
final state (first seen at the new place at ~99 s) exported t_L 38.1 s and T1 gave it every Gaussian created after
38.1 s, the mid-place ones included; the mid place never ended. The slot takes its own first frame, and later
observations merged into it cannot pull it earlier than their own first frames.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from update_layer.core.l2.state import Observation, PersistentObjectState, PhysicalState  # noqa: E402

S = 1_000_000_000


def obs(first, track_first_seen):
    return Observation(identity=2, points=torch.zeros((3, 3), dtype=torch.float32), normals=None, first=first, last=first,
                       track_first_seen=track_first_seen)


def test_new_site_slot_takes_its_own_first_frame_and_keeps_it():
    reg = PersistentObjectState()
    st = PhysicalState()
    reg.merge_observed_new(st, obs(99 * S, 38 * S))          # first sighting at the new site, running fs 38 s
    assert st.observed_new.track_first_seen == 99 * S
    assert st.observed_new.birth_time == 99 * S
    reg.merge_observed_new(st, obs(101 * S, 38 * S))         # a later sighting still carries the running fs
    assert st.observed_new.track_first_seen == 99 * S
    reg.merge_observed_new(st, obs(97 * S, 38 * S))          # an earlier sighting of the site moves it to ITS first
    assert st.observed_new.track_first_seen == 97 * S


def test_current_state_keeps_the_running_first_sighting():
    reg = PersistentObjectState()
    st = PhysicalState()
    st.fragments.append(reg.make_fragment(obs(40 * S, 36 * S)))   # CURRENT formed from sightings since 36 s
    st.current = 0
    reg.merge_observation_into_fragment(st.fragments[0], obs(50 * S, 36 * S))
    assert st.fragments[0].track_first_seen == 36 * S
