"""T1 ('timed') is a property of a map, decided when the map is made: a backend the update layer drives (run.py rows 4 / 5)
makes timed maps from their first keyframe on; a backend without the layer (GaME's rows 1-3) makes untimed ones; a map
carried into the next session keeps its own value. Before, it turned on only at the layer's first call, so a fresh A's
first round (10.8 s) trained without Z2 and the pixel rule."""
from update_layer.backends.game.game import GameBackend, TrackedGaME
from update_layer.run import dataset_config


def test_timed_is_decided_when_the_map_is_made():
    _, info, specs = dataset_config("synthetic")
    for layered in (True, False):
        be = GameBackend(info, own_update=not layered, layered=layered)
        be.start_session(specs[0], None)
        assert be.game.timed is layered
        carried = be.game
        other = GameBackend(info, own_update=False, layered=not layered)
        other.start_session(specs[0], carried)               # a carried map keeps its own value
        assert other.game.timed is layered
    assert TrackedGaME.timed is False
