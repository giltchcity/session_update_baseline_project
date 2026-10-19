"""CPU tests of calibrated keyframe rotation spacing, without renderer setup.

The production method body is executed unchanged on CPU camera fixtures. The
fixtures check observable geometry, the SO(3) angle and world-coordinate invariance.
"""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
import torch


tree = ast.parse((Path(__file__).resolve().parents[1]/"backends/game/game.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrackedGaME")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "is_keyframe")
scope = dict(np=np, math=math, Rotation=Rotation)
exec(compile(ast.Module(body=[method], type_ignores=[]), "production:is_keyframe", "exec"), scope)
select = scope["is_keyframe"]


def camera(pose=None, horizontal=.5, vertical=.25):
    K = np.array([[100., 0, 100*horizontal], [0, 100., 100*vertical], [0, 0, 1.]])
    return SimpleNamespace(keyframes={0: dict(pose=torch.tensor(np.eye(4) if pose is None else pose), intrinsics=K)},
                           _last_keyframe_id=0, config={"keyframe_translation_diff": 1.25})


def pose(axis=(1., 0, 0), radians=0., translation=(0, 0, 0)):
    p = np.eye(4); a = np.array(axis, dtype=float); a /= np.linalg.norm(a)
    p[:3, :3] = Rotation.from_rotvec(a*radians).as_matrix()
    p[:3, 3] = translation
    return p


def test_first_and_stationary_views():
    assert select(SimpleNamespace(keyframes={}), np.eye(4))
    assert not select(camera(), np.eye(4))


def test_vertical_extent_prevents_missing_the_central_view():
    c = camera(); radius = math.atan(.25)
    assert not select(c, pose(radians=radius*.99))
    assert select(c, pose(radians=radius*1.01))
    ray = pose(radians=radius*1.01)[:3, :3] @ np.array([0., 0, 1])
    assert abs(ray[1]/ray[2]) > .25


def test_spacing_comes_from_calibration():
    turn = pose(radians=math.radians(20))
    assert select(camera(vertical=.25), turn)
    assert not select(camera(horizontal=1., vertical=1.), turn)


def test_combined_axes_use_the_total_rotation_and_roll_is_retained():
    radius = math.atan(.25)
    assert select(camera(), pose(axis=(1, 1, 0), radians=radius*1.1))
    assert select(camera(), pose(axis=(0, 0, 1), radians=radius*1.1))


def test_inside_the_radius_keeps_the_optical_axis_in_the_image():
    radius = math.atan(.25)
    for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (-3, 2, 1)]:
        p = pose(axis, radius*.99)
        assert not select(camera(), p)
        ray = p[:3, :3] @ np.array([0., 0, 1])
        assert abs(ray[0]/ray[2]) < .5 and abs(ray[1]/ray[2]) < .25


def test_translation_spacing_is_unchanged():
    assert not select(camera(), pose(translation=(1.24, 0, 0)))
    assert select(camera(), pose(translation=(1.26, 0, 0)))


def test_selection_is_independent_of_world_coordinates():
    change = pose((1, 2, 3), .8, (23, -16, 7))
    for p in [np.eye(4), pose(radians=.3), pose(translation=(1.3, 0, 0))]:
        expected = select(camera(), p)
        transformed = select(camera(np.linalg.inv(change)), p @ np.linalg.inv(change))
        assert bool(transformed) == bool(expected)
