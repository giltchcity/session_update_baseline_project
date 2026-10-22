"""Ground truth for the 4D video (user 10-22): a session's scene (0A.usd / 0B.usd, as scripts/capture_ab_isaac.py
captured it) at a series of session times, from one fixed camera (a copy of the capture camera's settings with the
focal length set to the given fx at the given width), the ceiling hidden (/Root/Meshes/ceiling, the ceiling lights,
chandeliers and spot lights; the lamps' boxes are written to DONE_<session>.json) and the walls (/Root/Meshes/wall)
scaled in height to the camera's crop top (the map's ceiling cut, crop_box_world z), so the rooms are seen from above
as the map is. Session time t (s) is the scene time start + t: the times t0, t0 + step, ... t1, or the listed --times.
The renderer keeps a history while its camera stands still, and where the depth hardly changes (a TV taken off a wall)
a removed object stays as a ghost for many frames (10-22): --history cut renders one frame of an empty view before
each time, so each image is that time's alone (as a fresh start is; with --subframes 24 it matches a fresh render).
Writes <OUT>/<session>_<t>_rgb.png and _depth.npz.
  render_video_isaac.py --session a|b --camera CAMERA.json (--t0 S --t1 S --step S | --times S,S,...) --output DIR
                        [--history keep|reset|cut] [--subframes N] [--settle N] [--relax-history N] [--no-temporal-aa]"""
import argparse, json, math, re
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--session', choices=['a', 'b'], required=True)
parser.add_argument('--camera', type=Path, required=True)
parser.add_argument('--t0', type=float)
parser.add_argument('--t1', type=float)
parser.add_argument('--step', type=float)
parser.add_argument('--times')
parser.add_argument('--settle', type=int, default=1)       # render steps per time
parser.add_argument('--no-temporal-aa', action='store_true')
parser.add_argument('--history', choices=['keep', 'reset', 'cut'], default='keep')   # see the frame loop
parser.add_argument('--subframes', type=int, default=2)    # render subframes per step
parser.add_argument('--relax-history', type=int)           # the denoiser's history frames (default 31)
parser.add_argument('--output', type=Path, required=True)
args, _ = parser.parse_known_args()
ROOT = Path(r'D:\3Study\ETH\FT\Synthetic_ABC')
scene = ROOT / 'assets/InteriorAgent/kujiale_0021' / f'0{args.session.upper()}.usd'
cam_json = json.loads(args.camera.read_text())
W, H, fx = int(cam_json['width']), int(cam_json['height']), float(cam_json['K'][0][0])
args.output.mkdir(parents=True, exist_ok=True)
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': W, 'height': H, 'renderer': 'RayTracedLighting', 'anti_aliasing': 0,
                     'multi_gpu': False, 'sync_loads': True})
try:
    import numpy as np
    from PIL import Image
    import carb.settings, omni.usd, omni.timeline, omni.kit.app
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom, Gf
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate('omni.anim.curve.core', True)
    manager.set_extension_enabled_immediate('omni.anim.curve.bundle', True)
    context = omni.usd.get_context()
    if not context.open_stage(str(scene)):
        raise RuntimeError('Scene open failed')
    for _ in range(80):
        app.update()
    stage = context.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    hidden, lamps = [], []
    lamp_bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for prim in stage.Traverse():
        p, n = str(prim.GetPath()), prim.GetName()
        if p == '/Root/Meshes/ceiling' or re.match(r'(ceiling_light|chandelier|spot_light)_\d+$', n):
            if p != '/Root/Meshes/ceiling':                          # the lamps' boxes (metres), before they are hidden
                r = lamp_bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                lamps.append([p, [v * mpu for v in r.GetMin()], [v * mpu for v in r.GetMax()]])
            UsdGeom.Imageable(prim).MakeInvisible(); hidden.append(p)
    print('HIDDEN', len(hidden), hidden[:12], flush=True)
    for l in lamps:
        print('LAMP', l[0], [round(v, 2) for v in l[1]], [round(v, 2) for v in l[2]], flush=True)
    top = float(cam_json['crop_box_world']['z'][1]) / mpu
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
    for scope in stage.GetPrimAtPath('/Root/Meshes').GetChildren():
        r = bbox.ComputeWorldBound(scope).ComputeAlignedRange()
        print('SCOPE', scope.GetName(), [round(v, 2) for v in r.GetMin()], [round(v, 2) for v in r.GetMax()], flush=True)
    wall = stage.GetPrimAtPath('/Root/Meshes/wall')
    r = bbox.ComputeWorldBound(wall).ComputeAlignedRange()
    assert abs(r.GetMin()[2]) < 0.05 / mpu, 'walls do not start at the floor'
    wall.SetTypeName('Xform')                                                   # a scope has no transform; the session layer's
    UsdGeom.Xformable(wall).AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, top / r.GetMax()[2]))   # opinion: scaled about the floor
    bbox.Clear()
    print('WALLS', round(r.GetMax()[2] * mpu, 3), '->', round(bbox.ComputeWorldBound(wall).ComputeAlignedRange().GetMax()[2] * mpu, 3), flush=True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_auto_update(False); timeline.set_looping(False); timeline.pause(); timeline.commit()
    rep.orchestrator.set_capture_on_play(False)
    settings = carb.settings.get_settings()
    settings.set('/rtx/post/motionblur/enabled', False); settings.set('/rtx/post/dlss/execMode', 0)
    print('AA', {k: settings.get(k) for k in ('/rtx/post/aa/op', '/rtx/rendermode', '/rtx/post/dlss/execMode')}, flush=True)
    if args.relax_history:      # the denoiser keeps its history where the depth hardly changes (a TV taken off a wall)
        for k in ('/rtx/lightspeed/NRD_ReLAX/diffuseHistoryFrames', '/rtx/lightspeed/NRD_ReLAX/specularHistoryFrames',
                  '/rtx/lightspeed/NRD_ReblurDiffuse/maxAccumulatedFrameNum'):
            settings.set(k, args.relax_history)
        print('RELAX', args.relax_history, flush=True)
    if args.no_temporal_aa:     # a fixed camera keeps the temporal anti-aliasing's history: a removed object stays as a ghost
        settings.set('/rtx/post/aa/op', 0)
        print('AA off', settings.get('/rtx/post/aa/op'), flush=True)
    tcps = stage.GetTimeCodesPerSecond()
    start = stage.GetStartTimeCode() / tcps
    src = UsdGeom.Camera(stage.GetPrimAtPath('/Root/A_capture_camera'))
    cam = UsdGeom.Camera.Define(stage, '/Root/video_camera')
    for name in ('horizontalAperture', 'clippingRange', 'projection', 'fStop', 'focusDistance'):
        a = src.GetPrim().GetAttribute(name)
        if a and a.Get(Usd.TimeCode(start * tcps)) is not None:
            cam.GetPrim().CreateAttribute(name, a.GetTypeName()).Set(a.Get(Usd.TimeCode(start * tcps)))
    ha = float(cam.GetHorizontalApertureAttr().Get())
    cam.GetFocalLengthAttr().Set(fx * ha / W)                                   # fx pixels at width W
    cam.GetVerticalApertureAttr().Set(ha * H / W)                               # square pixels
    cam.GetHorizontalApertureOffsetAttr().Set(0.0); cam.GetVerticalApertureOffsetAttr().Set(0.0)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 1000.0))
    op = cam.AddTransformOp()
    T = np.array(cam_json['T_camera_to_world'], dtype=np.float64) @ np.diag([1., -1., -1., 1.])   # optical -> USD
    T[:3, 3] /= mpu
    M = Gf.Matrix4d(*T.T.flatten().tolist())
    op.Set(M)
    Tcut = T.copy(); Tcut[:3, 3] += np.array([0.0, 0.0, 1000.0]) / mpu                # far above the scene: an empty view
    Mcut = Gf.Matrix4d(*Tcut.T.flatten().tolist())
    product = rep.create.render_product('/Root/video_camera', (W, H))
    ann = {n: rep.AnnotatorRegistry.get_annotator(n, device='cpu') for n in ['rgb', 'distance_to_image_plane']}
    for a in ann.values():
        a.attach(product)
    times = ([float(v) for v in args.times.split(',')] if args.times else
             [round(args.t0 + k * args.step, 4) for k in range(int(math.floor((args.t1 - args.t0) / args.step + 1e-6)) + 1)])
    timeline.set_current_time(start + times[0]); timeline.commit()
    for _ in range(5):
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True, wait_for_render=True)
    for t in times:
        timeline.set_current_time(start + t); timeline.commit()
        for _ in range(2):
            app.update()
        # The renderer keeps its history while the camera stands still: a removed object stays as a fading ghost for
        # many frames (10-22, the TV of B at 60.5 s). 'reset': clear the accumulation; 'cut': one frame of an empty view
        # first, so nothing of the previous frame can be carried over.
        if args.history == 'reset':
            settings.set('/rtx/resetPtAccum', True)
        elif args.history == 'cut':
            op.Set(Mcut)
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=True, wait_for_render=True)
            op.Set(M)
        for _ in range(args.settle):
            rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0, pause_timeline=True, wait_for_render=True)
        rgb = np.asarray(ann['rgb'].get_data())[..., :3].copy()
        depth = np.asarray(ann['distance_to_image_plane'].get_data(), dtype=np.float32).reshape(H, W)
        Image.fromarray(rgb).save(args.output / f'{args.session}_{t:07.2f}_rgb.png')
        np.savez_compressed(args.output / f'{args.session}_{t:07.2f}_depth.npz', depth=depth)
        print('RENDERED', args.session, t, flush=True)
    (args.output / f'DONE_{args.session}.json').write_text(json.dumps({'scene': str(scene), 'camera': str(args.camera), 'times': times, 'settle': args.settle, 'history': args.history, 'subframes': args.subframes, 'relax_history': args.relax_history, 'hidden': hidden, 'lamps_m': lamps, 'walls_top_m': top * mpu}, indent=1))
finally:
    app.close()
