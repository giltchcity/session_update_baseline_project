"""Common observed-space geometry evaluator; all distances are Euclidean metres.

Source surfaces: seeded area samples -> fixed origin 2cm voxel union. Each occupied
surface voxel receives one vote, removing object/background overlap multiplicity.
Targets: exact closest-triangle distance (Open3D RaycastingScene, unsigned).
Observed-space domain: >=3 valid raw-depth rays place a query in front of, or <=
10cm behind, the measured surface. No method- or GT-distance-based spatial crop.
The unfiltered scores and domain coverage are retained as separate diagnostics.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from PIL import Image

ROOT = Path("/home/jixian/Desktop/FT")
OUT = ROOT / 'results/abc_eval_v2/geometry'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def surface_sample(mesh, spacing=.02, seed=20260909):
    """Fixed voxel origin; dense equal-area samples, then union without layer bias."""
    o3d.utility.random.seed(seed)
    area = float(mesh.get_surface_area())
    if not len(mesh.triangles) or area == 0:
        return np.empty((0, 3), np.float32), dict(area_m2=0, dense_samples=0,
            union_surface_voxels=0, spacing_m=spacing, random_seed=seed)
    count = max(1000, int(np.ceil(area / (.005 ** 2))))
    # Full carried maps contain many overlapping background triangles (C ~712m2
    # before voxel union). This is verified source geometry, not a unit anomaly.
    # Keep the same sampling density and let the voxel union remove multiplicity.
    if count > 50000000:
        raise ValueError('Unexpected mesh surface area: %.3f m2' % area)
    sampled = np.asarray(mesh.sample_points_uniformly(count).points)
    keys = np.floor(sampled / spacing).astype(np.int32)
    # Select a real surface sample nearest its cell centre, rather than an average
    # that could lie off all faces where crossing/thin surfaces share a cell.
    error = np.sum((sampled - (keys + .5) * spacing) ** 2, axis=1)
    order = np.lexsort((error, keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    first = np.r_[True, np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)]
    return sampled[order[first]].astype(np.float32), dict(area_m2=area, dense_samples=count,
        union_surface_voxels=int(first.sum()), spacing_m=spacing, random_seed=seed)


def observed_domain(points, domain, margin=.10):
    poses, depths, intr = domain['camera_to_world'], domain['depth'], domain['intrinsics']
    eligible = np.zeros(len(points), np.uint16)
    supported = np.zeros(len(points), np.uint16)
    free = np.zeros(len(points), np.uint16)
    for pose, depth in zip(poses, depths):
        p = (points - pose[:3, 3]) @ pose[:3, :3]
        z = p[:, 2]
        candidate = np.flatnonzero((z > .1) & (z <= 5.))
        if not len(candidate): continue
        q = p[candidate]
        u = np.rint(q[:, 0] / q[:, 2] * intr[0] + intr[2]).astype(int)
        v = np.rint(q[:, 1] / q[:, 2] * intr[1] + intr[3]).astype(int)
        inside = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
        candidate, u, v = candidate[inside], u[inside], v[inside]
        d = depth[v, u]
        valid = (d > .1) & (d <= 5.)
        candidate, d = candidate[valid], d[valid]
        delta = d - z[candidate]
        eligible[candidate[delta >= -margin]] += 1
        supported[candidate[np.abs(delta) <= margin]] += 1
        free[candidate[delta > margin]] += 1
    return eligible, supported, free


def distances(points, mesh):
    if not len(points): return np.empty(0)
    if not len(mesh.triangles): return np.full(len(points), np.inf)
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    result = []
    for begin in range(0, len(points), 100000):
        query = o3d.core.Tensor(np.ascontiguousarray(points[begin:begin+100000], dtype=np.float32))
        result.append(scene.compute_distance(query, nthreads=2).numpy())
    return np.concatenate(result).astype(np.float64)


def summaries(dp, dr):
    def percentile(values, q):
        if not len(values): return None
        if not np.isfinite(values).any(): return float('inf')
        return float(np.quantile(values, q))
    result = dict(accuracy_mean_m=float(np.mean(dp)) if len(dp) else None,
        completeness_mean_m=float(np.mean(dr)) if len(dr) else None,
        accuracy_median_m=float(np.median(dp)) if len(dp) else None,
        accuracy_p90_m=percentile(dp, .9),
        completeness_median_m=float(np.median(dr)) if len(dr) else None,
        completeness_p90_m=percentile(dr, .9),
        pred_points=len(dp), reference_points=len(dr))
    for threshold in [.05, .10, .20]:
        precision = float(np.mean(dp <= threshold)) if len(dp) else 0.
        recall = float(np.mean(dr <= threshold)) if len(dr) else 0.
        tag = '%02dcm' % round(threshold * 100)
        result.update({('P_' + tag): precision, ('R_' + tag): recall,
            ('F1_' + tag): 2 * precision * recall / (precision + recall) if precision + recall else 0.})
    return result


def reference_sample(session):
    reference_path = OUT / (session + '_reference_2cm.ply')
    domain_path = OUT / (session + '_domain.npz')
    signature = dict(reference_sha256=file_hash(reference_path),
                     domain_sha256=file_hash(domain_path),
                     sampling_and_domain_source_sha256=file_hash(__file__))
    reference = o3d.io.read_triangle_mesh(str(reference_path))
    cache = OUT / (session + '_reference_samples.npz')
    metadata_path = OUT / (session + '_reference_sampling.json')
    cache_valid = cache.exists() and metadata_path.exists() and all(
        json.loads(metadata_path.read_text()).get(key) == value for key, value in signature.items())
    if cache_valid:
        data = np.load(cache)
        return reference, data['points'], data['eligible'], data['supported'], data['free']
    points, meta = surface_sample(reference)
    domain = np.load(domain_path)
    eligible, supported, free = observed_domain(points, domain)
    np.savez_compressed(cache, points=points, eligible=eligible, supported=supported, free=free)
    meta.update(signature)
    metadata_path.write_text(json.dumps(meta, indent=2))
    return reference, points, eligible, supported, free


def score(session, method, mesh_path):
    start = time.time()
    reference, rp, re, rs, rf = reference_sample(session)
    prediction = o3d.io.read_triangle_mesh(str(mesh_path))
    pp, sample_meta = surface_sample(prediction)
    domain = np.load(OUT / (session + '_domain.npz'))
    pe, ps, pf = observed_domain(pp, domain)
    dp, dr = distances(pp, reference), distances(rp, prediction)
    primary = summaries(dp[pe >= 3], dr[re >= 3])
    result = dict(session=session, method=method, mesh=str(mesh_path),
        reference=str(OUT / (session + '_reference_2cm.ply')), primary=primary,
        unfiltered_diagnostic=summaries(dp, dr), sampling=sample_meta,
        coverage=dict(prediction_observed_fraction=float(np.mean(pe >= 3)),
                      reference_observed_fraction=float(np.mean(re >= 3)),
                      pred_total_points=len(pp), reference_total_points=len(rp)),
        domain_sensitivity={str(v): summaries(dp[pe >= v], dr[re >= v]) for v in [1, 3, 5]},
        depth_evidence_diagnostic=dict(
            pred_supported_3_fraction=float(np.mean(ps >= 3)),
            pred_free_3_never_supported_fraction=float(np.mean((pf >= 3) & (ps == 0))),
            reference_supported_3_fraction=float(np.mean(rs >= 3)),
            reference_free_3_never_supported_fraction=float(np.mean((rf >= 3) & (rs == 0)))),
        seconds=time.time()-start,
        coordinate_frame='session-A world',
        domain_policy='Person and nonfinal cabinet masked raw depth; >=3 of 2Hz depth frames expose the point as free or within 0.1m of measured surface. Occluded-only and unobserved points excluded; coverage reported. No prediction-distance or GT-distance crop.',
        geometry_identity_limit='Surface within threshold of legal replacement geometry counts as consistent; geometric equality does not prove mesh or object identity was replaced.',
        source_sha256=file_hash(__file__),
        prediction_sha256=file_hash(mesh_path),
        reference_sha256=file_hash(OUT / (session + '_reference_2cm.ply')),
        domain_sha256=file_hash(OUT / (session + '_domain.npz')))
    np.savez_compressed(OUT / (method + '_' + session + '_distance_samples.npz'),
                        pred_points=pp, pred_distance_m=dp, pred_eligible_rays=pe,
                        pred_supported_rays=ps, pred_free_rays=pf, reference_distance_m=dr)
    (OUT / (method + '_' + session + '_metrics.json')).write_text(json.dumps(result, indent=2))
    print(method, session, json.dumps(primary), flush=True)
    return result


def audit_reference(session):
    """Independent raw depth probes at held-out phase relative to domain sampling."""
    cfg = json.loads((ROOT / 'results/abc_eval_final_baseline/input_manifest.json').read_text())['sessions'][session]
    root = Path(cfg['rgbd']); intr = cfg['intrinsics']; align = np.array(cfg['transform'])
    paths = dict(corrected=OUT / (session + '_reference_2cm.ply'),
        old=ROOT / 'runs/panoptic_mapping/o3d_tsdf_ref_1080p_20260909' / ('session_'+session.lower()+'_o3d_tsdf_mask_v0.02.ply'))
    points = []
    for index in np.linspace(200, cfg['frame_count'] - 1, 20).astype(int):
        stem = '%06d' % index
        depth = np.asarray(Image.open(root / (stem+'_depth.tiff')))
        sem = np.asarray(Image.open(Path(cfg['semantic'])/(stem+'_segmentation.png')))
        v, u = np.mgrid[0:1080:16, 0:1920:16]
        z = depth[v,u]
        good = np.isfinite(z)&(z>.1)&(z<=5)&(sem[v,u]!=12)&(sem[v,u]!=10)
        q = np.column_stack(((u[good]-intr['u'])/intr['f_x']*z[good],
                             (v[good]-intr['v'])/intr['f_y']*z[good],z[good]))
        pose = align @ np.loadtxt(root/(stem+'_pose.txt'))
        points.append(q@pose[:3,:3].T+pose[:3,3])
    points = np.concatenate(points).astype(np.float32)
    audit = dict(session=session, raw_static_probes=len(points),
                 description='Raw RGBD camera-to-world backprojection; exclude person and cabinet; old reference transformed from its local session frame to A frame before comparison.')
    for name, path in paths.items():
        mesh = o3d.io.read_triangle_mesh(str(path))
        if name == 'old': mesh.transform(align)
        ds = distances(points, mesh)
        audit[name] = dict(median_m=float(np.median(ds)), p90_m=float(np.quantile(ds,.9)),
            within_5cm=float(np.mean(ds<=.05)), within_10cm=float(np.mean(ds<=.10)),
            vertices=len(mesh.vertices), triangles=len(mesh.triangles))
    (OUT / (session+'_reference_coordinate_audit.json')).write_text(json.dumps(audit, indent=2))
    print('REFERENCE AUDIT', json.dumps(audit), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', choices=list('ABC'), required=True)
    parser.add_argument('--method')
    parser.add_argument('--mesh', type=Path)
    parser.add_argument('--audit-reference', action='store_true')
    args = parser.parse_args()
    if args.audit_reference: audit_reference(args.session)
    if args.method: score(args.session, args.method, args.mesh)
