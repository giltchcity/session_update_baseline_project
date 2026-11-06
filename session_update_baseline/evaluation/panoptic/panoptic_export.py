#!/usr/bin/env python3
"""Read archived Panoptic .panmap files without ROS/Docker or any GT inputs.

Uses the checked-in native protobuf definitions, TSDF serialization and native
MarchingCubes lookup table. Mesh validity follows MeshIntegrator defaults:
all 8 corners weight > 1e-6, >4 belonging corners, no foreign-voxel closure.
This is a transparent fallback exporter; an executable-library parity check is
unavailable when Docker is stopped, and is reported rather than assumed.
"""
import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).parent / 'panoptic_reference_sources'
LABELS = ['Unknown', 'Instance', 'Background', 'FreeSpace']
STATES = ['New', 'Matched', 'Unobserved', 'Absent', 'Persistent']
# Input label contract maps compact native class IDs to the original ADE IDs.
ADE = [0, 3, 5, 7, 8, 10, 12, 14, 15, 24, 33, 35, 74, 75, 92, 115, 131, 139]
CORNERS = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.int64)
EDGES = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],
                  [6,7],[7,4],[0,4],[1,5],[2,6],[3,7]], dtype=np.int64)


def read_varint(data, pos):
    value, shift = 0, 0
    while True:
        b = data[pos]
        pos += 1
        value |= (b & 127) << shift
        if not b & 128:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError('Oversized varint')


def read_message(data, pos):
    size, pos = read_varint(data, pos)
    end = pos + size
    if end > len(data):
        raise ValueError('Truncated protobuf message')
    return memoryview(data)[pos:end], end


def parse_fields(data, stop_voxels=False):
    fields, pos = {}, 0
    while pos < len(data):
        start = pos
        tag, pos = read_varint(data, pos)
        field, wire = tag >> 3, tag & 7
        if stop_voxels and field == 7:
            return fields, start
        if wire == 0:
            value, pos = read_varint(data, pos)
        elif wire == 1:
            value = struct.unpack_from('<d', data, pos)[0]
            pos += 8
        elif wire == 5:
            value = struct.unpack_from('<f', data, pos)[0]
            pos += 4
        elif wire == 2:
            size, pos = read_varint(data, pos)
            value = bytes(data[pos:pos + size])
            pos += size
        else:
            raise ValueError('Unsupported protobuf wire type %s' % wire)
        if field in fields:
            raise ValueError('Unexpected repeated header field')
        fields[field] = value
    return (fields, pos) if stop_voxels else fields


@njit(cache=True)
def decode_words(data):
    result = np.empty(len(data) // 2 + 1, dtype=np.uint32)
    pos, n = 0, 0
    while pos < len(data):
        if data[pos] != 56:  # repeated uint32 voxel_data = 7; non-packed
            raise ValueError('Unexpected voxel-data tag')
        pos += 1
        value, shift = np.uint64(0), np.uint64(0)
        while True:
            b = data[pos]
            pos += 1
            value |= np.uint64(b & 127) << shift
            if not b & 128:
                break
            shift += np.uint64(7)
        result[n] = np.uint32(value)
        n += 1
    return result[:n]


def block_data(message):
    meta, pos = parse_fields(message, stop_voxels=True)
    words = decode_words(np.frombuffer(message[pos:], dtype=np.uint8))
    n, voxel = int(meta[1]), float(meta[2])
    origin = np.array([meta[3], meta[4], meta[5]])
    index = tuple(np.rint(origin / (voxel*n)).astype(int))
    assert np.max(np.abs(origin - np.array(index)*voxel*n)) < 1e-5
    return index, n, voxel, words


def belonging(words, kind, n):
    if kind == 1:
        halves = np.empty(len(words)*2, dtype=np.uint32)
        halves[0::2] = words >> 16
        halves[1::2] = words & 65535
        result = (halves >> 8) >= (halves & 255)
    elif kind == 0:
        result = (words >> 16) >= (words & 65535)
    else:
        raise ValueError('Unsupported class voxel type %s; no silent fallback' % kind)
    assert len(result) == n**3
    return result.reshape((n,n,n), order='F')


def triangle_table():
    source = (SOURCES / 'marching_cubes.cc').read_text()
    section = source.split('kTriangleTable[256][16] = {', 1)[1].split('};', 1)[0]
    rows = re.findall(r'\{([^{}]+)\}', section)
    table = np.array([[int(x) for x in row.split(',')] for row in rows], dtype=np.int64)
    assert table.shape == (256,16)
    return table


def mesh_blocks(tsdf, classes, n, voxel, table):
    vertices = []
    has_class = bool(classes)
    for index, (base_sdf, base_weight) in sorted(tsdf.items()):
        if has_class and index not in classes:
            continue  # Native updateMeshForBlock skips missing class blocks.
        sdf = np.zeros((n+1,)*3, dtype=np.float32)
        weight = np.zeros((n+1,)*3, dtype=np.float32)
        belongs = np.ones((n+1,)*3, dtype=np.uint8)
        for off in CORNERS:
            key = tuple(np.array(index) + off)
            if key not in tsdf:
                continue
            dst = tuple(slice(0,n) if v == 0 else slice(n,n+1) for v in off)
            src = tuple(slice(0,n) if v == 0 else slice(0,1) for v in off)
            sdf[dst] = tsdf[key][0][src]
            weight[dst] = tsdf[key][1][src]
            if has_class and key in classes:
                belongs[dst] = classes[key][src]
        corners = np.stack([sdf[x:x+n,y:y+n,z:z+n] for x,y,z in CORNERS], axis=-1)
        valid = np.logical_and.reduce([weight[x:x+n,y:y+n,z:z+n] > 1e-6 for x,y,z in CORNERS])
        if has_class:
            counts = np.sum([belongs[x:x+n,y:y+n,z:z+n] for x,y,z in CORNERS], axis=0)
            valid &= counts > 4
        configs = np.sum((corners < 0) * (1 << np.arange(8)), axis=-1)
        valid &= (configs != 0) & (configs != 255)
        local = np.argwhere(valid)
        if not len(local):
            continue
        distances = corners[valid]
        edge_indices = table[configs[valid]]
        row, col = np.where(edge_indices >= 0)
        pairs = EDGES[edge_indices[row,col]]
        p0, p1 = CORNERS[pairs[:,0]], CORNERS[pairs[:,1]]
        d0, d1 = distances[row,pairs[:,0]], distances[row,pairs[:,1]]
        t = np.full(len(d0), .5, dtype=np.float32)
        np.divide(d0,d0-d1,out=t,where=np.abs(d0-d1)>=1e-6)
        xyz = local[row] + np.array(index)*n + .5 + p0 + t[:,None]*(p1-p0)
        vertices.append((xyz*voxel).astype(np.float32))
    return np.concatenate(vertices) if vertices else np.empty((0,3),np.float32)


def transform_matrix(meta):
    f = parse_fields(meta[5])
    t = np.array([f.get(i,0.) for i in (1,2,3)])
    w,x,y,z = [f.get(i,0.) for i in (4,5,6,7)]
    assert abs(w*w+x*x+y*y+z*z-1) < 1e-4
    R = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                  [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                  [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    T = np.eye(4)
    T[:3,:3], T[:3,3] = R,t
    return T


def write_ply(path, vertices):
    vertices = np.asarray(vertices, dtype='<f4')
    assert len(vertices) % 3 == 0
    faces = np.empty(len(vertices)//3, dtype=[('n','u1'),('idx','<i4',(3,))])
    faces['n'] = 3
    faces['idx'] = np.arange(len(vertices),dtype=np.int32).reshape((-1,3))
    with path.open('wb') as out:
        out.write(('ply\nformat binary_little_endian 1.0\nelement vertex %s\n'
                   'property float x\nproperty float y\nproperty float z\n'
                   'element face %s\nproperty list uchar int vertex_indices\nend_header\n'
                   % (len(vertices),len(faces))).encode())
        out.write(vertices.tobytes())
        out.write(faces.tobytes())


def sample_surface(vertices, spacing, seed):
    tri = vertices.reshape((-1,3,3)).astype(np.float64)
    if not len(tri):
        return np.empty((0,3),np.float32), np.zeros(3), 0.
    areas = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
    total = areas.sum()
    if total <= 0:
        return np.empty((0,3),np.float32), vertices.mean(axis=0), 0.
    centroid = (tri.mean(axis=1)*areas[:,None]).sum(axis=0)/total
    count = max(1,int(np.ceil(total/spacing**2)))
    rng = np.random.RandomState(seed)
    chosen = tri[np.searchsorted(np.cumsum(areas),rng.uniform(0,total,count))]
    r = rng.uniform(size=(count,2))
    s = np.sqrt(r[:,0])
    samples = (1-s[:,None])*chosen[:,0] + (s*(1-r[:,1]))[:,None]*chosen[:,1] + (s*r[:,1])[:,None]*chosen[:,2]
    return samples.astype(np.float32), centroid, float(total)


def export(path, output, skip_class_filter=False):
    output.mkdir(parents=True, exist_ok=True)
    (output/'submaps').mkdir(exist_ok=True)
    data = path.read_bytes()
    first,pos = read_message(data,0)
    collection = parse_fields(first)
    table = triangle_table()
    rows, combined, samples_combined, point_ids = [], {'non_absent':[], 'confirmed':[]}, [], []
    for ordinal in range(collection[1]):
        header,pos = read_message(data,pos)
        meta = parse_fields(header)
        for field in (1,6,7,11,13,15):
            if field in meta and meta[field] >= 2**31:
                meta[field] = (meta[field] % 2**32) - 2**32
        tsdf, classes = {}, {}
        n,voxel = meta[4],meta[3]
        for _ in range(meta[2]):
            message,pos = read_message(data,pos)
            index,bn,bvoxel,words = block_data(message)
            assert bn == n and abs(bvoxel-voxel)<1e-8 and len(words)==n**3*3
            floats=words.reshape((-1,3))[:,:2].copy().view(np.float32)
            tsdf[index]=(floats[:,0].reshape((n,n,n),order='F'),floats[:,1].reshape((n,n,n),order='F'))
        for _ in range(meta.get(12,0)):
            message,pos = read_message(data,pos)
            index,bn,bvoxel,words = block_data(message)
            assert bn == n and abs(bvoxel-voxel)<1e-8
            classes[index]=belonging(words,meta[13],n)
        for _ in range(meta.get(14,0)):
            _,pos = read_message(data,pos)
        T = transform_matrix(meta)
        local = mesh_blocks(tsdf,{} if skip_class_filter else classes,n,voxel,table) if meta[7] != 3 else np.empty((0,3),np.float32)
        vertices = (local @ T[:3,:3].T + T[:3,3]).astype(np.float32)
        frame = meta.get(10,b'').decode()
        # getID() is not serialized. frame_name preserves the native original ID.
        original_id = int(frame.split('_')[-1]) if re.fullmatch(r'submap_\d+',frame) else ordinal
        name = meta.get(9,b'').decode()
        state = int(meta.get(11,0))
        mesh_path=output/'submaps'/('%04d_%s.ply'%(ordinal,frame))
        write_ply(mesh_path,vertices)
        sample,area_centroid,area=sample_surface(vertices,.02,17000+ordinal)
        sample_path=mesh_path.with_suffix('.npz')
        np.savez_compressed(sample_path,points=sample)
        row={'id':original_id,'load_ordinal':ordinal,'frame_name':frame,'instance_id':meta[1],
             'class_id':meta[6],'semantic':ADE[meta[6]] if 0<=meta[6]<len(ADE) else None,'panoptic_label_id':meta[7],
             'panoptic_label':LABELS[meta[7]],'name':name,'change_state_id':state,
             'change_state':STATES[state],'is_active':None,
             'is_active_status':'not_serialized_in_SubmapProto',
             'T_M_S':T.tolist(),'centroid':vertices.mean(axis=0,dtype=np.float64).tolist() if len(vertices) else None,
             'area_centroid':area_centroid.tolist(),'surface_area_m2':area,
             'vertex_count':len(vertices),'triangle_count':len(vertices)//3,
             'sample_count':len(sample),'mesh_path':str(mesh_path.resolve()),'surface_points_path':str(sample_path.resolve()),
             # These are adapter selection policies, not native existence states.
             # Preserve the five-valued native change_state separately above.
             'included_in_current_non_absent':state!=3 and meta[7]!=3 and len(vertices)>0,
             'included_in_confirmed_current':state in (0,1,4) and meta[7]!=3 and len(vertices)>0,
             'voxel_size_m':voxel,'num_blocks':meta[2],'num_class_blocks':meta.get(12,0),
             'class_voxel_type':meta.get(13)}
        rows.append(row)
        for variant,predicate in [('non_absent',row['included_in_current_non_absent']),('confirmed',row['included_in_confirmed_current'])]:
            if predicate:
                combined[variant].append(vertices)
        if row['included_in_current_non_absent']:
            samples_combined.append(sample)
            point_ids.append(np.full(len(sample),original_id,np.int32))
        print('%s %d/%d %-12s %-10s class=%d vertices=%d area=%.3f'%(path.stem,ordinal+1,collection[1],name,STATES[state],meta[6],len(vertices),area),flush=True)
    assert pos==len(data),'Unread bytes after last submap'
    for variant,meshes in combined.items():
        write_ply(output/('current_%s.ply'%variant),np.concatenate(meshes) if meshes else np.empty((0,3),np.float32))
    np.savez_compressed(output/'current_non_absent_surface.npz',points=np.concatenate(samples_combined),submap_id=np.concatenate(point_ids))
    manifest={'source':str(path.resolve()),'source_sha256':hashlib.sha256(data).hexdigest(),
              'world_frame':'Panoptic mission/world; input B and C were aligned into A',
              'active_freespace_submap_id':collection.get(2),'native_num_submaps':collection[1],
              'protocol':{'schema':'panoptic_native_state_and_current_geometry/v2',
                          'object_unit':'native kInstance submap; no GT grouping',
                          'current_non_absent':'all non-free-space, non-empty mesh with state != Absent; retains Unobserved',
                          'current_confirmed':'New/Matched/Persistent only',
                          'selection_flags':'included_in_* are evaluator selection policies, not native presence booleans',
                          'historical_key_aliases':{'present':'included_in_current_non_absent','confirmed_present':'included_in_confirmed_current'},
                          'display_scope':'current_non_absent matches export_current_mesh.sh selection, not all RViz-visible geometry; RViz active/persistent modes may display inactive/Absent/history submaps with nonzero alpha',
                          'meshing':'native classic MC lookup; all8 weights >1e-6; '+('class-layer filtering disabled for legacy reload diagnostic' if skip_class_filter else '>4 class-belonging corners; clear_foreign_voxels=false'),
                          'mesh_backend':'independent source-matched decoder; native library parity pending unavailable Docker',
                          'class_filter_enabled':not skip_class_filter,
                          'centroid':'mean native triangle vertices; area_centroid supplied separately',
                          'sampling':'uniform area, expected one point per 0.02m squared, seeded',
                          'class_mapping_ADE':ADE,
                          'isActive':'not serialized; deliberately null, not loaded default true'},
              'counts_by_state':dict(Counter(r['change_state'] for r in rows)),
              'objects':[r for r in rows if r['panoptic_label_id']==1], 'submaps':rows}
    (output/'objects.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--session',choices=['a','b','c','all'],default='all')
    parser.add_argument('--input-dir',type=Path,default=ROOT/'runs/panoptic_mapping/abc_live_20260909_now')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/abc_eval_v2/panoptic')
    parser.add_argument('--skip-class-filter',action='store_true',help='Explicit diagnostic: reproduce old reloads whose mesh integrator was missing its class-layer pointer')
    args=parser.parse_args()
    for session in ('abc' if args.session=='all' else args.session):
        export(args.input_dir/('session_%s.panmap'%session),args.output_dir/session,args.skip_class_filter)


if __name__=='__main__':
    main()
