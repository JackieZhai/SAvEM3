# -*- coding: utf-8 -*-

from cloudvolume import CloudVolume
import numpy as np
import kimimaro
import time
import os
from collections import defaultdict


def get_data_by_seg_id(cloud_path, seg_id, target_mip=0, padding=5):
    path = cloud_path if cloud_path.startswith('file://') else 'file://' + cloud_path

    
    vol = CloudVolume(path, mip=target_mip, parallel=False, fill_missing=True)

    try:
        
        mesh_data = vol.mesh.get(seg_id)[seg_id]
        vertices = mesh_data.vertices
        min_nm = np.min(vertices, axis=0)
        max_nm = np.max(vertices, axis=0)

        
        resolution = np.array(vol.resolution)
        min_vox = np.floor(min_nm / resolution).astype(int)
        max_vox = np.ceil(max_nm / resolution).astype(int)

        
        x1, y1, z1 = np.maximum(min_vox - padding, 0)
        vol_size = vol.bounds.maxpt
        x2, y2, z2 = np.minimum(max_vox + padding, vol_size)

        
        data = vol[x1:x2, y1:y2, z1:z2]
        data = np.array(data).squeeze()

        # ====================================================
        
        # ====================================================
        
        
        voxel_count = np.sum(data == seg_id)

        
        return data, (x1, y1, z1), resolution, voxel_count

    except Exception as e:
        print(f"Data retrieval failed: {e}")
        
        return None, None, None, 0


def skeletonize_segment(data, target_id, resolution, offset_vox):
    mask = (data == target_id)

    if not np.any(mask):
        print("The data block does not contain the target ID; skeletonization skipped.")
        return None

    skels = kimimaro.skeletonize(
        mask,
        teasar_params={
            'scale': 4,
            'const': 500,  
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100,
            'soma_acceptance_threshold': 3500,
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 300,
            'max_paths': None
        },
        
        
        anisotropy=resolution,
        fix_branching=True,
        progress=True
    )

    if not skels:
        print("No skeleton was generated.")
        return None

    skel = skels[list(skels.keys())[0]]

    
    offset_nm = np.array(offset_vox) * np.array(resolution)
    skel.vertices = skel.vertices + offset_nm

    return skel


def get_endpoints_and_vectors(skel, resolution, lookback_dist_nm=600):

    vertices = skel.vertices  # (N, 3) nm
    edges = skel.edges  # (M, 2) indices

    
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    
    
    tip_indices = [k for k, v in adj.items() if len(v) == 1]

    count = len(tip_indices)
    if count == 0:
        print("Warning: no endpoints were found.")
        return np.array([]), np.array([])

    tips_vox_list = []
    tips_vec_list = []

    resolution = np.array(resolution)

    
    for tip_idx in tip_indices:
        
        tip_pos_nm = vertices[tip_idx]

        
        
        curr = tip_idx
        prev = -1  

        path_dist = 0.0
        inner_pos_nm = tip_pos_nm  

        
        
        max_steps = 10
        steps = 0

        while path_dist < lookback_dist_nm and steps < max_steps:
            neighbors = adj[curr]

            
            next_node = -1
            for n in neighbors:
                if n != prev:
                    next_node = n
                    break

            if next_node == -1:
                
                break

            
            curr_pos = vertices[curr]
            next_pos = vertices[next_node]
            dist = np.linalg.norm(curr_pos - next_pos)
            path_dist += dist

            
            prev = curr
            curr = next_node
            inner_pos_nm = next_pos  
            steps += 1

            
            
            if len(adj[curr]) > 2:
                break

        
        
        vector = tip_pos_nm - inner_pos_nm

        
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm
        else:
            vector = np.array([0.0, 0.0, 0.0])  

        
        
        tip_pos_vox = tip_pos_nm / resolution

        tips_vox_list.append(tip_pos_vox)
        tips_vec_list.append(vector)

    tips_vox = np.array(tips_vox_list)
    tips_vec = np.array(tips_vec_list)

    return tips_vox, tips_vec


def is_messy_segment(endpoints, resolution,
                     max_endpoints=60,
                     density_threshold=1.3,
                     aspect_ratio_threshold=1.5):
    num_eps = len(endpoints)

    
    
    if num_eps < 3:
        return False

    
    if num_eps > max_endpoints:
        print(f"Pruning segment: too many endpoints ({num_eps} > {max_endpoints})")
        return True

    
    
    # pts = np.array(endpoints)
    pts = np.array(endpoints) * np.array(resolution)

    
    
    min_pt = np.min(pts, axis=0)
    max_pt = np.max(pts, axis=0)
    dims = max_pt - min_pt  # [dx, dy, dz]

    
    sorted_dims = np.sort(dims)
    max_dim = sorted_dims[2]  
    mid_dim = sorted_dims[1]
    min_dim = sorted_dims[0]

    
    if max_dim == 0: return True

    
    
    
    
    density = num_eps / max_dim * 1000
    if density > density_threshold:
        return True

    
    
    aspect_ratio = max_dim / (mid_dim + 1e-5)

    
    
    if num_eps > 15 and aspect_ratio < aspect_ratio_threshold:
        
        return True

    return False


def get_endpoints_vectors(target_id, target_mip, seg_path):
    seg_data, offset, resolution, voxel_count = get_data_by_seg_id(seg_path, target_id, target_mip=target_mip,
                                                                   padding=2)
    if seg_data is not None:
        skeleton = skeletonize_segment(seg_data, target_id, resolution, offset)

        endpoints_list, vector_list = get_endpoints_and_vectors(skeleton, resolution)
        return endpoints_list, vector_list, resolution, voxel_count
    return None, None, None, 0


def get_endpoints_vectors_precomputed(target_id, target_mip, seg_path):
    
    if not any(seg_path.startswith(p) for p in ['file://', 'precomputed://', 'gs://', 's3://', 'http://', 'https://']):
        path = 'file://' + seg_path
    else:
        path = seg_path
    
    try:
        
        vol = CloudVolume(path, mip=target_mip, parallel=False, fill_missing=True)
        
        
        skel = vol.skeleton.get(target_id)
        
        if skel is None:
            
            return None, None, None, 0

        resolution = np.array(vol.resolution)
        
        
        
        endpoints_list, vector_list = get_endpoints_and_vectors(skel, resolution)
        
        
        
        
        
        # seg_data, offset, res, voxel_count = get_data_by_seg_id(seg_path, target_id, target_mip=target_mip, padding=2)
        
        voxel_count = 0 
        # if voxel_count is None: voxel_count = 0

        return endpoints_list, vector_list, resolution, voxel_count

    except Exception as e:
        print(f"Failed to read precomputed skeleton (ID={target_id}): {e}")
        return None, None, None, 0


if __name__ == "__main__":
    raw_path = '/path/to/raw/precomputed'
    seg_path = '/path/to/segmentation/precomputed'

    target_id = 123456
    target_mip = 2

    # ----------------

    start_time = time.time()

    endpoints_list, vector_list, resolution, voxel_count = get_endpoints_vectors(123456, target_mip, seg_path)
    is_messy_segment(endpoints_list, resolution)

    endpoints_list, vector_list, resolution, voxel_count = get_endpoints_vectors(234567, target_mip, seg_path)
    is_messy_segment(endpoints_list, resolution)

    endpoints_list, vector_list, resolution, voxel_count = get_endpoints_vectors(345678, target_mip, seg_path)
    is_messy_segment(endpoints_list, resolution)

    endpoints_list, vector_list, resolution, voxel_count = get_endpoints_vectors(456789, target_mip, seg_path)
    is_messy_segment(endpoints_list, resolution)

    end_time = time.time()
    print("Runtime: {:.4f} seconds".format(end_time - start_time))
