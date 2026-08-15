import numpy as np
import math
from numba import jit
from numba.typed import Dict
from numba.core import types
from concurrent.futures import ThreadPoolExecutor
from cloudvolume import CloudVolume
import time
import csv


@jit(nopython=True)
def traverse_local_block_xyz(segmentation, center_local, vector, resolution, target_label, max_dist_nm):
    
    res_x, res_y, res_z = resolution
    x_dim, y_dim, z_dim = segmentation.shape
    cx, cy, cz = center_local

    
    x_rad = int(max_dist_nm / res_x) + 1
    y_rad = int(max_dist_nm / res_y) + 1
    z_rad = int(max_dist_nm / res_z) + 1

    
    # cos_theta_threshold = 0.94
    # cos_theta_threshold = 0.70
    # cos_theta_threshold = 0.50
    # cos_theta_threshold = 0.0
    cos_theta_threshold = -2

    
    # Key: Label ID (int64)
    
    stats = Dict.empty(
        key_type=types.uint64,
        value_type=types.float64[:]
    )

    
    x_min = max(0, cx - x_rad)
    x_max = min(x_dim, cx + x_rad + 1)
    y_min = max(0, cy - y_rad)
    y_max = min(y_dim, cy + y_rad + 1)
    z_min = max(0, cz - z_rad)
    z_max = min(z_dim, cz + z_rad + 1)

    for ix in range(x_min, x_max):
        for iy in range(y_min, y_max):
            for iz in range(z_min, z_max):

                neighbor_label = segmentation[ix, iy, iz]

                
                if neighbor_label == 0 or neighbor_label == target_label:
                    continue

                
                dx = (ix - cx) * res_x
                dy = (iy - cy) * res_y
                dz = (iz - cz) * res_z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist > max_dist_nm: continue

                
                if dist > 0:
                    
                    dot = vector[0] * (dx / dist) + vector[1] * (dy / dist) + vector[2] * (dz / dist)
                    if dot < cos_theta_threshold: continue

                
                is_touching = False

                
                if ix > 0 and segmentation[ix - 1, iy, iz] == target_label:
                    is_touching = True
                elif ix < x_dim - 1 and segmentation[ix + 1, iy, iz] == target_label:
                    is_touching = True
                
                elif iy > 0 and segmentation[ix, iy - 1, iz] == target_label:
                    is_touching = True
                elif iy < y_dim - 1 and segmentation[ix, iy + 1, iz] == target_label:
                    is_touching = True
                
                elif iz > 0 and segmentation[ix, iy, iz - 1] == target_label:
                    is_touching = True
                elif iz < z_dim - 1 and segmentation[ix, iy, iz + 1] == target_label:
                    is_touching = True

                
                if is_touching:
                    if neighbor_label not in stats:
                        # [SumX, SumY, SumZ, Count]
                        stats[neighbor_label] = np.zeros(4, dtype=np.float64)

                    val = stats[neighbor_label]
                    val[0] += ix
                    val[1] += iy
                    val[2] += iz
                    val[3] += 1.0

    return stats


# ==========================================

# ==========================================
def process_single_endpoint_xyz(vol, endpoint_vox_xyz, vector_xyz, target_id, endpoint_id, search_dist_nm=500):
    
    res_xyz = np.array(vol.resolution)

    
    padding_vox = (search_dist_nm / res_xyz).astype(int) + 2

    
    center = np.round(endpoint_vox_xyz).astype(int)

    
    vol_size = vol.bounds.maxpt  
    start = np.maximum(center - padding_vox, 0)
    end = np.minimum(center + padding_vox + 1, vol_size)

    try:
        
        # vol[x_start:x_end, y_start:y_end, z_start:z_end]
        cutout = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        
        
        
        # cutout = np.array(cutout).squeeze()
        cutout = np.array(cutout).squeeze()

        if cutout.ndim != 3:
            return []

        
        center_local = center - start

        
        stats = traverse_local_block_xyz(
            cutout,
            center_local,  # (cx, cy, cz)
            vector_xyz,  # (vx, vy, vz)
            res_xyz,  # (rx, ry, rz)
            target_id,
            search_dist_nm
        )

        
        results = []
        for label, arr in stats.items():
            count = arr[3]
            
            mean_x_local = arr[0] / count
            mean_y_local = arr[1] / count
            mean_z_local = arr[2] / count

            
            global_x = mean_x_local + start[0]
            global_y = mean_y_local + start[1]
            global_z = mean_z_local + start[2]

            results.append({
                'target_id': target_id,  
                'neighbor_id': label,
                'x': global_x,
                'y': global_y,
                'z': global_z,
                'endpoint_id': endpoint_id,
                'voxel_count': int(count)
            })

        return results

    except Exception as e:
        print(f"Error processing endpoint {endpoint_vox_xyz}: {e}")
        return []


def save_connections_to_csv(connections, filename):
    if not connections:
        print("No connections found; skipping CSV export.")
        return

    print(f"Saving {len(connections)} connections to {filename} ...")

    
    headers = ['target_id', 'neighbor_id', 'contact_x', 'contact_y', 'contact_z']

    
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        
        writer.writerow(headers)

        
        for conn in connections:
            writer.writerow([
                conn['target_id'],
                conn['neighbor_id'],
                int(conn['x']),  
                int(conn['y']),
                int(conn['z'])
            ])

    print(f"Saved: {filename}")


def evaluate_performance(ans_list, connections):
    
    predicted_ids = set()
    for conn in connections:
        predicted_ids.add(conn['neighbor_id'])

    
    ground_truth = set(ans_list)

    
    matched = ground_truth.intersection(predicted_ids)
    missed = ground_truth - predicted_ids

    
    score = (len(matched) / len(ground_truth)) * 100.0

    
    print("-" * 40)
    print(f"Matched IDs  : {sorted(list(matched))} (Count: {len(matched)})")
    print(f"Missed IDs   : {sorted(list(missed))} (Count: {len(missed)})")
    print("-" * 40)
    print(f"Recall Score : {score:.2f}%")
    print("=" * 40 + "\n")

    return score


# ==========================================

# ==========================================
def find_neighbors(cloud_path, target_id, endpoints_xyz, vectors_xyz, skel_resolution_xyz, max_workers=8):

    
    if not any(cloud_path.startswith(p) for p in ['file://', 'precomputed://', 'gs://', 'https://']):
        path = 'file://' + cloud_path
    else:
        path = cloud_path
    
    vol = CloudVolume(path, mip=0, parallel=False, fill_missing=True)

    res0_xyz = np.array(vol.resolution)

    
    # Scale Factor = High_Res / Low_Res
    scale_factor = np.array(skel_resolution_xyz) / res0_xyz
    endpoints_mip0 = endpoints_xyz * scale_factor

    all_connections = []

    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(len(endpoints_mip0)):
            f = executor.submit(
                process_single_endpoint_xyz,
                vol,
                endpoints_mip0[i],  # (x, y, z)
                vectors_xyz[i],  # (vx, vy, vz)
                target_id,
                i
            )
            futures.append(f)

        for i, f in enumerate(futures):
            res = f.result()
            if res:
                # print(f"  [Endpoint {i}] Found {len(res)} neighbors: {[r['neighbor_id'] for r in res]}")
                all_connections.extend(res)

    return all_connections


def get_neighbors(seg_path, target_id, endpoints_list, vectors_list, ske_resolution):
    connection = find_neighbors(seg_path, target_id, endpoints_list, vectors_list, ske_resolution)
    return connection


if __name__ == "__main__":
    ans = [234567, 345678, 456789]
    cloud_path = '/path/to/segmentation/precomputed'
    target_id = 123456

    data = np.load('example_endpoints_vectors.npz')
    endpoints_list = data['endpoints']
    vector_list = data['vector']
    # print(endpoints_list)
    # print(vector_list)

    start_time = time.time()

    connection = find_neighbors(cloud_path, target_id, endpoints_list, vector_list, [40, 40, 40])

    save_connections_to_csv(connection, 'connections.csv')

    evaluate_performance(ans, connection)

    end_time = time.time()
    print("Runtime: {:.4f} seconds".format(end_time - start_time))
