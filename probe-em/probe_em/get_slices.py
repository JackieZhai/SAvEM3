import numpy as np
import pandas as pd
import cv2
import os
from cloudvolume import CloudVolume
from scipy.ndimage import label, find_objects
import time
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import shutil


# ==========================================

# ==========================================
def get_largest_connected_component_2d(mask):
    if not np.any(mask): return mask
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(mask, structure=structure)
    if num_features == 0: return mask
    counts = np.bincount(labeled_array.ravel())
    largest_label = counts[1:].argmax() + 1
    return (labeled_array == largest_label)


def get_center_connected_component_2d(mask):
    if not np.any(mask): return mask

    
    structure = np.ones((3, 3), dtype=int)
    labeled_array, num_features = label(mask, structure=structure)

    if num_features == 0: return mask
    if num_features == 1: return mask.astype(bool)  

    
    h, w = mask.shape
    center_y, center_x = h / 2, w / 2

    
    min_dist_sq = float('inf')
    best_label = -1

    
    slices = find_objects(labeled_array)

    for i, sl in enumerate(slices):
        if sl is None: continue
        label_id = i + 1

        
        y_slice, x_slice = sl
        obj_cy = (y_slice.start + y_slice.stop) / 2
        obj_cx = (x_slice.start + x_slice.stop) / 2

        
        # y_coords, x_coords = np.where(labeled_array == label_id)
        # obj_cy = np.mean(y_coords)
        # obj_cx = np.mean(x_coords)

        
        dist_sq = (obj_cy - center_y) ** 2 + (obj_cx - center_x) ** 2

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_label = label_id

    
    return (labeled_array == best_label)


def check_boundaries(seg_2d, ids_to_check):
    h, w = seg_2d.shape
    edges = np.concatenate([seg_2d[0, :], seg_2d[h - 1, :], seg_2d[:, 0], seg_2d[:, w - 1]])
    for target_id in ids_to_check:
        if target_id in edges: return True
    return False


def crop_tight_2d(img, mask, padding=10):
    if not np.any(mask): return img, mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    h, w = img.shape
    y_min = max(0, y_min - padding);
    y_max = min(h, y_max + padding + 1)
    x_min = max(0, x_min - padding);
    x_max = min(w, x_max + padding + 1)
    return img[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


# ==========================================

# ==========================================
def prepare_sam2_video_simple(seg_vol, raw_vol, center, target_id, neighbor_id, r_vox, output_dir, base_name):
    cx, cy, cz = center

    
    def check_layer_ids(z_idx):
        
        if z_idx < 0 or z_idx >= seg_vol.bounds.maxpt[2]:
            return False, False

        start = np.maximum(center - r_vox, 0)
        end = np.minimum(center + r_vox + 1, seg_vol.bounds.maxpt)
        start[2], end[2] = z_idx, z_idx + 1

        
        seg = np.array(seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        return (target_id in seg), (neighbor_id in seg)

    
    has_t_z0, has_n_z0 = check_layer_ids(cz)

    start_z = -1
    step = 0  # 1: down (Z+), -1: up (Z-)

    
    if has_t_z0 and not has_n_z0:
        start_z = cz  
        _, has_n_down = check_layer_ids(cz + 1)

        if has_n_down:
            step = 1  
        else:
            step = -1  

    
    elif has_n_z0 and not has_t_z0:
        has_t_down, _ = check_layer_ids(cz + 1)
        has_t_up, _ = check_layer_ids(cz - 1)

        if has_t_down:
            start_z = cz + 1  
            step = -1  
            
        elif has_t_up:
            start_z = cz - 1  
            step = 1  
            
        else:
            return {'status': 'failed', 'msg': "Neighbor at Z0, but Target not found in Z+/-1"}

    
    else:
        
        
        return {'status': 'failed', 'msg': f"Z0 status unclear: T={has_t_z0}, N={has_n_z0}"}

    
    num_frames = 2
    z_sequence = [start_z + i * step for i in range(num_frames)]

    
    instance_dir = os.path.join(output_dir, base_name)
    frames_dir = os.path.join(instance_dir, "video_frames")

    if os.path.exists(instance_dir):
        shutil.rmtree(instance_dir)
    os.makedirs(frames_dir, exist_ok=True)

    
    
    start_xy = np.maximum(center - r_vox, 0)
    end_xy = np.minimum(center + r_vox + 1, seg_vol.bounds.maxpt)

    neighbor_frame_idx = -1  

    for i, z_curr in enumerate(z_sequence):
        if z_curr < 0 or z_curr >= seg_vol.bounds.maxpt[2]: continue

        start = start_xy.copy();
        end = end_xy.copy()
        start[2], end[2] = z_curr, z_curr + 1

        
        raw_img = np.array(raw_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        if raw_img.ndim == 2: raw_img = raw_img.transpose(1, 0)  # xy -> yx
        if raw_img.dtype != np.uint8:
            raw_img = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR))

        
        seg_img = np.array(seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        if seg_img.ndim == 2: seg_img = seg_img.transpose(1, 0)

        
        if i == 0:
            mask_t_bool = (seg_img == target_id)
            mask_t_filtered = get_largest_connected_component_2d(mask_t_bool)
            mask_t_save = mask_t_filtered.astype(np.uint8) * 255

            cv2.imwrite(os.path.join(instance_dir, "prompt_mask_00000.png"), mask_t_save)
            continue

        
        
        if neighbor_frame_idx == -1 and neighbor_id in seg_img:
            neighbor_frame_idx = i
            mask_n_bool = (seg_img == neighbor_id)
            mask_n_filtered = get_largest_connected_component_2d(mask_n_bool)
            mask_n_save = mask_n_filtered.astype(np.uint8) * 255

            cv2.imwrite(os.path.join(instance_dir, f"neighbor_mask_{i:05d}.png"), mask_n_save)

    
    with open(os.path.join(instance_dir, "info.txt"), "w") as f:
        f.write(f"target_id: {target_id}\nneighbor_id: {neighbor_id}\n")
        f.write(f"target_frame: 0\nneighbor_frame: {neighbor_frame_idx}\n")
        f.write(f"z_start: {start_z}\nz_step: {step}\n")

    return {'status': 'saved', 'path': instance_dir}


def prepare_sam2_video_region(seg_vol, raw_vol, center, target_id, neighbor_id, r_vox, output_dir, base_name, num_frames=5):
    cx, cy, cz = center

    
    def check_layer_ids(z_idx):
        if z_idx < 0 or z_idx >= seg_vol.bounds.maxpt[2]: return False, False
        start = np.maximum(center - r_vox, 0)
        end = np.minimum(center + r_vox + 1, seg_vol.bounds.maxpt)
        start[2], end[2] = z_idx, z_idx + 1
        seg = np.array(seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        return (target_id in seg), (neighbor_id in seg)

    
    has_t_z0, has_n_z0 = check_layer_ids(cz)
    start_z = -1
    step = 0

    if has_t_z0 and not has_n_z0:
        start_z = cz
        _, has_n_down = check_layer_ids(cz + 1)
        step = 1 if has_n_down else -1

    elif has_n_z0 and not has_t_z0:
        has_t_down, _ = check_layer_ids(cz + 1)
        has_t_up, _ = check_layer_ids(cz - 1)
        if has_t_down:
            start_z = cz + 1;
            step = -1
        elif has_t_up:
            start_z = cz - 1;
            step = 1
        else:
            return {'status': 'failed', 'msg': "Neighbor at Z0, but Target not found in Z+/-1"}
    else:
        return {'status': 'failed', 'msg': f"Z0 status unclear"}

    
    # num_frames = 5
    z_sequence = [start_z + i * step for i in range(num_frames)]

    
    instance_dir = os.path.join(output_dir, base_name)
    video_dir = os.path.join(instance_dir, "video_frames")
    seg_dir = os.path.join(instance_dir, "segmentation_data")  

    if os.path.exists(instance_dir): shutil.rmtree(instance_dir)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)

    
    start_xy = np.maximum(center - r_vox, 0)
    end_xy = np.minimum(center + r_vox + 1, seg_vol.bounds.maxpt)

    neighbor_frame_idx = -1

    for i, z_curr in enumerate(z_sequence):
        if z_curr < 0 or z_curr >= seg_vol.bounds.maxpt[2]: continue

        start = start_xy.copy();
        end = end_xy.copy()
        start[2], end[2] = z_curr, z_curr + 1

        
        raw_img = np.array(raw_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        if raw_img.ndim == 2: raw_img = raw_img.transpose(1, 0)

        if raw_img.dtype != np.uint8:
            raw_vis = cv2.normalize(raw_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            raw_vis = raw_img
        cv2.imwrite(os.path.join(video_dir, f"{i:05d}.jpg"), cv2.cvtColor(raw_vis, cv2.COLOR_GRAY2BGR))

        
        
        seg_img = np.array(seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]).squeeze()
        if seg_img.ndim == 2: seg_img = seg_img.transpose(1, 0)

        
        np.save(os.path.join(seg_dir, f"{i:05d}_seg.npy"), seg_img)

        
        if i == 0:
            mask_t_bool = (seg_img == target_id)
            mask_t_filtered = get_largest_connected_component_2d(mask_t_bool)
            mask_t_save = mask_t_filtered.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(instance_dir, "prompt_mask_00000.png"), mask_t_save)

        
        if neighbor_frame_idx == -1 and neighbor_id in seg_img:
            neighbor_frame_idx = i

    
    with open(os.path.join(instance_dir, "info.txt"), "w") as f:
        f.write(f"target_id: {target_id}\nneighbor_id: {neighbor_id}\n")
        f.write(f"neighbor_frame: {neighbor_frame_idx}\n")
        f.write(f"z_start: {start_z}\nz_step: {step}\n")

    return {'status': 'saved', 'path': instance_dir}



def process_single_connection(seg_vol, raw_vol, conn, output_dir, output_dir3d,
                              initial_radius_nm=4000,
                              max_radius_nm=32100,
                              expansion_factor=2.0,
                              num_frames=5):
    try:
        target_id = int(conn['target_id'])
        neighbor_id = int(conn['neighbor_id'])
        
        cx = int(conn.get('contact_x', conn.get('x')))
        cy = int(conn.get('contact_y', conn.get('y')))
        cz = int(conn.get('contact_z', conn.get('z')))

        
        endpoint_id = int(conn.get('endpoint_id', 0))

        base_name = f"{target_id}_{neighbor_id}_{endpoint_id}"

        
        if os.path.exists(os.path.join(output_dir, f"{base_name}_xy_overlay.jpg")):
            return {'status': 'skipped', 'msg': f"Skipped {base_name}"}

        res = np.array(seg_vol.resolution)
        current_radius_nm = initial_radius_nm

        final_seg_2d = None
        final_box = None

        
        while True:
            r_vox = (current_radius_nm / res).astype(int)
            center = np.array([cx, cy, cz])

            
            raw_start = center - r_vox
            raw_end = center + r_vox + 1

            
            vol_max = seg_vol.bounds.maxpt
            start = np.maximum(raw_start, 0)
            end = np.minimum(raw_end, vol_max)

            
            start[2] = cz
            end[2] = cz + 1

            
            seg_cutout = seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            seg_2d = np.array(seg_cutout).squeeze()
            if seg_2d.ndim == 2: seg_2d = seg_2d.T

            final_seg_2d = seg_2d
            final_box = (start, end)

            
            touching_min = np.any(raw_start[:2] < 0)  
            touching_max = np.any(raw_end[:2] > vol_max[:2])  
            if touching_min or touching_max:
                break
            
            
            if current_radius_nm >= max_radius_nm: break
            if check_boundaries(seg_2d, [target_id, neighbor_id]):
                current_radius_nm *= expansion_factor
                continue
            else:
                break

        
        mask_t = (final_seg_2d == target_id)
        mask_n = (final_seg_2d == neighbor_id)

        
        # mask_t = get_largest_connected_component_2d(mask_t)
        # mask_n = get_largest_connected_component_2d(mask_n)
        mask_t = get_center_connected_component_2d(mask_t)
        mask_n = get_center_connected_component_2d(mask_n)

        
        if not np.any(mask_t) or not np.any(mask_n):
            
            r_vox_final = (current_radius_nm / res).astype(int)

            # video_res = prepare_sam2_video_simple(
            #     seg_vol, raw_vol, center, target_id, neighbor_id, r_vox_final,
            #     output_dir3d, base_name
            # )
            video_res = prepare_sam2_video_region(
                seg_vol, raw_vol, center, target_id, neighbor_id, r_vox_final,
                output_dir3d, base_name, num_frames=num_frames
            )

            return {'status': 'z_gap', 'conn': conn, 'msg': f"Z-Gap: {video_res.get('msg', 'Saved video sequence')}"}

        
        start, end = final_box
        raw_cutout = raw_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        raw_2d = np.array(raw_cutout).squeeze()
        if raw_2d.ndim == 2: raw_2d = raw_2d.T

        combined_mask = np.logical_or(mask_t, mask_n)
        raw_crop, _ = crop_tight_2d(raw_2d, combined_mask, padding=20)

        
        rows = np.any(combined_mask, axis=1);
        cols = np.any(combined_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]];
        x_min, x_max = np.where(cols)[0][[0, -1]]
        pad = 20;
        h, w = raw_2d.shape
        y_min = max(0, y_min - pad);
        y_max = min(h, y_max + pad + 1)
        x_min = max(0, x_min - pad);
        x_max = min(w, x_max + pad + 1)

        mask_t_crop = mask_t[y_min:y_max, x_min:x_max]
        mask_n_crop = mask_n[y_min:y_max, x_min:x_max]

        final_mask = np.zeros_like(mask_t_crop, dtype=np.uint8)
        final_mask[mask_t_crop] = 1
        final_mask[mask_n_crop] = 2

        
        max_size = 512
        h_c, w_c = final_mask.shape
        if max(h_c, w_c) > max_size:
            scale = max_size / max(h_c, w_c)
            final_mask = cv2.resize(final_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            raw_crop = cv2.resize(raw_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if raw_crop.dtype != np.uint8:
            raw_crop = cv2.normalize(raw_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir, f"{base_name}_xy_raw.jpg"), raw_crop)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_xy_mask.png"), final_mask)

        vis_img = cv2.cvtColor(raw_crop, cv2.COLOR_GRAY2BGR)
        vis_img[final_mask == 1] = vis_img[final_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
        vis_img[final_mask == 2] = vis_img[final_mask == 2] * 0.5 + np.array([0, 0, 255]) * 0.5
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_xy_overlay.jpg"), vis_img)

        return {'status': 'success', 'msg': f"Success {base_name}"}

    except Exception as e:
        return {'status': 'error', 'msg': f"Error {target_id}-{neighbor_id}: {str(e)}"}



def process_single_connection_multi_slice(seg_vol, raw_vol, conn, output_dir, output_dir3d,
                                          initial_radius_nm=4000,
                                          max_radius_nm=32100,
                                          expansion_factor=2.0,
                                          num_frames=5):
    try:
        
        target_id = int(conn['target_id'])
        neighbor_id = int(conn['neighbor_id'])
        cx = int(conn.get('contact_x', conn.get('x')))
        cy = int(conn.get('contact_y', conn.get('y')))
        cz = int(conn.get('contact_z', conn.get('z')))
        endpoint_id = int(conn.get('endpoint_id', 0))

        base_name = f"{target_id}_{neighbor_id}_{endpoint_id}"

        
        if os.path.exists(os.path.join(output_dir, f"{base_name}_z0_xy_overlay.jpg")):
            return {'status': 'skipped', 'msg': f"Skipped {base_name}"}

        res = np.array(seg_vol.resolution)
        current_radius_nm = initial_radius_nm

        final_seg_2d = None
        final_box = None

        
        
        while True:
            r_vox = (current_radius_nm / res).astype(int)
            center = np.array([cx, cy, cz])

            
            raw_start = center - r_vox
            raw_end = center + r_vox + 1
            vol_max = seg_vol.bounds.maxpt
            start = np.maximum(raw_start, 0)
            end = np.minimum(raw_end, vol_max)

            
            start[2] = cz
            end[2] = cz + 1

            
            touching_min = np.any(raw_start[:2] < 0)
            touching_max = np.any(raw_end[:2] > vol_max[:2])
            if touching_min or touching_max:
                break  

            
            seg_cutout = seg_vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            seg_2d = np.array(seg_cutout).squeeze()
            if seg_2d.ndim == 2: seg_2d = seg_2d.T

            final_seg_2d = seg_2d
            final_box = (start, end)  

            
            if current_radius_nm >= max_radius_nm: break
            if check_boundaries(seg_2d, [target_id, neighbor_id]):
                current_radius_nm *= expansion_factor
                continue
            else:
                break

        
        
        
        mask_t_center = (final_seg_2d == target_id)
        mask_n_center = (final_seg_2d == neighbor_id)

        
        mask_t_center = get_center_connected_component_2d(mask_t_center)
        mask_n_center = get_center_connected_component_2d(mask_n_center)

        if not np.any(mask_t_center) or not np.any(mask_n_center):
            
            r_vox_final = (current_radius_nm / res).astype(int)

            
            video_res = prepare_sam2_video_region(
                seg_vol, raw_vol, center, target_id, neighbor_id, r_vox_final,
                output_dir3d, base_name, num_frames=num_frames
            )
            return {'status': 'z_gap', 'conn': conn, 'msg': f"Z-Gap: {video_res.get('msg', 'Saved video sequence')}"}

        
        

        saved_count = 0

        
        z_offsets = [-1, 0, 1]

        
        
        xy_start = final_box[0].copy()
        xy_end = final_box[1].copy()

        for z_off in z_offsets:
            current_z = cz + z_off

            
            if current_z < 0 or current_z >= seg_vol.bounds.maxpt[2]:
                continue

            
            curr_start = xy_start.copy()
            curr_end = xy_end.copy()
            curr_start[2] = current_z
            curr_end[2] = current_z + 1

            
            try:
                seg_layer = np.array(
                    seg_vol[curr_start[0]:curr_end[0], curr_start[1]:curr_end[1], curr_start[2]:curr_end[2]]).squeeze()
                if seg_layer.ndim == 2: seg_layer = seg_layer.T
            except:
                continue

            
            mask_t = (seg_layer == target_id)
            mask_n = (seg_layer == neighbor_id)

            
            mask_t = get_center_connected_component_2d(mask_t)
            mask_n = get_center_connected_component_2d(mask_n)

            
            if not np.any(mask_t) or not np.any(mask_n):
                continue

            
            raw_layer = np.array(
                raw_vol[curr_start[0]:curr_end[0], curr_start[1]:curr_end[1], curr_start[2]:curr_end[2]]).squeeze()
            if raw_layer.ndim == 2: raw_layer = raw_layer.T

            
            
            combined_mask = np.logical_or(mask_t, mask_n)
            raw_crop, _ = crop_tight_2d(raw_layer, combined_mask, padding=20)

            
            rows = np.any(combined_mask, axis=1);
            cols = np.any(combined_mask, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]];
            xmin, xmax = np.where(cols)[0][[0, -1]]
            pad = 20;
            h, w = raw_layer.shape
            ymin = max(0, ymin - pad);
            ymax = min(h, ymax + pad + 1)
            xmin = max(0, xmin - pad);
            xmax = min(w, xmax + pad + 1)

            mask_t_crop = mask_t[ymin:ymax, xmin:xmax]
            mask_n_crop = mask_n[ymin:ymax, xmin:xmax]

            final_mask = np.zeros_like(mask_t_crop, dtype=np.uint8)
            final_mask[mask_t_crop] = 1
            final_mask[mask_n_crop] = 2

            
            max_size = 512
            h_c, w_c = final_mask.shape
            if max(h_c, w_c) > max_size:
                scale = max_size / max(h_c, w_c)
                final_mask = cv2.resize(final_mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                raw_crop = cv2.resize(raw_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            if raw_crop.dtype != np.uint8:
                raw_crop = cv2.normalize(raw_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            
            # -1 -> "zn1", 0 -> "z0", 1 -> "zp1"
            suffix = f"z{z_off}" if z_off >= 0 else f"zn{abs(z_off)}"

            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{suffix}_xy_raw.jpg"), raw_crop)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{suffix}_xy_mask.png"), final_mask)

            vis_img = cv2.cvtColor(raw_crop, cv2.COLOR_GRAY2BGR)
            vis_img[final_mask == 1] = vis_img[final_mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
            vis_img[final_mask == 2] = vis_img[final_mask == 2] * 0.5 + np.array([0, 0, 255]) * 0.5
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{suffix}_xy_overlay.jpg"), vis_img)

            saved_count += 1

        if saved_count > 0:
            return {'status': 'success', 'msg': f"Saved {saved_count} slices for {base_name}"}
        else:
            
            
            return {'status': 'error', 'msg': f"Failed to save any slice for {base_name}"}

    except Exception as e:
        return {'status': 'error', 'msg': f"Error {target_id}-{neighbor_id}: {str(e)}"}


def process_single_connection_wrapper(args):
    raw_path, seg_path, conn, output_dir, output_dir3d, num_frames = args

    
    
    seg_vol = CloudVolume(seg_path, mip=0, parallel=False, fill_missing=True, cache=False)
    raw_vol = CloudVolume(raw_path, mip=0, parallel=False, fill_missing=True, cache=False)

    
    return process_single_connection(seg_vol, raw_vol, conn, output_dir, output_dir3d, num_frames=num_frames)
    # return process_single_connection_multi_slice(seg_vol, raw_vol, conn, output_dir, output_dir3d, num_frames=num_frames)



def get_slices(raw_path, seg_path, connections, output_folder, output_folder3d, max_workers=8, num_frames=5):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder3d, exist_ok=True)

    if isinstance(connections, pd.DataFrame):
        connections_list = connections.to_dict('records')
    else:
        connections_list = connections

    
    tasks = []
    for conn in connections_list:
        tasks.append((raw_path, seg_path, conn, output_folder, output_folder3d, num_frames))

    # print(f"Starting Multi-Processing with {max_workers} workers...")

    
    z_gap_connections = []

    
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_connection_wrapper, t) for t in tasks]

        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                
                result = f.result(timeout=120)

                
                status = result.get('status')

                if status == 'z_gap':
                    
                    z_gap_connections.append(result['conn'])
                    # print(f"Found Z-Gap: {result['msg']}")
                elif status == 'error':
                    print(result['msg'])
            except Exception as e:
                print(f"Task Timeout or Error: {e}")

    return z_gap_connections


if __name__ == "__main__":
    start_time = time.time()

    
    raw_path = '/path/to/raw/precomputed'
    seg_path = '/path/to/segmentation/precomputed'
    csv_file = 'connections.csv'
    output_folder = './slices_output_2d_dynamic_cs'

    
    df = pd.read_csv(csv_file)

    
    get_slices(raw_path, seg_path, df, output_folder, max_workers=16)

    end_time = time.time()
    print("Runtime: {:.4f} seconds".format(end_time - start_time))
