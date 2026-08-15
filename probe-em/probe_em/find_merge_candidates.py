import matplotlib


matplotlib.use('Agg')

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import glob
import pandas as pd
from tqdm import tqdm
import shutil

from probe_em.device import resolve_device, autocast_dtype, sam2_config_name

# ==========================================

# ==========================================

def get_robust_prompts_n1(source_mask, target_mask, num_points=3, safety_margin=5.0):
    
    src = source_mask.astype(np.uint8)
    tgt = target_mask.astype(np.uint8)

    if cv2.countNonZero(src) == 0: return None, None

    points = []

    # =========================================================
    
    
    # =========================================================
    dist_internal = cv2.distanceTransform(src, cv2.DIST_L2, 5)

    
    
    _, max_val, _, max_loc_anchor = cv2.minMaxLoc(dist_internal)
    points.append(max_loc_anchor)

    # =========================================================
    
    # =========================================================
    
    safe_mask = dist_internal > safety_margin

    
    
    if cv2.countNonZero(safe_mask.astype(np.uint8)) == 0:
        
        safe_mask = dist_internal >= (max_val * 0.5)

    # =========================================================
    
    
    # =========================================================
    dist_to_target = cv2.distanceTransform((1 - tgt).astype(np.uint8), cv2.DIST_L2, 5)

    
    
    
    valid_dist_map = dist_to_target.copy()

    
    
    

    y_indices, x_indices = np.where(safe_mask)

    
    if len(y_indices) == 0:
        return np.array(points), np.ones(len(points), dtype=np.int32)

    
    distances_to_tgt = valid_dist_map[y_indices, x_indices]
    coords = np.column_stack((x_indices, y_indices))  # (x, y)

    
    
    
    sorted_indices = np.argsort(distances_to_tgt)

    # =========================================================
    
    # =========================================================
    if num_points >= 2:
        
        
        top_k_rear = max(1, int(len(sorted_indices) * 0.05))
        
        rear_candidates_idx = sorted_indices[-top_k_rear:]

        selected_rear_idx = rear_candidates_idx[np.random.randint(len(rear_candidates_idx))]
        rear_point = coords[selected_rear_idx]

        
        if np.linalg.norm(rear_point - max_loc_anchor) > 5:
            points.append(rear_point)
        else:
            
            points.append(rear_point)

    # =========================================================
    
    # =========================================================
    points_needed = num_points - len(points)

    if points_needed > 0:
        
        
        top_k_probe = max(3, int(len(sorted_indices) * 0.1))
        probe_candidates_idx = sorted_indices[:top_k_probe]

        if len(probe_candidates_idx) >= points_needed:
            sel = np.random.choice(probe_candidates_idx, points_needed, replace=False)
            for idx in sel: points.append(coords[idx])
        else:
            
            for i in range(points_needed):
                idx = probe_candidates_idx[i % len(probe_candidates_idx)]
                points.append(coords[idx])

    
    points = np.array(points)
    labels = np.ones(len(points), dtype=np.int32)  # 1 = Positive

    return points, labels


def get_robust_prompts(source_mask, target_mask, num_points=3):
    src = source_mask.astype(np.uint8)
    tgt = target_mask.astype(np.uint8)
    if cv2.countNonZero(src) == 0: return None, None
    points = []

    # Anchor
    dist_internal = cv2.distanceTransform(src, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_internal)
    points.append(max_loc)

    # Probes
    if num_points > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        src_eroded = cv2.erode(src, kernel, iterations=1)
        if cv2.countNonZero(src_eroded) == 0: src_eroded = src
        dist_to_target = cv2.distanceTransform((1 - tgt).astype(np.uint8), cv2.DIST_L2, 5)
        valid_dist = dist_to_target.copy()
        valid_dist[src_eroded == 0] = np.inf
        y_indices, x_indices = np.where(src_eroded > 0)
        if len(y_indices) > 0:
            distances = valid_dist[y_indices, x_indices]
            coords = np.column_stack((x_indices, y_indices))
            sorted_indices = np.argsort(distances)
            top_k_limit = max(5, int(len(sorted_indices) * 0.2))
            candidate_indices = sorted_indices[:top_k_limit]
            needed = num_points - 1
            if len(candidate_indices) >= needed:
                sel = np.random.choice(candidate_indices, needed, replace=False)
                for idx in sel: points.append(coords[idx])
            else:
                for idx in candidate_indices: points.append(coords[idx])

    return np.array(points), np.ones(len(points), dtype=np.int32)


def get_negative_prompts(mask1, mask2, num_neg_points=3, buffer_size=30):
    combined_mask = np.logical_or(mask1, mask2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_size, buffer_size))
    unsafe_area = cv2.dilate(combined_mask, kernel, iterations=1)
    safe_bg = 1 - unsafe_area
    if cv2.countNonZero(safe_bg) == 0: return None, None
    points = []
    dist_bg = cv2.distanceTransform(safe_bg, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_bg)
    points.append(max_loc)
    if num_neg_points > 1:
        y_idx, x_idx = np.where(safe_bg > 0)
        count = len(y_idx)
        needed = num_neg_points - 1
        if count >= needed:
            indices = np.random.choice(count, needed, replace=False)
            for idx in indices: points.append([x_idx[idx], y_idx[idx]])
        else:
            for i in range(count): points.append([x_idx[i], y_idx[i]])
    return np.array(points), np.zeros(len(points), dtype=np.int32)


def calculate_overlap_ratio(pred_mask, target_gt_mask):
    if pred_mask is None: return 0.0
    pred_bool = pred_mask > 0
    target_bool = target_gt_mask > 0
    intersection = np.logical_and(pred_bool, target_bool).sum()
    target_area = target_bool.sum()
    if target_area == 0: return 0.0
    return intersection / target_area


def process_pair(predictor, raw_path, mask_path, overlay_path, output_dir,
                 total_trials=5, min_pass_trials=4, need_negative=True,
                 max_score_threshold=0.5, min_score_threshold=0.1):
    
    filename = os.path.basename(raw_path)
    base_name = filename.replace('_raw.jpg', '')
    parts = base_name.split('_')

    if len(parts) >= 2:
        id1 = parts[0]
        id2 = parts[1]
    else:
        id1 = "Unknown"
        id2 = "Unknown"

    if not os.path.exists(raw_path) or not os.path.exists(mask_path):
        return None

    
    image_bgr = cv2.imread(raw_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask1 = (mask_img == 1)
    mask2 = (mask_img == 2)

    
    count1 = np.sum(mask1)
    count2 = np.sum(mask2)
    if count1 == 0 or count2 == 0: return None
    ratio = min(count1, count2) / max(count1, count2)
    if ratio < 0.03: return None

    
    predictor.set_image(image_rgb)

    pass_count = 0
    fail_count = 0
    run_count = 0
    scores_history = []

    
    max_allowed_fails = total_trials - min_pass_trials

    
    last_vis_data = {}

    
    h, w = image_rgb.shape[:2]
    total_pixels = h * w
    
    OVER_SEGMENTATION_THRESHOLD = 0.90

    # ==========================================================
    
    # ==========================================================
    for i in range(total_trials):
        
        if pass_count >= min_pass_trials:
            break  

        if fail_count > max_allowed_fails:
            break  

        
        if need_negative:
            neg_points, neg_labels = get_negative_prompts(mask1, mask2, num_neg_points=3, buffer_size=30)
        else:
            neg_points, neg_labels = None, None

        # Test A: 1->2
        # pos_points_1, pos_labels_1 = get_robust_prompts(mask1, mask2, num_points=3)
        pos_points_1, pos_labels_1 = get_robust_prompts_n1(mask1, mask2, num_points=4)
        sam_mask_1 = None
        score_1_to_2 = 0.0
        if pos_points_1 is not None:
            if neg_points is not None:
                pts = np.vstack([pos_points_1, neg_points])
                lbls = np.concatenate([pos_labels_1, neg_labels])
            else:
                pts, lbls = pos_points_1, pos_labels_1
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
            sam_mask_1 = masks[np.argmax(scores)]
            
            mask_ratio = np.sum(sam_mask_1 > 0) / total_pixels
            if mask_ratio > OVER_SEGMENTATION_THRESHOLD:
                
                score_1_to_2 = 0.0
            else:
                score_1_to_2 = calculate_overlap_ratio(sam_mask_1, mask2)

        # Test B: 2->1
        # pos_points_2, pos_labels_2 = get_robust_prompts(mask2, mask1, num_points=3)
        pos_points_2, pos_labels_2 = get_robust_prompts_n1(mask2, mask1, num_points=4)
        sam_mask_2 = None
        score_2_to_1 = 0.0
        if pos_points_2 is not None:
            if neg_points is not None:
                pts = np.vstack([pos_points_2, neg_points])
                lbls = np.concatenate([pos_labels_2, neg_labels])
            else:
                pts, lbls = pos_points_2, pos_labels_2
            masks, scores, _ = predictor.predict(point_coords=pts, point_labels=lbls, multimask_output=True)
            sam_mask_2 = masks[np.argmax(scores)]
            
            mask_ratio = np.sum(sam_mask_2 > 0) / total_pixels
            if mask_ratio > OVER_SEGMENTATION_THRESHOLD:
                score_2_to_1 = 0.0
            else:
                score_2_to_1 = calculate_overlap_ratio(sam_mask_2, mask1)

        
        current_max_score = max(score_1_to_2, score_2_to_1)
        current_min_score = min(score_1_to_2, score_2_to_1)
        scores_history.append(current_max_score)

        if (current_max_score > max_score_threshold) and (current_min_score > min_score_threshold):
            pass_count += 1
        else:
            fail_count += 1

        run_count += 1

        
        last_vis_data = {
            "score1": score_1_to_2, "mask1": sam_mask_1, "pts1": pos_points_1,
            "score2": score_2_to_1, "mask2": sam_mask_2, "pts2": pos_points_2,
            "neg_pts": neg_points
        }

    # ==========================================================
    
    # ==========================================================
    is_connected = (pass_count >= min_pass_trials)

    vis_filename = f"{base_name}_result.jpg"
    vis_save_path = os.path.join(output_dir, vis_filename)

    
    plt.figure(figsize=(24, 7))
    s1 = last_vis_data.get("score1", 0)
    m1 = last_vis_data.get("mask1")
    p1 = last_vis_data.get("pts1")
    s2 = last_vis_data.get("score2", 0)
    m2 = last_vis_data.get("mask2")
    p2 = last_vis_data.get("pts2")
    neg = last_vis_data.get("neg_pts")

    # Subplot 1
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title(f"Test 1->2 (Last: {s1:.2%})")
    ax1.imshow(image_rgb)
    if m1 is not None:
        if m1.ndim == 3: m1 = m1.squeeze()
        vis = np.zeros((*mask_img.shape, 4));
        vis[m1 > 0] = [0, 0.4, 1.0, 0.5]
        ax1.imshow(vis)
        if p1 is not None:
            ax1.scatter(p1[0, 0], p1[0, 1], c='red', marker='D', s=100, edgecolors='white')
            if len(p1) > 1: ax1.scatter(p1[1:, 0], p1[1:, 1], c='yellow', marker='*', s=150, edgecolors='black')
        if neg is not None: ax1.scatter(neg[:, 0], neg[:, 1], c='magenta', marker='X', s=80, edgecolors='white')
    ax1.axis('off')

    # Subplot 2
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title(f"Test 2->1 (Last: {s2:.2%})")
    ax2.imshow(image_rgb)
    if m2 is not None:
        if m2.ndim == 3: m2 = m2.squeeze()
        vis = np.zeros((*mask_img.shape, 4));
        vis[m2 > 0] = [0, 0.4, 1.0, 0.5]
        ax2.imshow(vis)
        if p2 is not None:
            ax2.scatter(p2[0, 0], p2[0, 1], c='red', marker='D', s=100, edgecolors='white')
            if len(p2) > 1: ax2.scatter(p2[1:, 0], p2[1:, 1], c='yellow', marker='*', s=150, edgecolors='black')
        if neg is not None: ax2.scatter(neg[:, 0], neg[:, 1], c='magenta', marker='X', s=80, edgecolors='white')
    ax2.axis('off')

    # Subplot 3
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title(
        f"Pass: {pass_count}/{run_count} (Req: {min_pass_trials})\nFinal: {'CONNECTED' if is_connected else 'Disconnected'}")
    if os.path.exists(overlay_path):
        overlay_img = cv2.imread(overlay_path)
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        ax3.imshow(overlay_img)
    else:
        ax3.imshow(image_rgb)
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(vis_save_path)
    plt.close()

    
    return {
        "target_id": id1,
        "neighbor_id": int(id2),
        "avg_score": np.mean(scores_history) if scores_history else 0.0,
        "is_connected": int(is_connected)
    }


def find_merge_candidates(checkpoint, model_cfg, input_dir, output_dir, save_temp=False, device=None,
                          max_score_threshold=0.5, min_score_threshold=0.1):
    os.makedirs(output_dir, exist_ok=True)

    device = resolve_device(device)
    print(f"Loading SAM 2 image predictor on {device} ...")
    sam2_model = build_sam2(sam2_config_name(model_cfg), checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    
    raw_files = glob.glob(os.path.join(input_dir, "*_raw.jpg"))

    
    connected_neighbors = set()

    
    dtype = autocast_dtype(device)

    with torch.inference_mode(), torch.autocast(device, dtype=dtype):
        for raw_path in tqdm(raw_files, desc="SAM2 Inference"):
            mask_path = raw_path.replace("_raw.jpg", "_mask.png")
            overlay_path = raw_path.replace("_raw.jpg", "_overlay.jpg")

            try:
                
                res = process_pair(predictor, raw_path, mask_path, overlay_path, output_dir, total_trials=5,
                                   min_pass_trials=4, need_negative=False,
                                   max_score_threshold=max_score_threshold,
                                   min_score_threshold=min_score_threshold)

                if res and res['is_connected']:
                    
                    connected_neighbors.add(res['neighbor_id'])

            except Exception as e:
                
                print(f"[Warning] Error processing {os.path.basename(raw_path)}: {e}")

    if os.path.exists(input_dir) and (save_temp == 0):
        shutil.rmtree(input_dir)

    return list(connected_neighbors)


if __name__ == "__main__":
    checkpoint = "/path/to/sam2_checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    device = resolve_device()

    input_dir = "./slices_output_2d_dynamic_c"
    output_dir = "./sam2_results"
    os.makedirs(output_dir, exist_ok=True)
    csv_output_path = os.path.join(output_dir, "connection_results.csv")

    print(f"Loading SAM 2 on {device}...")
    sam2_model = build_sam2(sam2_config_name(model_cfg), checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    raw_files = glob.glob(os.path.join(input_dir, "*_raw.jpg"))
    print(f"Found {len(raw_files)} pairs to process.")

    results_list = []

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        for raw_path in tqdm(raw_files):
            mask_path = raw_path.replace("_raw.jpg", "_mask.png")
            overlay_path = raw_path.replace("_raw.jpg", "_overlay.jpg")
            try:
                res = process_pair(predictor, raw_path, mask_path, overlay_path, output_dir, num_trials=3)
                if res:
                    results_list.append(res)
            except Exception as e:
                print(f"Error processing {raw_path}: {e}")

    if results_list:
        df = pd.DataFrame(results_list)

        
        cols = ["target_id", "neighbor_id", "vote_count", "avg_score", "is_connected", "vis_path"]
        df = df[cols]  

        df.to_csv(csv_output_path, index=False)
        print(f"\nProcessing complete. Results saved to: {csv_output_path}")

        num_connected = df['is_connected'].sum()
        print(f"Total: {len(df)}, Connected: {num_connected}, Disconnected: {len(df) - num_connected}")
    else:
        print("No results generated.")
