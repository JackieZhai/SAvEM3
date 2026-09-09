import os
import glob
import cv2
import numpy as np
import torch
import matplotlib


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm
import shutil
import warnings
import pandas as pd

from probe_em.device import resolve_device, inference_autocast, sam2_config_name


warnings.filterwarnings("ignore")


# ==========================================

# ==========================================
def show_mask(mask, ax, color_code=[30, 144, 255]):
    if mask is None: return
    color = np.array([color_code[0] / 255, color_code[1] / 255, color_code[2] / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def calculate_collision(pred_mask, seg_npy, target_id, min_pixels=50, coverage_threshold=0.1):
    if pred_mask is None or seg_npy is None: return {}

    
    mask_bool = pred_mask > 0

    
    
    if seg_npy.shape != mask_bool.shape:
        
        # OpenCV cannot resize uint64 labels without losing segment IDs.
        rows = np.arange(mask_bool.shape[0]) * seg_npy.shape[0] // mask_bool.shape[0]
        columns = np.arange(mask_bool.shape[1]) * seg_npy.shape[1] // mask_bool.shape[1]
        seg_npy = seg_npy[rows[:, None], columns[None, :]]

    covered_pixels = seg_npy[mask_bool]

    if len(covered_pixels) == 0: return {}

    
    unique_ids, counts = np.unique(covered_pixels, return_counts=True)

    results = {}

    for uid, count in zip(unique_ids, counts):
        
        if uid == 0 or uid == target_id:
            continue

        
        if count < min_pixels:
            continue

        
        
        
        total_area = np.sum(seg_npy == uid)
        coverage = count / total_area if total_area > 0 else 0

        if coverage > coverage_threshold:
            results[uid] = coverage

    return results


# ==========================================

# ==========================================
def process_single_video_folder_collision(folder_path, predictor, output_root):
    base_name = os.path.basename(folder_path)

    
    try:
        target_id = int(base_name.split('_')[0])
    except (ValueError, IndexError) as error:
        raise ValueError(f'Invalid ASP case name: {base_name}') from error

    video_dir = os.path.join(folder_path, "video_frames")
    seg_dir = os.path.join(folder_path, "segmentation_data")  
    prompt_path = os.path.join(folder_path, "prompt_mask_00000.png")

    
    
    if not os.path.exists(video_dir) or not os.path.exists(seg_dir):
        raise FileNotFoundError(f'Incomplete ASP case: {folder_path}')

    
    inference_state = predictor.init_state(video_path=video_dir)

    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(prompt_path)
    prompt_img = cv2.imread(prompt_path, cv2.IMREAD_UNCHANGED)
    prompt_bool = (prompt_img > 0)

    _, _, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=prompt_bool
    )

    
    found_connections = {}  # {id: max_score}

    
    frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".jpg")])
    num_frames = len(frame_files)

    
    video_segments = {}
    video_segments[0] = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()

    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
        pred_mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
        video_segments[out_frame_idx] = pred_mask

        
        
        seg_npy_path = os.path.join(seg_dir, f"{out_frame_idx:05d}_seg.npy")
        if not os.path.isfile(seg_npy_path):
            raise FileNotFoundError(seg_npy_path)

        if os.path.exists(seg_npy_path):
            seg_npy = np.load(seg_npy_path)  # uint64 array

            
            collisions = calculate_collision(pred_mask, seg_npy, target_id)

            
            for uid, score in collisions.items():
                if uid not in found_connections:
                    found_connections[uid] = score
                else:
                    found_connections[uid] = max(found_connections[uid], score)

    
    plt.figure(figsize=(4 * num_frames, 5))
    ids_str = ", ".join([str(k) for k in list(found_connections.keys())[:3]])  
    if len(found_connections) > 3: ids_str += "..."
    plt.suptitle(f"{base_name} | Found: {ids_str if ids_str else 'None'}", fontsize=12)

    for i in range(num_frames):
        ax = plt.subplot(1, num_frames, i + 1)
        img_path = os.path.join(video_dir, frame_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)

        
        if i in video_segments:
            show_mask(video_segments[i], ax, color_code=[30, 144, 255])

        ax.set_title(f"F{i}")
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_root, f"{base_name}_collision.jpg")
    plt.savefig(save_path)
    plt.close()

    
    result_list = []
    for uid, score in found_connections.items():
        result_list.append({
            'target_id': target_id,
            'neighbor_id': int(uid),
            'score': score,
            'case_name': base_name
        })

    return result_list


# ==========================================

# ==========================================
def find_merge_candidates_3d_region(checkpoint, model_cfg, input_dir, output_dir, save_temp=False,
                                    device=None, predictor=None, strict=True):
    os.makedirs(output_dir, exist_ok=True)

    subfolders = sorted(path for path in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(path))
    if not subfolders:
        return []
    device = resolve_device(device)
    if predictor is None:
        sam2_predictor = build_sam2_video_predictor(sam2_config_name(model_cfg), checkpoint, device=device)
    else:
        sam2_predictor = predictor() if callable(predictor) else predictor

    
    subfolders = sorted(glob.glob(os.path.join(input_dir, "*")))
    subfolders = [f for f in subfolders if os.path.isdir(f)]

    print(f"Found {len(subfolders)} 3D cases to process.")

    all_logs = []
    connected_neighbors = set()

    with torch.inference_mode(), inference_autocast(device):
        for folder in tqdm(subfolders, desc="SAM2 3D Collision"):
            try:
                
                connections = process_single_video_folder_collision(folder, sam2_predictor, output_dir)

                
                for item in connections:
                    all_logs.append(item)
                    
                    if item['score'] > 0.6:
                        connected_neighbors.add(item['neighbor_id'])

            except Exception as e:
                print(f"Error processing {folder}: {e}")
                import traceback
                traceback.print_exc()
                if strict:
                    raise

    
    if os.path.exists(input_dir) and (not save_temp):
        try:
            shutil.rmtree(input_dir)
            print(f"Cleaned up temp dir: {input_dir}")
        except OSError as e:
            print(f"Error deleting temp dir: {e}")

    
    if all_logs:
        df = pd.DataFrame(all_logs)
        df.to_csv(os.path.join(output_dir, "3d_collision_report.csv"), index=False)
        print(f"Report saved. Found {len(connected_neighbors)} unique neighbors.")

    return list(connected_neighbors)


# ==========================================

# ==========================================
if __name__ == "__main__":
    
    checkpoint = "/path/to/sam2_checkpoint.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    input_root_dir = "/path/to/temp_slices3d"
    output_root_dir = "/path/to/temp_vis3d"

    device = resolve_device()

    
    merged_ids = find_merge_candidates_3d_region(
        checkpoint, model_cfg, input_root_dir, output_root_dir,
        save_temp=True,  
        device=device
    )

    print("-" * 30)
    print("3D Merged Candidates:", merged_ids)
