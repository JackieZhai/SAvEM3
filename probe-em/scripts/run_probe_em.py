import os
import argparse
import sys

import time
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core pipeline functions
from probe_em.get_endpoints_vectors import is_messy_segment, get_endpoints_vectors_precomputed
from probe_em.get_neighbors import get_neighbors, save_connections_to_csv
from probe_em.get_slices import get_slices
from probe_em.find_merge_candidates import find_merge_candidates
from probe_em.device import resolve_device
from probe_em.find_merge_candidates_3d_region import find_merge_candidates_3d_region

DEFAULT_CONFIG = {
    "raw_path": "/path/to/raw/precomputed",
    "seg_path": "/path/to/segmentation/precomputed",
    "checkpoint_sam": "/path/to/sam2_checkpoint.pt",
    "model_cfg_sam": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "voxel_threshold": 200,
    "sam2_num_frames": 5,
    "debug_limit": 80,
    "target_mip": 2,
    "output_root": "trace_results",
    "suffix": "sam",
    "gpu_id": "0",
    "device": "auto",
    "max_workers": 4,
    "slice_workers": 8,
    "seed_ids": [123456789],
    "seed_list_file": None,
}


def load_config(config_path):
    config = DEFAULT_CONFIG.copy()
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Copy configs/config.example.json to configs/config.json and edit the paths first."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def validate_config(config):
    required_paths = ["raw_path", "seg_path", "checkpoint_sam", "model_cfg_sam"]

    missing = [
        key for key in required_paths
        if not config.get(key) or str(config[key]).startswith("/path/to/")
    ]
    if missing:
        raise ValueError(
            "Please set these fields in your config file before running: "
            + ", ".join(missing)
        )


def load_seed_ids(config):
    seed_list_file = config.get("seed_list_file")
    if seed_list_file and os.path.exists(seed_list_file):
        print(f">>> Found seed list: {seed_list_file}")
        with open(seed_list_file, "r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip().isdigit()]
    return [int(seed_id) for seed_id in config.get("seed_ids", [])]


class NeuronTracer:
    def __init__(self, raw_path, seg_path, sam_checkpoint=None, sam_cfg=None,
                 voxel_threshold=10000, sam2_num_frames=5, slice_workers=8,
                 debug_limit=None, gpu_id='0', device=None):
        self.raw_path = raw_path
        self.seg_path = seg_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_cfg = sam_cfg
        self.gpu_id = gpu_id
        self.device = resolve_device(device, gpu_id)

        self.voxel_threshold = voxel_threshold
        self.sam2_num_frames = sam2_num_frames
        self.slice_workers = slice_workers
        self.debug_limit = debug_limit

        self.stack = []
        self.visited = set()
        self.graph = nx.DiGraph()
        self.all_merged_ids = set()
        self.voxel_counts = {}

    def reset(self):
        """
        Reset tracer state before a new tracing run.
        """
        self.stack = []
        self.visited = set()
        self.graph = nx.DiGraph()
        self.all_merged_ids = set()
        self.voxel_counts = {}
        print("NeuronTracer state has been reset.")

    def trace(self, seed_id, save_path, target_mip=2):
        """
        Main tracing loop.
        """
        self.stack.append(seed_id)
        self.all_merged_ids.add(seed_id)

        process_count = 0

        print(f"Starting tracing from seed ID: {seed_id}")
        print(f"Config: voxel_threshold={self.voxel_threshold}, debug_limit={self.debug_limit}")

        while self.stack:
            if self.debug_limit and process_count >= self.debug_limit:
                print(f"\nReached debug limit ({self.debug_limit} nodes); stopping.")
                break

            current_id = self.stack.pop()

            if current_id in self.visited:
                continue

            self.visited.add(current_id)
            process_count += 1

            # print("\n" + "=" * 50)
            # print(f"[Seed {seed_id} - node {process_count}] Processing ID: {current_id}")
            # print("=" * 50)

            try:
                endpoints, vectors, res, voxel_count = get_endpoints_vectors_precomputed(
                    current_id, target_mip, self.seg_path
                )
                print('finish endpoints')

                # print(f"Voxel count: {voxel_count}")
                self.voxel_counts[current_id] = int(voxel_count)

                if (current_id != seed_id) and is_messy_segment(endpoints, res,max_endpoints=40):
                    # print("Pruned segment; downstream tracing skipped.")
                    if current_id in self.all_merged_ids:
                        self.all_merged_ids.remove(current_id)
                    continue

                connections = get_neighbors(
                    self.seg_path, current_id, endpoints, vectors, res
                )
                print('finish neighbors')

                if not connections:
                    # print("No geometric neighbors found.")
                    continue

                temp_slice_dir = os.path.join(save_path, f'temp_slices_{current_id}')
                temp_vis_dir = os.path.join(save_path, f'temp_vis_{current_id}')
                temp_slice3d_dir = os.path.join(save_path, f'temp_slices3d_{current_id}')
                temp_vis3d_dir = os.path.join(save_path, f'temp_vis3d_{current_id}')

                if not os.path.exists(temp_vis_dir): os.makedirs(temp_vis_dir)
                if not os.path.exists(temp_vis3d_dir): os.makedirs(temp_vis3d_dir)
                save_connections_to_csv(connections, os.path.join(temp_vis_dir, 'neighbors.csv'))

                z_gap_conns = get_slices(
                    self.raw_path, self.seg_path, connections, temp_slice_dir,
                    temp_slice3d_dir, max_workers=self.slice_workers,
                    num_frames=self.sam2_num_frames
                )
                save_connections_to_csv(z_gap_conns, os.path.join(temp_vis3d_dir, 'neighbors_z.csv'))
                
                merged_candidates = find_merge_candidates(
                    self.sam_checkpoint, self.sam_cfg,
                    temp_slice_dir, temp_vis_dir, save_temp=False,
                    device=self.device
                )
                merged_candidates_3d = find_merge_candidates_3d_region(
                    self.sam_checkpoint, self.sam_cfg,
                    temp_slice3d_dir, temp_vis3d_dir, save_temp=False,
                    device=self.device
                )
                merged_candidates = list(set(merged_candidates).union(merged_candidates_3d))
                # print(f"SAM 2 verified connections: {merged_candidates}")

                for child_id_str in merged_candidates:
                    try:
                        child_id = int(child_id_str)
                    except ValueError:
                        # print(f"Warning: could not convert ID {child_id_str} to int; skipping.")
                        continue
                    
                    if child_id not in self.visited:
                        self.graph.add_edge(current_id, child_id)
                    self.all_merged_ids.add(child_id)

                    if child_id not in self.visited and child_id not in self.stack:
                        self.stack.append(child_id)
                        # print(f"Added to stack: {child_id}")

            except Exception as e:
                # print(f"Error while processing seed {seed_id}, ID {current_id}: {e}")
                continue

        print(f"Finished tracing seed {seed_id}.")
        return list(self.all_merged_ids)

    def save_results(self, seed_id, save_path):
        """
        Save traced IDs, the merge tree JSON, and a graph visualization.
        """
        os.makedirs(save_path, exist_ok=True)
        id_list = sorted(list(self.all_merged_ids))
        np.savetxt(os.path.join(save_path, f'trace_{seed_id}_ids.txt'), id_list, fmt='%d')
        # print(f"Saved ID list: trace_{seed_id}_ids.txt ({len(id_list)} IDs)")

        segment_ids_str = [str(x) for x in id_list]
        ng_data = {"segments": segment_ids_str}
        json_str = json.dumps(ng_data, indent=2)
        final_output = json_str.strip().strip("{}").strip()

        ng_filename = f'trace_{seed_id}_ng_segments.txt'
        with open(os.path.join(save_path, ng_filename), 'w', encoding='utf-8') as f:
            f.write(final_output)

        adj_data = nx.to_dict_of_lists(self.graph)
        json_data = {str(k): [int(v) for v in vals] for k, vals in adj_data.items()}
        with open(os.path.join(save_path, f'trace_{seed_id}_tree.json'), 'w') as f:
            json.dump(json_data, f, indent=2)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=False, node_size=30, node_color="skyblue", alpha=0.6, edge_color="gray")
        labels = {n: str(n) for n in self.graph.nodes() if self.graph.degree(n) > 1}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=8)
        plt.title(f"Neuron Merge Tree - Seed: {seed_id}")
        plt.savefig(os.path.join(save_path, f"trace_{seed_id}_graph.png"))
        plt.close()


def run_one_seed(seed_id, config):
    """
    Run tracing for one seed ID.
    """
    save_path = os.path.join(config.get("output_root", "trace_results"), f"{seed_id}_results_{config['suffix']}")
    
    if os.path.exists(save_path):
        print(f"Skipping existing result: {seed_id}")
        return seed_id, "skipped"

    tracer = NeuronTracer(
        config['raw_path'], config['seg_path'], 
        sam_checkpoint=config['checkpoint_sam'], sam_cfg=config['model_cfg_sam'],
        voxel_threshold=config['voxel_threshold'],
        sam2_num_frames=config['sam2_num_frames'], 
        slice_workers=int(config.get('slice_workers', 8)),
        debug_limit=config['debug_limit'],
        gpu_id=config.get('gpu_id', '0'),
        device=config.get('device', 'auto')
    )
    
    try:
        final_ids = tracer.trace(seed_id, save_path, target_mip=config['target_mip'])
        tracer.save_results(seed_id, save_path)
        return seed_id, "done"
    except Exception as e:
        print(f"Seed {seed_id} failed: {e}")
        return seed_id, f"error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Probe-EM tracing.")
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Path to a JSON config file. Copy configs/config.example.json to configs/config.json first.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)

    device = resolve_device(config.get("device", "auto"), config.get("gpu_id", "0"))
    if str(device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.get("gpu_id", "0"))
        print(f">>> CUDA_VISIBLE_DEVICES = {config.get('gpu_id', '0')}")
    print(f">>> Probe-EM device: {device}")

    target_ids = load_seed_ids(config)
    if not target_ids:
        raise ValueError("No seed IDs were provided. Set seed_ids or seed_list_file in the config.")

    max_workers = int(config.get("max_workers", 4))

    print(f"Starting Probe-EM tracing with {max_workers} workers")
    print(f"SAM 2 checkpoint: {config['checkpoint_sam']}")
    t1 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one_seed, sid, config): sid for sid in target_ids}

        for future in as_completed(futures):
            sid = futures[future]
            try:
                sid, status = future.result()
                print(f">>> Finished seed {sid}: {status}")
            except Exception as e:
                print(f">>> Failed seed {sid}: {e}")

    t2 = time.time()
    print(f"\nTotal runtime: {t2 - t1:.4f} seconds")
