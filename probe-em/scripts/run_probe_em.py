import os
import argparse
import sys
import random
import hashlib
import multiprocessing

import time
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cloudvolume import CloudVolume
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
from probe_em.runtime import PredictorCache
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
    "resume": False,
    "random_seed": 42,
    # Optional z-axis affine alignment (see probe_em/z_align.py):
    # register every z-slice of the trace region to a reference slice (ECC),
    # trace in the aligned space, then map coordinate outputs back.
    "align_z": False,
    "align_z_margin_xy": 256,      # mip0 voxels around the seed skeleton bbox
    "align_z_margin_z": 64,        # mip0 slices above/below the seed bbox
    "align_z_estimation_mip": 1,   # mip used for ECC estimation (faster)
    "align_z_motion": "translation",  # per-slice motion model: translation | euclidean | affine
    "align_z_field": None,         # optional precomputed field npz (reused if set)
    "align_z_max_z": 400,          # cap on the z extent of the field
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
    if not os.path.isfile(config['checkpoint_sam']):
        raise FileNotFoundError(config['checkpoint_sam'])
    for key in ('max_workers', 'slice_workers', 'sam2_num_frames'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if config['target_mip'] < 0 or config['debug_limit'] is not None and config['debug_limit'] < 0:
        raise ValueError('target_mip/debug_limit must be nonnegative')


def load_seed_ids(config):
    seed_list_file = config.get("seed_list_file")
    if seed_list_file:
        print(f">>> Found seed list: {seed_list_file}")
        with open(seed_list_file, "r", encoding="utf-8") as f:
            seeds = [int(line.strip()) for line in f if line.strip()]
    else:
        seeds = [int(seed_id) for seed_id in config.get("seed_ids", [])]
    if any(seed <= 0 or seed > np.iinfo(np.uint64).max for seed in seeds):
        raise ValueError('seed IDs must be positive uint64 integers')
    return list(dict.fromkeys(seeds))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class NeuronTracer:
    def __init__(self, raw_path, seg_path, sam_checkpoint=None, sam_cfg=None,
                 voxel_threshold=10000, sam2_num_frames=5, slice_workers=8,
                 debug_limit=None, gpu_id='0', device=None,
                 align_z=False, align_z_margin_xy=256, align_z_margin_z=64,
                 align_z_estimation_mip=1, align_z_motion="translation",
                 align_z_field=None, align_z_max_z=400,
                 output_root="trace_results", random_seed=42):
        self.raw_path = raw_path
        self.seg_path = seg_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_cfg = sam_cfg
        self.gpu_id = gpu_id
        self.device = resolve_device(device, gpu_id)
        self.predictors = PredictorCache(sam_checkpoint, sam_cfg, self.device)
        self.random_seed = int(random_seed)
        self.errors = []

        self.voxel_threshold = voxel_threshold
        self.sam2_num_frames = sam2_num_frames
        self.slice_workers = slice_workers
        self.debug_limit = debug_limit

        # z-axis affine alignment (optional)
        self.align_z = bool(align_z)
        self.align_z_margin_xy = int(align_z_margin_xy)
        self.align_z_margin_z = int(align_z_margin_z)
        self.align_z_estimation_mip = int(align_z_estimation_mip)
        self.align_z_motion = align_z_motion if align_z_motion in ("translation", "euclidean", "affine") else "translation"
        self.align_z_field_path = align_z_field
        self.align_z_max_z = int(align_z_max_z)
        self.z_field = None  # built lazily in trace() (needs the seed skeleton)
        self.z_field_saved_path = None
        self.output_root = output_root

        self.stack = []
        self.visited = set()
        self.graph = nx.DiGraph()
        self.all_merged_ids = set()
        self.voxel_counts = {}

    def _build_z_field(self, seed_id):
        """Load or estimate the per-z-slice affine field around the seed."""
        from probe_em.z_align import ZAffineField, estimate_z_affine_field

        if self.align_z_field_path and os.path.exists(self.align_z_field_path):
            print(f"[align-z] loading field from {self.align_z_field_path}")
            return ZAffineField.load(self.align_z_field_path)

        # seed skeleton bbox (vertices are absolute nm -> mip0 voxels)
        seg_vol = CloudVolume(self.seg_path, mip=2, parallel=False, fill_missing=True)
        try:
            skel = seg_vol.skeleton.get(seed_id)
        except Exception as e:
            print(f"[align-z] skeleton fetch failed: {e}")
            skel = None
        if skel is None or len(skel.vertices) == 0:
            print("[align-z] seed skeleton unavailable; alignment disabled")
            return None

        res0 = np.array(seg_vol.meta.resolution(0), dtype=float)
        if not np.allclose(res0, [8, 8, 30]):
            raise ValueError('The optional legacy z-alignment field currently requires 8x8x30 nm; disable align_z for other datasets')
        vox = skel.vertices / res0
        lo = np.floor(vox.min(axis=0)).astype(int)
        hi = np.ceil(vox.max(axis=0)).astype(int)

        z0 = max(lo[2] - self.align_z_margin_z, 0)
        z1 = hi[2] + self.align_z_margin_z
        if z1 - z0 > self.align_z_max_z:
            mid = (z0 + z1) // 2
            z0, z1 = mid - self.align_z_max_z // 2, mid + self.align_z_max_z // 2
        print(f"[align-z] estimating field over z in [{z0}, {z1}] "
              f"(window xy {lo[0] - self.align_z_margin_xy}..{hi[0] + self.align_z_margin_xy}, "
              f"{lo[1] - self.align_z_margin_xy}..{hi[1] + self.align_z_margin_xy})")

        raw_vol = CloudVolume(self.raw_path, mip=self.align_z_estimation_mip,
                              parallel=False, fill_missing=True, progress=False)
        window = [lo[0] - self.align_z_margin_xy, lo[1] - self.align_z_margin_xy,
                  hi[0] + self.align_z_margin_xy, hi[1] + self.align_z_margin_xy]
        bounds = np.array(raw_vol.bounds.maxpt) * np.array(raw_vol.resolution) / res0
        window = [max(0, int(window[0])), max(0, int(window[1])),
                  min(int(bounds[0]) - 1, int(window[2])), min(int(bounds[1]) - 1, int(window[3]))]
        field = estimate_z_affine_field(raw_vol, z0, z1, window,
                                        est_mip=self.align_z_estimation_mip,
                                        motion=self.align_z_motion)
        return field

    def _ensure_z_field(self, seed_id):
        if not self.align_z:
            return None
        if self.z_field is None:
            self.z_field = self._build_z_field(seed_id)
            if self.z_field is not None:
                save_path = self.align_z_field_path
                if save_path is None:
                    os.makedirs(self.output_root, exist_ok=True)
                    save_path = os.path.join(self.output_root,
                                             f"align_z_field_seed{seed_id}.npz")
                self.z_field.save(save_path)
                self.z_field_saved_path = save_path
                print(f"[align-z] field saved to {save_path}")
        return self.z_field

    def reset(self):
        """
        Reset tracer state before a new tracing run.
        """
        self.stack = []
        self.visited = set()
        self.graph = nx.DiGraph()
        self.all_merged_ids = set()
        self.voxel_counts = {}
        self.errors = []
        self.z_field = None
        print("NeuronTracer state has been reset.")

    def trace(self, seed_id, save_path, target_mip=2, resume=False):
        """
        Main tracing loop.
        """
        state_path = Path(save_path) / 'trace_state.json'
        if resume and state_path.is_file():
            state = json.loads(state_path.read_text())
            if int(state['seed']) != seed_id:
                raise ValueError('Resume state seed does not match')
            self.stack = [int(value) for value in state['pending']]
            self.visited = {int(value) for value in state['visited']}
            self.all_merged_ids = {int(value) for value in state['accepted']}
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.all_merged_ids)
            self.graph.add_edges_from((int(u), int(v)) for u, v in state['edges'])
            for error in state.get('errors', []):
                failed_id = int(error['segment'])
                self.visited.discard(failed_id)
                if failed_id not in self.stack:
                    self.stack.append(failed_id)
        else:
            self.stack.append(seed_id)
            self.all_merged_ids.add(seed_id)
            self.graph.add_node(seed_id)

        z_field = self._ensure_z_field(seed_id)

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
                # Stable per-segment prompt sampling, including after resumption.
                node_seed = (self.random_seed + current_id) % (2**32)
                np.random.seed(node_seed)
                random.seed(node_seed)
                endpoints, vectors, res, voxel_count = get_endpoints_vectors_precomputed(
                    current_id, target_mip, self.seg_path
                )
                if endpoints is None or vectors is None or res is None:
                    raise RuntimeError(f'No readable skeleton for segment {current_id}; check mip and skeleton metadata')
                print('finish endpoints')

                # print(f"Voxel count: {voxel_count}")
                self.voxel_counts[current_id] = int(voxel_count)

                if (current_id != seed_id) and is_messy_segment(endpoints, res,max_endpoints=40):
                    # print("Pruned segment; downstream tracing skipped.")
                    if current_id in self.all_merged_ids:
                        self.all_merged_ids.remove(current_id)
                    if current_id in self.graph:
                        self.graph.remove_node(current_id)
                    continue

                connections = get_neighbors(
                    self.seg_path, current_id, endpoints, vectors, res,
                    z_field=z_field
                )
                print('finish neighbors')

                if not connections:
                    # print("No geometric neighbors found.")
                    continue

                # contact coordinates are in the ALIGNED space when alignment
                # is enabled; CSVs are saved in original space
                csv_connections = connections
                if z_field is not None:
                    csv_connections = z_field.inverse_connections(connections)

                temp_slice_dir = os.path.join(save_path, f'temp_slices_{current_id}')
                temp_vis_dir = os.path.join(save_path, f'temp_vis_{current_id}')
                temp_slice3d_dir = os.path.join(save_path, f'temp_slices3d_{current_id}')
                temp_vis3d_dir = os.path.join(save_path, f'temp_vis3d_{current_id}')

                if not os.path.exists(temp_vis_dir): os.makedirs(temp_vis_dir)
                if not os.path.exists(temp_vis3d_dir): os.makedirs(temp_vis3d_dir)
                save_connections_to_csv(csv_connections, os.path.join(temp_vis_dir, 'neighbors.csv'))

                z_gap_conns = get_slices(
                    self.raw_path, self.seg_path, connections, temp_slice_dir,
                    temp_slice3d_dir, max_workers=self.slice_workers,
                    num_frames=self.sam2_num_frames, z_field=z_field
                )
                if z_field is not None:
                    z_gap_conns = z_field.inverse_connections(z_gap_conns)
                save_connections_to_csv(z_gap_conns, os.path.join(temp_vis3d_dir, 'neighbors_z.csv'))
                
                merged_candidates = find_merge_candidates(
                    self.sam_checkpoint, self.sam_cfg,
                    temp_slice_dir, temp_vis_dir, save_temp=False,
                    device=self.device, predictor=self.predictors.image
                )
                merged_candidates_3d = find_merge_candidates_3d_region(
                    self.sam_checkpoint, self.sam_cfg,
                    temp_slice3d_dir, temp_vis3d_dir, save_temp=False,
                    device=self.device, predictor=self.predictors.video
                )
                merged_candidates = list(set(merged_candidates).union(merged_candidates_3d))
                # print(f"SAM 2 verified connections: {merged_candidates}")

                for child_id_str in sorted(merged_candidates, key=int):
                    try:
                        child_id = int(child_id_str)
                    except ValueError:
                        # print(f"Warning: could not convert ID {child_id_str} to int; skipping.")
                        continue
                    if child_id <= 0:
                        continue
                    if child_id in self.visited and child_id not in self.all_merged_ids:
                        continue
                    
                    if child_id not in self.visited:
                        self.graph.add_edge(current_id, child_id)
                    self.all_merged_ids.add(child_id)

                    if child_id not in self.visited and child_id not in self.stack:
                        self.stack.append(child_id)
                        # print(f"Added to stack: {child_id}")

            except Exception as e:
                print(f"Error while processing seed {seed_id}, ID {current_id}: {e}")
                self.errors.append({'segment': str(current_id), 'error': str(e)})
            finally:
                atomic_json(state_path, {'schema_version': 1, 'seed': str(seed_id),
                    'pending': [str(value) for value in self.stack],
                    'visited': [str(value) for value in sorted(self.visited)],
                    'accepted': [str(value) for value in sorted(self.all_merged_ids)],
                    'edges': [[str(u), str(v)] for u, v in self.graph.edges], 'errors': self.errors})

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
        final_output = json_str

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
    
    signature_config = {key: value for key, value in config.items()
                        if key not in ('resume', 'debug_limit', 'max_workers', 'seed_ids', 'seed_list_file')}
    signature = hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()
    status_path = Path(save_path) / 'status.json'
    if os.path.exists(save_path):
        previous = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if previous.get('config_hash') != signature:
            raise FileExistsError(f'Existing results have no matching configuration: {save_path}. Choose a new output_root/suffix.')
        if previous.get('status') == 'complete':
            return seed_id, 'skipped'
        if not config.get('resume'):
            raise FileExistsError(f'Incomplete run at {save_path}; use --resume or a new output_root')
    atomic_json(status_path, {'status': 'running', 'config_hash': signature, 'seed': str(seed_id)})

    tracer = NeuronTracer(
        config['raw_path'], config['seg_path'], 
        sam_checkpoint=config['checkpoint_sam'], sam_cfg=config['model_cfg_sam'],
        voxel_threshold=config['voxel_threshold'],
        sam2_num_frames=config['sam2_num_frames'], 
        slice_workers=int(config.get('slice_workers', 8)),
        debug_limit=config['debug_limit'],
        gpu_id=config.get('gpu_id', '0'),
        device=config.get('device', 'auto'),
        align_z=config.get('align_z', False),
        align_z_margin_xy=config.get('align_z_margin_xy', 256),
        align_z_margin_z=config.get('align_z_margin_z', 64),
        align_z_estimation_mip=config.get('align_z_estimation_mip', 1),
        align_z_motion=config.get('align_z_motion', 'translation'),
        align_z_field=config.get('align_z_field'),
        align_z_max_z=config.get('align_z_max_z', 400),
        output_root=config.get('output_root', 'trace_results'),
        random_seed=config.get('random_seed', 42),
    )
    
    try:
        final_ids = tracer.trace(seed_id, save_path, target_mip=config['target_mip'], resume=config.get('resume', False))
        tracer.save_results(seed_id, save_path)
        status = 'failed' if tracer.errors else 'limited' if tracer.stack else 'complete'
        atomic_json(status_path, {'status': status, 'config_hash': signature, 'seed': str(seed_id),
                    'accepted_count': len(final_ids), 'pending_count': len(tracer.stack), 'errors': tracer.errors})
        return seed_id, status
    except Exception as e:
        print(f"Seed {seed_id} failed: {e}")
        atomic_json(status_path, {'status': 'failed', 'config_hash': signature, 'seed': str(seed_id), 'error': str(e)})
        return seed_id, f"error: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Probe-EM tracing.")
    parser.add_argument(
        "--config",
        default="configs/config.json",
        help="Path to a JSON config file. Copy configs/config.example.json to configs/config.json first.",
    )
    parser.add_argument('--resume', action='store_true', help='Continue a matching incomplete run')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config['resume'] = True
    validate_config(config)

    device = resolve_device(config.get("device", "auto"), config.get("gpu_id", "0"))
    # Device indices refer to the already-visible CUDA devices; do not change
    # CUDA_VISIBLE_DEVICES after torch has inspected/initialized the backend.
    print(f">>> Probe-EM device: {device}")

    target_ids = load_seed_ids(config)
    if not target_ids:
        raise ValueError("No seed IDs were provided. Set seed_ids or seed_list_file in the config.")

    max_workers = int(config.get("max_workers", 4))

    print(f"Starting Probe-EM tracing with {max_workers} workers")
    print(f"SAM 2 checkpoint: {config['checkpoint_sam']}")
    t1 = time.time()

    failed = False
    # spawn avoids forking a process after CUDA/MPS has been initialized.
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(run_one_seed, sid, config): sid for sid in target_ids}

        for future in as_completed(futures):
            sid = futures[future]
            try:
                sid, status = future.result()
                print(f">>> Finished seed {sid}: {status}")
                failed |= status == 'failed' or status.startswith('error:')
            except Exception as e:
                print(f">>> Failed seed {sid}: {e}")
                failed = True

    t2 = time.time()
    print(f"\nTotal runtime: {t2 - t1:.4f} seconds")
    sys.exit(1 if failed else 0)
