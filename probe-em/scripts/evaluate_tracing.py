import os
import json
import argparse
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from tqdm import tqdm

class TracingEvaluator:
    def __init__(self, seg_path, gt_folder, merge_threshold=10):
        self.seg_path = seg_path if seg_path.startswith('file://') else 'file://' + seg_path
        self.gt_folder = gt_folder
        self.merge_threshold = merge_threshold
        self.vol = CloudVolume(self.seg_path, parallel=False, fill_missing=True)
        
        self.node_count_cache = {}
        
        self.gt_db = self._load_all_gts()

    def _load_segments_txt(self, filepath):
        """
        Load segment IDs from either a Neuroglancer-style segments file or a
        plain text file with one ID per line.
        """
        if not os.path.exists(filepath):
            return set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('"segments"'):
                    json_data = json.loads("{" + content + "}")
                    ids = [int(x) for x in json_data['segments']]
                else:
                    ids = []
                    for line in content.splitlines():
                        line = line.strip().strip(',').strip('"')
                        if line and line.isdigit():
                            ids.append(int(line))
                return set(ids)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return set()

    def _load_all_gts(self):
        """
        Load all ground-truth neuron segment sets from gt_folder.
        """
        print(f">>> Loading ground-truth segment sets from: {self.gt_folder}")
        db = {} 
        if not os.path.exists(self.gt_folder):
            print(f"Warning: ground-truth folder does not exist: {self.gt_folder}")
            return db

        for filename in os.listdir(self.gt_folder):
            if filename.endswith('_gt_segments.txt'):
                try:
                    seed_id = int(filename.split('_')[0])
                    filepath = os.path.join(self.gt_folder, filename)
                    member_ids = self._load_segments_txt(filepath)
                    db[seed_id] = member_ids
                except:
                    continue
        print(f"--- Loaded {len(db)} ground-truth neurons.")
        return db

    def get_node_count(self, seg_id):
        """
        Get the skeleton node count for one segment ID.
        """
        if seg_id in self.node_count_cache:
            return self.node_count_cache[seg_id]
        
        try:
            skel = self.vol.skeleton.get(seg_id)
            count = len(skel.vertices) if skel else 0
        except Exception:
            count = 0
            
        self.node_count_cache[seg_id] = count
        return count

    def calculate_neuron_metrics(self, seed_id, result_ids, exclude_seed=True):
        """
        Calculate tracing metrics for one seed neuron.
        """
        target_gt_set = None
        if seed_id in self.gt_db:
            target_gt_set = self.gt_db[seed_id]
        else:
            for gid, members in self.gt_db.items():
                if seed_id in members:
                    target_gt_set = members
                    break
        
        if target_gt_set is None:
            return None

        result_set = set(result_ids)

        valid_gt_list = [gid for gid in target_gt_set if self.get_node_count(gid) >= self.merge_threshold]
        valid_result_list = [rid for rid in result_set if self.get_node_count(rid) >= self.merge_threshold]
        
        if exclude_seed:
            valid_gt_list = [gid for gid in valid_gt_list if gid != seed_id]
            valid_result_list = [rid for rid in valid_result_list if rid != seed_id]

        filtered_gt_set = set(valid_gt_list)
        filtered_result_set = set(valid_result_list)

        tp_ids = filtered_result_set.intersection(filtered_gt_set)
        tp_nodes = sum(self.get_node_count(tid) for tid in tp_ids)
        
        total_gt_nodes = sum(self.get_node_count(gid) for gid in filtered_gt_set)
        recall = tp_nodes / total_gt_nodes if total_gt_nodes > 0 else 1.0

        extra_ids = filtered_result_set - filtered_gt_set
        extra_nodes = sum(self.get_node_count(eid) for eid in extra_ids)
        
        total_traced_nodes = tp_nodes + extra_nodes
        precision = tp_nodes / total_traced_nodes if total_traced_nodes > 0 else 1.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'tp_nodes': tp_nodes,
            'extra_nodes': extra_nodes,
            'total_gt_nodes': total_gt_nodes
        }

    def run_evaluation(
        self,
        results_base_dir,
        suffix="sam",
        exclude_seed=True,
        excel_name="evaluation_results.xlsx",
        output_dir="quantification_results",
    ):
        """
        Run evaluation and save an Excel report.
        """
        print(f"\n>>> Analyzing tracing results in: {results_base_dir}")
        print(f">>> Result folder suffix: {suffix}")
        print(f">>> Exclude seed segment: {exclude_seed}")
        stats = []

        for seed_id in tqdm(self.gt_db.keys(), desc="Evaluating"):
            test_folder = os.path.join(results_base_dir, f"{seed_id}_results_{suffix}")
            target_file = os.path.join(test_folder, f"trace_{seed_id}_ng_segments.txt")
            
            if os.path.exists(target_file):
                result_ids = self._load_segments_txt(target_file)
                metrics = self.calculate_neuron_metrics(seed_id, result_ids, exclude_seed=exclude_seed)
                if metrics:
                    stats.append({
                        'Target ID': seed_id,
                        'Recall': metrics['recall'],
                        'Precision': metrics['precision'],
                        'F1 Score': metrics['f1'],
                        'Extra Nodes': metrics['extra_nodes'],
                        'GT Nodes': metrics['total_gt_nodes'],
                        'TP Nodes': metrics['tp_nodes']
                    })

        print("\n" + "="*110)
        print(f"{'Target ID':<12} | {'Recall':<12} | {'Precision':<12} | {'F1 Score':<12} | {'ExtraNodes':<12}")
        print("-" * 110)
        
        for s in stats:
            print(f"{s['Target ID']:<12} | {s['Recall']*100:>10.2f}% | {s['Precision']*100:>10.2f}% | {s['F1 Score']*100:>10.2f}% | {s['Extra Nodes']:<12}")
        
        if stats:
            avg_recall = np.mean([s['Recall'] for s in stats])
            avg_precision = np.mean([s['Precision'] for s in stats])
            avg_f1 = np.mean([s['F1 Score'] for s in stats])
            std_recall = np.std([s['Recall'] for s in stats])
            std_precision = np.std([s['Precision'] for s in stats])
            std_f1 = np.std([s['F1 Score'] for s in stats])
            print("-" * 110)
            print(f"{'AVERAGE':<12} | {avg_recall*100:>10.2f}% | {avg_precision*100:>10.2f}% | {avg_f1*100:>10.2f}% | (Count: {len(stats)})")
            print(f"{'STD':<12} | {std_recall*100:>10.2f}% | {std_precision*100:>10.2f}% | {std_f1*100:>10.2f}% |")
            print(f"{'MEAN±STD':<12} | {avg_recall*100:.2f}±{std_recall*100:.2f} | {avg_precision*100:.2f}±{std_precision*100:.2f} | {avg_f1*100:.2f}±{std_f1*100:.2f} |")
            
            os.makedirs(output_dir, exist_ok=True)
            excel_path = os.path.join(output_dir, excel_name)
            
            df = pd.DataFrame(stats)
            avg_row = {
                'Target ID': 'AVERAGE',
                'Recall': avg_recall,
                'Precision': avg_precision,
                'F1 Score': avg_f1,
                'Extra Nodes': np.mean([s['Extra Nodes'] for s in stats]),
                'GT Nodes': np.sum([s['GT Nodes'] for s in stats]),
                'TP Nodes': np.sum([s['TP Nodes'] for s in stats])
            }
            std_row = {
                'Target ID': 'STD',
                'Recall': std_recall,
                'Precision': std_precision,
                'F1 Score': std_f1,
                'Extra Nodes': np.std([s['Extra Nodes'] for s in stats]),
                'GT Nodes': '',
                'TP Nodes': ''
            }
            df = pd.concat([df, pd.DataFrame([avg_row]), pd.DataFrame([std_row])], ignore_index=True)
            
            df.to_excel(excel_path, index=False)
            print(f"\n>>> Saved results to: {excel_path}")
        else:
            print("No matching tracing result files were found.")
        print("="*110 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Probe-EM tracing results.")
    parser.add_argument("--seg-path", required=True, help="Path or CloudVolume URL for the segmentation volume.")
    parser.add_argument("--gt-folder", required=True, help="Folder containing *_gt_segments.txt files.")
    parser.add_argument("--results-dir", required=True, help="Base directory containing per-seed tracing result folders.")
    parser.add_argument("--suffix", default="sam", help="Result folder suffix, e.g. seed_results_<suffix>.")
    parser.add_argument("--merge-threshold", type=int, default=0, help="Minimum skeleton node count for a segment to be evaluated.")
    parser.add_argument("--include-seed", action="store_true", help="Include the seed segment when computing metrics.")
    parser.add_argument("--output-dir", default="quantification_results", help="Directory for the Excel report.")
    parser.add_argument("--excel-name", default="evaluation_results.xlsx", help="Output Excel filename.")
    args = parser.parse_args()

    evaluator = TracingEvaluator(args.seg_path, args.gt_folder, merge_threshold=args.merge_threshold)
    evaluator.run_evaluation(
        args.results_dir,
        suffix=args.suffix,
        exclude_seed=not args.include_seed,
        excel_name=args.excel_name,
        output_dir=args.output_dir,
    )
