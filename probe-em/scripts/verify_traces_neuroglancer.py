"""Interactive Neuroglancer tool for offline tracing verification.

The tool loads raw and segmentation volumes in Neuroglancer, displays traced
segment sets produced by Probe-EM, and lets a user save corrected segment sets
as ground-truth labels.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import neuroglancer
from cloudvolume import CloudVolume
from cloudvolume.server import view


DEFAULTS = {
    "bind_address": "0.0.0.0",
    "port": 0,
    "host_ip": "127.0.0.1",
    "raw_path": None,
    "seg_path": None,
    "raw_source": None,
    "seg_source": None,
    "results_folder": "trace_results",
    "gt_folder": "gt_labels",
    "data_port_start": 10023,
    "data_bind_address": "0.0.0.0",
    "home_position": None,
}


def load_config_defaults(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Verification config must be a JSON object.")
    allowed = set(DEFAULTS)
    return {k: v for k, v in data.items() if k in allowed}


def parse_home_position(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError("home_position must contain exactly three values.")
        return [int(x) for x in value]
    parts = [x.strip() for x in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Use x,y,z for --home-position.")
    return [int(x) for x in parts]


def ensure_cloudpath(cv_path):
    if cv_path.startswith(("file://", "s3://", "gs://", "http://", "https://")):
        return cv_path
    return Path(cv_path).resolve().as_uri()


def cloudvolume_process(cv_path, port, data_bind_address):
    cloudpath = ensure_cloudpath(cv_path)
    print(f"Starting CloudVolume server: {cloudpath} on port {port}")
    vol = CloudVolume(cloudpath)
    view(vol.cloudpath, hostname=data_bind_address, port=port)


def read_segment_file(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    if '"segments"' in content:
        json_text = content if content.startswith("{") else "{" + content + "}"
        data = json.loads(json_text)
        return [int(x) for x in data.get("segments", [])]

    segment_ids = []
    for line in content.splitlines():
        token = line.strip().strip(",").strip('"')
        if token.isdigit():
            segment_ids.append(int(token))
    return segment_ids


def write_segment_file(path, segment_ids):
    data = {"segments": [str(x) for x in sorted(segment_ids)]}
    content = json.dumps(data, indent=2).strip().strip("{}").strip()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class VerificationToolNamespace:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.results_folder = args.results_folder
        self.gt_folder = args.gt_folder
        self.home_position = args.home_position

    def _set_status_message(self, msg, position=None):
        """Show a short status message and add a yellow annotation marker."""
        short_msg = msg.split("\n")[0]
        try:
            with self.viewer.config_state.txn() as s:
                s.status_message = f"{short_msg} (click yellow marker for details)"
        except Exception:
            pass

        print(f"[Info]: {msg.replace(chr(10), ' ')}")

        try:
            with self.viewer.txn() as s:
                if "info" not in s.layers:
                    s.layers["info"] = neuroglancer.LocalAnnotationLayer(
                        dimensions=s.dimensions,
                        shader="""
void main() {
  setColor(vec4(1.0, 1.0, 0.0, 1.0));
  setPointMarkerSize(10.0);
}
""",
                    )

                layer = s.layers["info"]
                if position is None:
                    try:
                        pos = s.position.voxel_coordinates
                    except Exception:
                        pos = s.position
                    if pos is None:
                        return
                else:
                    pos = position

                layer.annotations.append(
                    neuroglancer.PointAnnotation(
                        id=f"verification_log_{time.time()}",
                        point=pos,
                        description=msg,
                    )
                )
        except Exception as e:
            print(f"Warning: failed to set annotation message: {e}")

    def action_reset(self, action_state):
        """Clear selected segments and log annotations. Hotkey: c."""
        with self.viewer.txn() as s:
            if "segmentation" in s.layers:
                s.layers["segmentation"].segments = []
            if "info" in s.layers:
                s.layers["info"].annotations = []
        self._set_status_message("Display reset.")

    def action_query_result(self, action_state):
        """Load tracing results for the segment under the cursor. Hotkey: q."""
        mouse_pos = action_state.mouse_voxel_coordinates
        target_id = self._selected_segment_id(action_state)
        if not target_id:
            self._set_status_message("No segmentation ID selected.", position=mouse_pos)
            return

        self._set_status_message(f"Querying tracing results for ID {target_id}...", position=mouse_pos)

        if not os.path.exists(self.results_folder):
            self._set_status_message(f"Results folder not found: {self.results_folder}", position=mouse_pos)
            return

        found_folders = sorted(
            d for d in os.listdir(self.results_folder) if d.startswith(f"{target_id}_results")
        )
        if not found_folders:
            self._set_status_message(
                f"No tracing results found for ID {target_id} in {self.results_folder}.",
                position=mouse_pos,
            )
            return

        target_folder = os.path.join(self.results_folder, found_folders[0])
        id_list = []
        for fname in (f"trace_{target_id}_ng_segments.txt", f"trace_{target_id}_ids.txt"):
            fpath = os.path.join(target_folder, fname)
            if os.path.exists(fpath):
                try:
                    id_list = read_segment_file(fpath)
                    break
                except Exception as e:
                    print(f"Failed to read {fpath}: {e}")

        if id_list:
            self._update_viewer(id_list)
            self._set_status_message(
                f"Loaded tracing results for ID {target_id}.\nTotal segments: {len(id_list)}",
                position=mouse_pos,
            )
        else:
            self._set_status_message(f"Result file is empty or could not be parsed for ID {target_id}.", position=mouse_pos)

    def action_save_gt(self, action_state):
        """Save currently displayed segments as a ground-truth label. Hotkey: g."""
        mouse_pos = action_state.mouse_voxel_coordinates
        id_list = []
        with self.viewer.txn() as s:
            if "segmentation" in s.layers:
                id_list = [int(x) for x in s.layers["segmentation"].segments]

        if not id_list:
            self._set_status_message("No selected segments to save.", position=mouse_pos)
            return

        mouse_id = self._selected_segment_id(action_state)
        if not mouse_id:
            self._set_status_message(
                "Hover over the main segment ID before pressing 'g'.",
                position=mouse_pos,
            )
            return

        os.makedirs(self.gt_folder, exist_ok=True)
        save_path = os.path.join(self.gt_folder, f"{mouse_id}_gt_segments.txt")

        try:
            write_segment_file(save_path, id_list)
            self._set_status_message(
                f"Saved ground-truth labels.\nID: {mouse_id}\nSegments: {len(id_list)}\nFile: {save_path}",
                position=mouse_pos,
            )

            with self.viewer.txn() as s:
                if "segmentation" in s.layers:
                    s.layers["segmentation"].segments = []
        except Exception as e:
            self._set_status_message(f"Failed to save ground-truth labels: {e}", position=mouse_pos)

    def action_show_all_gt(self, action_state):
        """Display all already annotated ground-truth segment IDs. Hotkey: w."""
        mouse_pos = action_state.mouse_voxel_coordinates
        self._set_status_message("Scanning ground-truth labels...", position=mouse_pos)
        if not os.path.exists(self.gt_folder):
            self._set_status_message(f"Ground-truth folder not found: {self.gt_folder}", position=mouse_pos)
            return

        all_gt_ids = set()
        count_files = 0
        for filename in os.listdir(self.gt_folder):
            if not filename.endswith("_gt_segments.txt"):
                continue
            count_files += 1
            fpath = os.path.join(self.gt_folder, filename)
            try:
                all_gt_ids.update(read_segment_file(fpath))
            except Exception:
                continue

        if all_gt_ids:
            self._update_viewer(list(all_gt_ids))
            self._set_status_message(
                f"Loaded {count_files} annotation files with {len(all_gt_ids)} segment IDs.",
                position=mouse_pos,
            )
        else:
            self._set_status_message("No valid ground-truth labels found.", position=mouse_pos)

    def action_jump_to_home(self, action_state):
        """Jump to the configured home position. Hotkey: b."""
        if self.home_position is None:
            self._set_status_message("No home position configured. Use --home-position x,y,z.")
            return
        with self.viewer.txn() as s:
            s.position = self.home_position
        self._set_status_message(f"Jumped to home position: {self.home_position}", position=self.home_position)

    def action_load_single_gt(self, action_state):
        """Load saved ground-truth labels for the selected ID. Hotkey: t."""
        mouse_pos = action_state.mouse_voxel_coordinates
        target_id = self._selected_segment_id(action_state)
        if not target_id:
            self._set_status_message("No segmentation ID selected.", position=mouse_pos)
            return

        if not os.path.exists(self.gt_folder):
            self._set_status_message(f"Ground-truth folder not found: {self.gt_folder}", position=mouse_pos)
            return

        gt_path = os.path.join(self.gt_folder, f"{target_id}_gt_segments.txt")
        if not os.path.exists(gt_path):
            self._set_status_message(f"No saved annotation found for ID {target_id}.", position=mouse_pos)
            return

        try:
            id_list = read_segment_file(gt_path)
            if id_list:
                self._update_viewer(id_list)
                self._set_status_message(
                    f"Loaded saved annotation for ID {target_id} with {len(id_list)} segments.",
                    position=mouse_pos,
                )
            else:
                self._set_status_message(f"Saved annotation for ID {target_id} is empty.", position=mouse_pos)
        except Exception as e:
            self._set_status_message(f"Failed to read saved annotation: {e}", position=mouse_pos)

    def action_show_traceable_ids(self, action_state):
        """Display all seed IDs with generated tracing results. Hotkey: a."""
        mouse_pos = action_state.mouse_voxel_coordinates
        self._set_status_message("Scanning tracing result folders...", position=mouse_pos)

        if not os.path.exists(self.results_folder):
            self._set_status_message(f"Results folder not found: {self.results_folder}", position=mouse_pos)
            return

        traceable_ids = set()
        try:
            for dname in os.listdir(self.results_folder):
                dpath = os.path.join(self.results_folder, dname)
                if os.path.isdir(dpath) and "_results" in dname:
                    prefix = dname.split("_results")[0]
                    if prefix.isdigit():
                        traceable_ids.add(int(prefix))
        except Exception as e:
            self._set_status_message(f"Failed to scan result folder: {e}", position=mouse_pos)
            return

        if traceable_ids:
            self._update_viewer(list(traceable_ids))
            self._set_status_message(
                f"Loaded {len(traceable_ids)} traceable seed IDs.",
                position=mouse_pos,
            )
        else:
            self._set_status_message("No tracing results found.", position=mouse_pos)

    @staticmethod
    def _selected_segment_id(action_state):
        selected_values = action_state.selected_values
        if "segmentation" not in selected_values:
            return None
        value = selected_values["segmentation"]
        if value is None:
            return None
        return int(value.value) if hasattr(value, "value") else int(value)

    def _update_viewer(self, new_ids):
        with self.viewer.txn() as s:
            if "segmentation" in s.layers:
                current = set(int(x) for x in s.layers["segmentation"].segments)
                current.update(int(x) for x in new_ids)
                s.layers["segmentation"].segments = list(current)


def start_data_servers(raw_path, seg_path, port_start, data_bind_address):
    cv_paths = [raw_path, seg_path]
    port_list = list(range(port_start, port_start + len(cv_paths)))
    pool = ProcessPoolExecutor(max_workers=len(cv_paths))
    ports_map = {"raw": port_list[0], "seg": port_list[1]}
    for path, port in zip(cv_paths, port_list):
        pool.submit(cloudvolume_process, path, port, data_bind_address)
    return pool, ports_map


def build_parser(defaults):
    parser = argparse.ArgumentParser(
        description="Interactive Neuroglancer tool for offline tracing verification."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config for this tool.")
    parser.add_argument("--bind-address", "--bind_address", dest="bind_address", default=defaults["bind_address"])
    parser.add_argument("--port", type=int, default=defaults["port"])
    parser.add_argument("--host-ip", "--host_ip", dest="host_ip", default=defaults["host_ip"])
    parser.add_argument("--raw-path", "--raw_path", dest="raw_path", default=defaults["raw_path"])
    parser.add_argument("--seg-path", "--seg_path", dest="seg_path", default=defaults["seg_path"])
    parser.add_argument("--raw-source", "--raw_source", dest="raw_source", default=defaults["raw_source"])
    parser.add_argument("--seg-source", "--seg_source", dest="seg_source", default=defaults["seg_source"])
    parser.add_argument("--results-folder", "--results_folder", dest="results_folder", default=defaults["results_folder"])
    parser.add_argument("--gt-folder", "--gt_folder", dest="gt_folder", default=defaults["gt_folder"])
    parser.add_argument("--data-port-start", "--data_port_start", dest="data_port_start", type=int, default=defaults["data_port_start"])
    parser.add_argument("--data-bind-address", "--data_bind_address", dest="data_bind_address", default=defaults["data_bind_address"])
    parser.add_argument("--home-position", "--home_position", dest="home_position", type=parse_home_position, default=defaults["home_position"])
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    known_args, _ = config_parser.parse_known_args()

    defaults = dict(DEFAULTS)
    defaults.update(load_config_defaults(known_args.config))
    defaults["home_position"] = parse_home_position(defaults.get("home_position"))

    parser = build_parser(defaults)
    return parser.parse_args()


def main():
    args = parse_args()
    data_pool = None

    if args.raw_source or args.seg_source:
        if not (args.raw_source and args.seg_source):
            raise ValueError("Provide both --raw-source and --seg-source, or provide --raw-path and --seg-path.")
        raw_source = args.raw_source
        seg_source = args.seg_source
    else:
        if not (args.raw_path and args.seg_path):
            raise ValueError("Provide --raw-path and --seg-path, or provide --raw-source and --seg-source.")
        data_pool, data_ports = start_data_servers(
            args.raw_path,
            args.seg_path,
            args.data_port_start,
            args.data_bind_address,
        )
        raw_source = f"precomputed://http://{args.host_ip}:{data_ports['raw']}"
        seg_source = f"precomputed://http://{args.host_ip}:{data_ports['seg']}"

    neuroglancer.set_server_bind_address(bind_address=args.bind_address, bind_port=args.port)
    viewer = neuroglancer.Viewer()

    with viewer.txn() as s:
        s.layers["image"] = neuroglancer.ImageLayer(source=raw_source)
        s.layers["segmentation"] = neuroglancer.SegmentationLayer(source=seg_source, segments=[])
        s.layout = "4panel"

    tool = VerificationToolNamespace(viewer, args)

    viewer.actions.add("reset", tool.action_reset)
    viewer.actions.add("query", tool.action_query_result)
    viewer.actions.add("save-gt", tool.action_save_gt)
    viewer.actions.add("show-all-gt", tool.action_show_all_gt)
    viewer.actions.add("jump-home", tool.action_jump_to_home)
    viewer.actions.add("load-single-gt", tool.action_load_single_gt)
    viewer.actions.add("show-traceable", tool.action_show_traceable_ids)

    with viewer.config_state.txn() as s:
        s.input_event_bindings.viewer["keyc"] = "reset"
        s.input_event_bindings.viewer["keyq"] = "query"
        s.input_event_bindings.viewer["keyg"] = "save-gt"
        s.input_event_bindings.viewer["keyw"] = "show-all-gt"
        s.input_event_bindings.viewer["keyb"] = "jump-home"
        s.input_event_bindings.viewer["keyt"] = "load-single-gt"
        s.input_event_bindings.viewer["keya"] = "show-traceable"

    print(f"\nNeuroglancer viewer: {viewer}")
    print("=" * 72)
    print("Hotkeys:")
    print("  q: Load tracing results for the segment under the cursor.")
    print("  a: Highlight all seed IDs that have tracing result folders.")
    print("  t: Load saved ground-truth labels for the selected ID.")
    print("  w: Display all already annotated ground-truth segment IDs.")
    print("  g: Save the current segment set as ground truth, then clear display.")
    print("  b: Jump to --home-position, if configured.")
    print("  c: Clear selected segments and log annotations.")
    print("=" * 72)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        if data_pool is not None:
            data_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
