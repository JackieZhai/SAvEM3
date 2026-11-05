# LC-NE test

**LC-NE test** is the example dataset for SAEM²-SAEvM³ evaluation and interactive
Neuroglancer review. It contains corresponding zebrafish electron-microscopy
volumes derived from the `zfishlc` collection. The dataset license is Apache-2.0.

The dataset is prepared and distributed separately from this source repository.
A public download link will be added when the dataset release is available.
Model checkpoints are separate downloads; see the repository's
[data and pretrained weights](../README.md#data-and-pretrained-weights) instructions.

## Dataset contents

| Directory | Contents |
| --- | --- |
| `zfishlc_raw` | Raw electron-microscopy images |
| `zfishlc_mito` | Mitochondrial context intensities |
| `zfishlc_syn` | Synaptic context intensities |
| `zfishlc_ves` | Vesicle context intensities |
| `zfishlc_lc` | Neuron instance labels, including IDs 30001-30005 |
| `zfishlc_lcpre` | Additional supplied label volume |
| `zfishlc_lcsyn` | Additional supplied label volume |

The volumes use Neuroglancer precomputed layout. Keep each `info` file with its
scale and mesh directories. Use metadata for bounds, voxel offsets, resolutions
and codecs. Corresponding volumes share physical coordinates; their chunk sizes
and available scales can differ. Native raw resolution is 4 x 4 x 30 nm, and the
process-window demonstration uses 8 x 8 x 30 nm imagery.

Mito/syn/ves are grayscale context images, not instance IDs or calibrated
probabilities. The auxiliary `lcpre` and `lcsyn` annotation semantics should be
taken from the dataset release metadata rather than inferred from filenames.

## Neuroglancer review

Use the [test-window browser](../probe-em/README_windows.md) with a local manifest
pointing to the downloaded volumes and cached process windows. Its selector
contains **LC-NE test overview** and five process windows at different distances
from the annotated soma regions.

The overview displays raw, neuron meshes and process-window bounding boxes.
Mito/syn/ves are available only in this overview and start hidden. The overview
uses original physical coordinates and leaves local registration disabled.

Process windows support positive/negative points, explicit SAM2 image prediction,
cross-section propagation and retained review state. Local registration defaults
on, **A** toggles it, and **D/F** moves between sections. Meshes use near-isotropic
sampling; `showSlices=false` remains the default. See the
[interactive guide](../probe-em/README_interactive.md) for controls.

## Evaluation scope

The [bounded evaluation guide](../probe-em/README_local_gt.md) records original,
ECC and SOFIMA comparisons and the measured process-window results. Selected
neuron annotations are not exhaustive background truth. Generic SAM2 weights
and successful interactive propagation do not establish whole-neuron accuracy
or quantitative reproduction of the paper.
