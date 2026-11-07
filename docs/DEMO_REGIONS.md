# LC-NE test windows

The five process windows are supplied in `demo/lc_ne_process_windows.zip` with
the [LC-NE test release](LC_NE_TEST.md). Each contains a 512 x 512 x 63 raw
cutout at 8 x 8 x 30 nm, instance labels, a binary target mask, and a local
realignment field. Coordinates refer to the cropped, SOFIMA-aligned dataset.

| Neuron | Approximate distance from soma (um) | XYZ voxel offset | Inclusive Z range |
| --- | ---: | --- | --- |
| 30001 | 11.0 | 6731, 3696, 1117 | 1117-1179 |
| 30002 | 15.6 | 5827, 4493, 653 | 653-715 |
| 30003 | 23.2 | 7532, 4067, 485 | 485-547 |
| 30004 | 29.9 | 8228, 4873, 149 | 149-211 |
| 30005 | 38.0 | 4242, 1919, 1045 | 1045-1107 |

All five use the same local alignment method and parameters, with a separate
field fitted to each region. The fields preserve the whole-volume coordinate
trend and fix each window's middle section. See [local realignment](../probe-em/README_alignment.md).

`demo/windows.json` opens the overview plus these windows. For the five windows
alone, use `demo/windows.roi-only.json`. The [window guide](../probe-em/README_windows.md)
contains the launch command. The supplied masks are reference annotations;
interactive predictions are saved separately.
