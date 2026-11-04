# SAEM2 image and bounded 3D trials

This experiment reuses the same five cached zebrafish regions and seeds as the
[tracing investigation](README_trace_quality.md). It does not replace Probe-EM
results or modify online labels. Image predictions are compared first, followed
by small-slab 3D reconstruction from SAEM2 membrane probabilities.

## Checkpoint and inference

A historical ViT-B `epoch_11.pth` was found in a local backup. All 152 decoder
tensors load strictly, including `mem_token` and `mem_mlp`. They equal the decoder
in the accompanying combined checkpoint. Its frozen image/prompt encoders also
exactly match the base SAM and local HQ-SAM encoders used for this trial. The
membrane head is not randomly initialized or substituted with ordinary HQ-SAM.

The exact training data/run remain **unconfirmed**. Nearby SNEMI, CREMI and AxonEM
folders do not establish the training split or model quality. This ViT-B backup
must not be presented as the paper's ViT-H checkpoint. No training is performed.

For each ROI, the largest seed cross-section and four sections on either side
are selected, clipped to available data. The initial run has 45 sections: 33 have
a seed prompt and 12 do not. SAEM2 ViT-B, HQ-SAM ViT-B and generic SAM2 Hiera Tiny
receive the same single interior point in the largest existing seed component.
Unprompted sections are not scored as model failures. Inference is CPU float32,
with two Torch threads. Existing labels are not ground truth.

`saem2/predictor.py` caches encoder features and follows the historical evaluation
convention: single SAM + HQ logits, thresholded at zero. The membrane head is
separate, not a confidence score. Its full-image-box prediction uses no labels,
including on unprompted sections. SAM's standard aspect-preserving resize/padding
is inverted before native-grid export. This is not a fixed-physical-FOV ablation.

Optimized local registration defaults ON. To test original-space inference, use
`--no-align-z` and a new output directory. Alignment is fixed during inference;
these small-slab viewers do not have a display-only A toggle.

## Membrane postprocessing

Two baselines are retained:

1. `membrane-seed`: select the 4-connected probability<0.5 component at each slice
   prompt, fill enclosed holes, and clip to the valid ROI. This naive baseline
   leaks through membrane gaps; hole filling can absorb other enclosed cells when
   the component reaches background. It is a failure diagnostic, not recommended
   reconstruction.
2. `prepare-postprocess` and `repro/saem2/postprocess_trial.py`: slice watershed
   (threshold 0.25, seed sigma 2 native pixels) plus existing local 3D multicut
   (beta 0.5, no lifted edges). Registration padding is excluded. Only the central
   point selects an output instance after graph optimization. No old masks or
   other slice prompts enter the graph or its costs.

The latter predicts new voxels/instances, unlike Probe-EM ID joining. It uses no
distilled SAvEM3 student or SAM2 video memory. All selected objects reach a slab
end; none is a completed neuron. There is no proofread ground truth.

Observed behavior is mixed. Direct point masks can select intracellular objects,
and some generic SAM2/HQ masks cover very large areas. Region 4's membrane/multicut
example follows a more plausible elongated profile on the inspected section; the
original example still selects an inner structure. These are visual observations,
not measured biological accuracy. Do not promote this trial to the default tracer
based on agreement with existing labels or appealing meshes.

## Reproduce

From the repository root in the Probe-EM environment:

~~~bash
python -m pip install -r probe-em/requirements-saem2.txt
python probe-em/scripts/try_saem2.py run \
  --output probe-em/demo_session_saem2_trial \
  --decoder-checkpoint /absolute/path/to/epoch_11.pth \
  --hq-checkpoint /absolute/path/to/sam_hq_vit_b.pth \
  --sam2-checkpoint /absolute/path/to/sam2_hiera_tiny.pt \
  --model-type vit_b --slices 9
python probe-em/scripts/try_saem2.py membrane-seed --output probe-em/demo_session_saem2_trial
python probe-em/scripts/try_saem2.py prepare-postprocess --output probe-em/demo_session_saem2_trial
~~~

Use the existing ELF environment for the graph step. The wrapper also configures
the macOS C++ headers needed by the broader WaterZ tests:

~~~bash
./repro/env/run_postprocess.sh repro/saem2/postprocess_trial.py --input probe-em/demo_session_saem2_trial
~~~

Back in the Probe-EM environment:

~~~bash
python probe-em/scripts/try_saem2.py summarize --output probe-em/demo_session_saem2_trial
python probe-em/scripts/try_saem2.py serve --output probe-em/demo_session_saem2_trial --open-browser
~~~

`--cases original region_1 ...` selects fixed existing cases, not different online
regions. Raw data remain from
`precomputed://https://ng.zebrafish.digital-brain.cn/srv/raw/`.
The successful initial run is cached locally in `demo_session_saem2_v2/`.
Earlier failed setup attempts remain separate and explicitly marked failed.

## Viewer, outputs and tests

The default prediction, when available, is `saem2_membrane_3d`. Other layers show
direct SAEM2/SAM2/HQ slice masks, the input seed (not GT), SAEM2 membranes and the
naive leaking component baseline. Reference layers disable picking/hover previews.
`showSlices` defaults false; 8x8x30 nm data use 32x32x30 nm display meshes while
2D images/labels stay native. Only nine sections per region are included here.

`manifest.json` records weight SHA-256 values, cached copies and scope. Each case
has lossless `arrays.npz`, per-section prompts/diagnostics, `postprocess_inputs.npz`
(ZYX), `saem2_multicut/` labels/selection/status, and `comparison.png`. Agreement
fields are named `not_accuracy`. Image summaries use per-slice medians; multicut
summaries use 3D voxel agreement. These are not interchangeable metrics.

~~~bash
SAEM2_CHECKPOINT=/absolute/path/to/epoch_11.pth \
SAVEM3_HQ_CHECKPOINT=/absolute/path/to/sam_hq_vit_b.pth \
  python -m pytest -q tests/test_saem2_inference.py
./repro/env/run_postprocess.sh -m pytest -q tests/test_saem2_postprocess.py tests/test_reproduction.py
~~~

Checkpoint compatibility is not evidence of training provenance or accuracy.
Confirm the intended EM-trained weights and proofread membrane, organelle and
neurite boundaries before expanding to full-volume reconstruction.
