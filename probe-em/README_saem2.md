# SAEM² image backend

The interactive viewer can use SAEM² for point-guided image prediction. Install
`probe-em/requirements-saem2.txt`, then provide a full HQ-SAM checkpoint and a
trained SAEM² decoder with the same ViT architecture:

```bash
python probe-em/scripts/demo.py \
  --local-dir /data/LC-NE-test/demo/process_windows/neuron_30001/cache \
  --seed 30001 --interactive --prompt-backend saem2 \
  --prompt-model-type vit_b --prompt-checkpoint /models/sam_hq_vit_b.pth \
  --prompt-decoder /models/saem2_decoder.pth --prompt-device auto \
  --align-field /data/LC-NE-test/demo/process_windows/neuron_30001/alignment/field.npz \
  --output-dir /data/saem2-review
```

Add **P/N** points and press **I** to predict. This backend produces independent
image masks; **V** propagation requires SAM2. See [interactive controls](README_interactive.md).

`saem2.predictor.SAEM2Predictor` exposes the SAM + HQ mask and membrane probability
separately. Its membrane output uses a full-image box. `saem2.membrane.seeded_component`
extracts a bounded component from a membrane map and an interior point.
The [training guide](../saem2/README.md) describes decoder training and checkpoints.
