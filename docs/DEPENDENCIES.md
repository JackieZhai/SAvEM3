# Dependencies

The training and Probe-EM manifests pin `torch==2.14.1` with
`torchvision==0.29.1`. Install the matching pair for the target platform;
see [Probe-EM installation](../probe-em/INSTALL.md) and the
[training environment](../repro/env/README.md).

SOFIMA uses a separate worker environment. Native graph-cut dependencies use
the postprocessing environment. ONNX Runtime is optional; see
[SAM2 runtime options](../probe-em/README_acceleration.md).

## Maintenance

After changing dependency versions, run `python -m pip check`, the regression
suite in [TESTING.md](TESTING.md), and the applicable model integration tests.
Keep all manifests and installation examples consistent. Review Dependabot
alerts for the default branch and audit installed environments separately;
editable source checkouts may require manual review.

Load checkpoints from trusted sources. Model checkpoints and generated data
are stored outside the repository.
