# Dependency security remediation and validation

Rechecked on 2026-11-03 against the GitHub default branch at
`b1b5d4f941a81fc3ea5dda4be1bcc933359ea00f`, which matched the local HEAD.
Repository-specific Dependabot alerts were read through the GitHub API. The local
dependency manifests now pin **torch 2.14.1 and torchvision 0.29.1**. This Torch
version is outside all eight reported affected ranges below. Validation used a
new Python 3.12 environment; existing demo environments were preserved.

The remote default branch still had eight open alerts when inspected. No alerts
were dismissed, no exploits were executed, and these local changes have not been
committed or pushed. GitHub closure requires publication and a dependency rescan.

## Why the reported alert appears

[Alert 1](https://github.com/JackieZhai/SAvEM3/security/dependabot/1) identifies
`torch` as a direct dependency in the remote `probe-em/requirements.txt`. That file pins
`torch==2.6.0`, which falls in the advisory's affected range `<2.7.1-rc1`.
This is [CVE-2025-2953 / GHSA-3749-ghw9-m3mg](https://github.com/advisories/GHSA-3749-ghw9-m3mg):
a local denial-of-service issue involving `torch.mkldnn_max_pool2d`, currently
rated Low by GitHub. It is not evidence that the repository or demos were hacked.

Dependabot evaluates the default branch's dependency graph, not the PyTorch
installed in a developer's other environments. Notifications can follow either
dependency changes or new advisory entries. See
[GitHub's alert documentation](https://docs.github.com/en/code-security/concepts/supply-chain-security/dependabot-alerts).
The exact reason this particular email was sent on its delivery date cannot be
established without the email's timestamp and notification history.

Before this remediation:

- `probe-em/INSTALL.md` and `probe-em/README.md` repeat the 2.6.0 installation.
- `requirements-train.txt` allows `torch>=2.1`, including vulnerable older versions.
  The observed eight GitHub alerts reference the Probe-EM manifest, not this file.
- The historical local Probe-EM environment contains torch 2.6.0, torchvision 0.21.0 and
  torchaudio 2.6.0. The separate training environment contains torch 2.8.0 and
  torchvision 0.23.0. Its newer Torch does not fix the repository's manifest.

No direct `mkldnn_max_pool2d` or MKLDNN conversion calls were found in publishable
source files. This is not proof of non-exploitability: library internals and
deployment inputs have not been exhaustively analyzed. Model files and pickle
inputs must still be trusted.

## The repository has more than this one alert

All eight open alerts observed in the API response reference
`probe-em/requirements.txt` and package `torch`:

| Alert / CVE | Affected range | First patched version recorded by GitHub |
| --- | --- | --- |
| [1 / CVE-2025-2953](https://github.com/advisories/GHSA-3749-ghw9-m3mg) | `<2.7.1-rc1` | `2.7.1-rc1` |
| [2 / CVE-2025-3730](https://github.com/advisories/GHSA-887c-mr87-cxwp) | `<=2.7.1` | `2.8.0` |
| [3 / CVE-2025-2148](https://github.com/advisories/GHSA-c678-jfcj-6jmf) | `<=2.6.0` | Not recorded |
| [4 / CVE-2025-2998](https://github.com/advisories/GHSA-f4hp-rmr7-r7v8) | `<=2.6.0` | Not recorded |
| [5 / CVE-2025-2149](https://github.com/advisories/GHSA-x3gm-94wq-g975) | `<=2.6.0` | Not recorded |
| [6 / CVE-2025-2999](https://github.com/advisories/GHSA-vgrw-7cvw-pwgx) | `<2.9.1` | `2.9.1` |
| [7 / CVE-2025-3001](https://github.com/advisories/GHSA-qfhq-4f3w-5fph) | `<2.10.0` | `2.10.0` |
| [8 / CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) | `<=2.12.1` | `2.13.0` |

This table is a dated advisory snapshot, not a certification that any release has
no vulnerabilities. An absent first-patched version must not be treated as proof
of a particular fix. The SAEM2 and HQ-SAM loss modules use `torch.jit.script`,
which is the API named by alert 8; normal use alone does not demonstrate exploitation.

## Implemented upgrade

`2.7.1-rc1` names the first patched candidate for alert 1; installing an RC is not
required. Stable 2.7.1 satisfies that advisory. The official matching set is
torch 2.7.1, torchvision 0.22.1 and torchaudio 2.7.1, but it does **not** address all
eight alerts. The digest's `~>` notation is not a pip requirements operator. Use
Python requirement syntax such as `==` or `>=` with an intentionally chosen,
validated version range. See [the official version matrix](https://pytorch.org/get-started/previous-versions/).

Do not upgrade Torch alone while leaving torchvision pinned at 0.21.0, or keep
the cu124 index when the chosen release has no matching CUDA 12.4 wheel. Also do
not install a newer version manually and then reinstall the old requirements:
that would request the vulnerable 2.6.0 pin again.

The selected stable pair, torch 2.14.1 / torchvision 0.29.1, is listed in the
[official installation matrix](https://pytorch.org/get-started/previous-versions/).
It supports Python >=3.10; the tested environment uses Python 3.12.8 on macOS
26.4.1 arm64. The installation guides recommend Python 3.12 and document the
matching CUDA 12.6 or CPU index; macOS installs from PyPI. This is an explicitly
tested pair, not a claim that it is the newest available release.

Both `requirements-train.txt` and `probe-em/requirements.txt` pin this pair. The
unused torchaudio dependency was removed. Both manifests use
`opencv-python-headless` to avoid installing two providers of the `cv2` namespace.
The legacy `attrdict` package failed to import on Python 3.12 because it references
removed `collections` aliases; `attrdict3==2.0.2` preserves the existing import
and configuration API. Training environment setup and all maintained installation
examples were updated together. Native ELF/WaterZ retains its separate environment.

Backend tests exposed two additional compatibility bugs: `saem2.modeling` depended
on a prior `sys.path` mutation, and resized SAEM2 prompts could become float64,
which MPS does not support. Package/script imports now work independently, and
point/box tensors explicitly use float32. Model parameter names are unchanged.

## Validation and limits

- `pip check`: no broken requirements.
- `pip-audit`: **0 known vulnerabilities across 163 audited distributions**;
  the editable local `SAM-2==1.0` package is the one skipped distribution because
  it is not available on PyPI. This is not an audit of that source or the OS.
- Upgraded CPU suite: **130 passed, 4 skipped**. Skips are native postprocessing
  packages intentionally absent from this environment.
- MPS model/training/loss suite: **9 passed**; interactive suite: **10 passed**.
  MPS fallback was enabled. Real local SAM, HQ-SAM, trained SAEM2 and SAM2 weights
  were used; SAM2 video memory, teacher export and student training/resume were
  also exercised. Synthetic inputs validate compatibility, not biological quality.
- Separate native postprocessing suite: **11 passed, 2 skipped**.
- The scripted SAEM2/HQ-SAM mask losses were evaluated and differentiated with
  benign tensors in isolated processes, including on MPS. No advisory payloads
  were executed.

CUDA hardware was unavailable, so CUDA execution is not claimed tested. The
pre-existing demo environments still contain their historical versions; use the
new environment for the corrected manifests and interactive demo. Local evidence
is stored under ignored `.cache/security-20261103/` (JUnit, audit JSON and browser
artifacts). See [the validation record](VALIDATION.md) for the interactive checks.

Publishing these changes and confirming the GitHub rescan remains separate from
local remediation. A local edit or environment upgrade alone cannot close the
default-branch alerts.
