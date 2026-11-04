"""Pinned dependency and benign compatibility checks; never run vulnerability PoCs."""
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version
import pytest


ROOT = Path(__file__).resolve().parents[1]
TORCH = '2.14.1'
VISION = '0.29.1'
# Dated GitHub Dependabot snapshot, rechecked on 2026-11-03.
TORCH_ALERTS = [
    ('GHSA-3749-ghw9-m3mg', '<2.7.1rc1'),
    ('GHSA-887c-mr87-cxwp', '<=2.7.1'),
    ('GHSA-c678-jfcj-6jmf', '<=2.6.0'),
    ('GHSA-f4hp-rmr7-r7v8', '<=2.6.0'),
    ('GHSA-x3gm-94wq-g975', '<=2.6.0'),
    ('GHSA-vgrw-7cvw-pwgx', '<2.9.1'),
    ('GHSA-qfhq-4f3w-5fph', '<2.10.0'),
    ('GHSA-rrmf-rvhw-rf47', '<=2.12.1'),
]


def requirements(path):
    values = {}
    for line in (ROOT / path).read_text().splitlines():
        line = line.partition('#')[0].strip()
        if line:
            requirement = Requirement(line)
            values[requirement.name.lower().replace('_', '-')] = requirement
    return values


@pytest.mark.parametrize('manifest', ['requirements-train.txt', 'probe-em/requirements.txt'])
def test_manifests_pin_the_same_compatible_torch_pair(manifest):
    values = requirements(manifest)
    assert str(values['torch'].specifier) == f'=={TORCH}'
    assert str(values['torchvision'].specifier) == f'=={VISION}'
    assert 'torchaudio' not in values
    assert 'opencv-python' not in values  # Avoid two providers of the cv2 namespace.
    assert 'opencv-python-headless' in values


@pytest.mark.parametrize('advisory,affected', TORCH_ALERTS)
def test_pinned_torch_is_outside_each_reported_affected_range(advisory, affected):
    assert not SpecifierSet(affected).contains(Version(TORCH), prereleases=True), advisory


@pytest.mark.parametrize('path', ['README.md', 'probe-em/README.md',
                                'probe-em/INSTALL.md', 'repro/env/setup.sh'])
def test_installation_examples_match_the_pinned_pair(path):
    text = (ROOT / path).read_text()
    assert f'torch=={TORCH} torchvision=={VISION}' in text
    assert 'torch==2.6.0' not in text
    assert 'python=3.9' not in text


def test_legacy_configuration_provider():
    if importlib.util.find_spec('attrdict') is None:
        pytest.skip('Training configuration dependency is not installed')
    from attrdict import AttrDict
    config = AttrDict({'TRAIN': {'base_lr': 1e-4}, 'MODEL': {'filters': [32, 64]}})
    assert config.TRAIN.base_lr == 1e-4
    assert tuple(config.MODEL.filters) == (32, 64)
    config['TRAIN']['base_lr'] = 2e-4
    assert config.TRAIN.base_lr == 2e-4
    values = requirements('requirements-train.txt')
    assert 'attrdict3' in values and 'attrdict' not in values


@pytest.mark.parametrize('folder', ['saem2', 'sam-hq/train'])
def test_scripted_mask_losses_forward_backward(folder):
    pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    # Separate processes avoid collisions between historical top-level utils packages.
    script = '''
import os
import torch
from utils.loss_mask import dice_loss, dice_loss_jit, sigmoid_ce_loss, sigmoid_ce_loss_jit
torch.set_num_threads(2)
device = os.environ.get('SAVEM3_TEST_DEVICE', 'cpu')
inputs = torch.linspace(-2, 2, 32, device=device).reshape(2, 16).requires_grad_()
targets = torch.ones_like(inputs)
loss = dice_loss_jit(inputs, targets, 2.0) + sigmoid_ce_loss_jit(inputs, targets, 2.0)
expected = dice_loss(inputs, targets, 2.0) + sigmoid_ce_loss(inputs, targets, 2.0)
torch.testing.assert_close(loss, expected)
loss.backward()
assert torch.isfinite(loss) and torch.isfinite(inputs.grad).all()
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT / folder,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(os.environ.get('SAVEM3_SECURITY_VALIDATE_RUNTIME') != '1',
                    reason='Explicit upgraded-runtime validation only')
def test_upgraded_runtime_matches_the_manifest():
    assert Version(importlib.metadata.version('torch')).base_version == TORCH
    assert Version(importlib.metadata.version('torchvision')).base_version == VISION
