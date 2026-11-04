"""Regression checks for the English-only publishing policy."""
from pathlib import Path

import pytest

from repro.check_english import audit, has_chinese_text


@pytest.mark.parametrize('text', [
    'English comments; SAEM²/SAvEM³; 8×8×30 nm.',
    'https://example.org/data?x=1&y=2',
    r'Invalid escapes: \uZZZZ, \UFFFFFFFF, &#999999999;',
])
def test_english_and_scientific_notation_are_allowed(text):
    assert not has_chinese_text(text)


@pytest.mark.parametrize('text', [
    chr(0x4E2D),
    chr(0xFF1A),
    chr(0x3002),
    chr(0x20000),
    '\\u' + '4e2d',
    '\\U' + '00020000',
    '\\u' + 'd840' + '\\u' + 'dc00',
    '&#x' + '4e2d;',
    '&#' + str(0x4E2D) + ';',
])
def test_literal_and_encoded_han_or_punctuation_are_detected(text):
    assert has_chinese_text(text)


def test_publishable_repository_text_is_english():
    checked, failures = audit(Path(__file__).resolve().parents[1])
    assert checked > 0
    assert not failures, '\n'.join(failures)
