"""The navigation extension preserves existing assets and isolates viewer bounds."""
import json
from pathlib import Path
import sys

import pytest

neuroglancer = pytest.importorskip('neuroglancer')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'probe-em'))
from probe_em.viewer_navigation import install_section_navigation


def test_extension_delegates_assets_and_configures_each_viewer_separately(monkeypatch):
    from neuroglancer import server

    class Source:
        def get(self, name, query):
            if name in ('', 'index.html'):
                return b'<html><head></head><body>custom client</body></html>', 'text/html'
            return (name+query).encode(), 'application/octet-stream'

    class Viewer:
        def __init__(self, token):
            self.token = token

    monkeypatch.setattr(server, 'global_static_content_source', Source())
    first, second = Viewer('first'), Viewer('second')
    install_section_navigation(first, 100, 162)
    source = server.global_static_content_source
    install_section_navigation(second, 500, 562)
    assert server.global_static_content_source is source
    html, mime = source.get('', '')
    assert b'custom client' in html and mime == 'text/html'
    assert html.count(b'src="probe-em-navigation.js"') == 1
    assert source.get('original.wasm', '?version=2') == (b'original.wasm?version=2', 'application/octet-stream')
    assert json.loads(source.get('probe-em-navigation.json', '?viewer=first')[0]) == {
        'enabled': True, 'min_z': 100, 'max_z': 162}
    assert json.loads(source.get('probe-em-navigation.json', '?viewer=second')[0])['min_z'] == 500
    assert json.loads(source.get('probe-em-navigation.json', '?viewer=unconfigured')[0]) == {'enabled': False}
    with pytest.raises(ValueError, match='integer Z bounds'):
        install_section_navigation(first, 4.5, 8)
    with pytest.raises(ValueError, match='integer Z bounds'):
        install_section_navigation(first, 9, 8)
