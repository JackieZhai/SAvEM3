"""Install browser-side section movement without interrupting mouse drags."""
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs
from weakref import WeakValueDictionary

_lock = threading.RLock()


class _NavigationContentSource:
    def __init__(self, base):
        self.base = base
        self.viewers = WeakValueDictionary()
        self.javascript = Path(__file__).with_name('viewer_navigation.js').read_bytes()

    def get(self, name, query):
        if name == 'probe-em-navigation.js':
            return self.javascript, 'application/javascript'
        if name == 'probe-em-navigation.json':
            token = parse_qs(query.lstrip('?')).get('viewer', [''])[0]
            with _lock:
                viewer = self.viewers.get(token)
                bounds = getattr(viewer, 'probe_em_section_bounds', None)
            config = ({'enabled': True, 'min_z': bounds[0], 'max_z': bounds[1]}
                      if bounds is not None else {'enabled': False})
            return json.dumps(config).encode(), 'application/json'
        data, content_type = self.base.get(name, query)
        if name in ('', 'index.html') and 'text/html' in content_type:
            script = b'<script defer src="probe-em-navigation.js"></script>'
            if script not in data:
                data = data.replace(b'</head>', script+b'</head>', 1)
        return data, content_type


def install_section_navigation(viewer, z_min, z_max):
    """Enable absolute-Z D/F for this viewer; retain its Python action fallback.

    The bundled Neuroglancer assets are delegated unchanged. The extra script
    edits the browser's live position, so a server state reset cannot cancel an
    ongoing drag. Other viewers sharing the server are left unconfigured.
    """
    import neuroglancer
    from neuroglancer import server, static
    if int(z_min) != z_min or int(z_max) != z_max or z_min > z_max:
        raise ValueError('Section navigation requires ordered integer Z bounds')
    with _lock:
        source = server.global_static_content_source
        if not isinstance(source, _NavigationContentSource):
            source = _NavigationContentSource(source or static.get_default_static_content_source())
            neuroglancer.set_static_content_source(source=source)
        viewer.probe_em_section_bounds = (int(z_min), int(z_max))
        source.viewers[viewer.token] = viewer
