"""Route review messages to the native viewer or its compact browser toolbar."""
import threading


class ToolbarStatus:
    def __init__(self, messages=None):
        self.messages = dict(messages or {})
        self.lock = threading.RLock()

    def set(self, key, message):
        with self.lock:
            self.messages[key] = str(message)

    def snapshot(self):
        with self.lock:
            return [{'id': key, 'text': text} for key, text in self.messages.items()]


def attach_toolbar_status(viewer):
    """Capture startup messages before publishing the viewer URL."""
    feed = ToolbarStatus(viewer.config_state.state.status_messages.items())
    viewer.probe_em_toolbar_status = feed
    with viewer.config_state.txn() as state:
        state.status_messages = {}
    return feed


def set_status(viewer, key, message):
    feed = getattr(viewer, 'probe_em_toolbar_status', None)
    if feed is not None:
        feed.set(key, message)
    else:
        with viewer.config_state.txn() as state:
            state.status_messages[key] = message
