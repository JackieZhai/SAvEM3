/* Move through the stack without replacing the live Neuroglancer drag state. */
(() => {
  'use strict';
  const match = location.pathname.match(/\/v\/([^/]+)/);
  if (!match) return;
  const url = new URL('probe-em-navigation.json', location.href);
  url.searchParams.set('viewer', match[1]);
  fetch(url, {cache: 'no-store'}).then(response => {
    if (!response.ok) throw new Error('Section navigation configuration is unavailable');
    return response.json();
  }).then(config => {
    if (!config.enabled) return;
    if (!Number.isInteger(config.min_z) || !Number.isInteger(config.max_z) || config.min_z > config.max_z) {
      throw new Error('Invalid section navigation bounds');
    }
    let dragPointer = null;
    document.addEventListener('pointerdown', event => {
      if (event.target instanceof Element && event.target.closest('.neuroglancer-rendered-data-panel')) {
        dragPointer = event.pointerId;
      }
    }, true);
    const finishDrag = event => {
      if (event.pointerId === dragPointer) dragPointer = null;
    };
    document.addEventListener('pointerup', finishDrag, true);
    document.addEventListener('pointercancel', finishDrag, true);
    window.addEventListener('blur', () => { dragPointer = null; });
    document.addEventListener('keydown', event => {
      const target = event.target;
      if (!config.review_controls || !event.repeat || event.isComposing || event.altKey ||
          event.ctrlKey || event.metaKey || event.shiftKey ||
          !['KeyY', 'KeyC', 'Enter', 'NumpadEnter'].includes(event.code)) return;
      if (target instanceof Element && (target.closest('input,textarea,select') || target.isContentEditable)) return;
      // Acceptance and each merge confirmation require a distinct key press.
      event.preventDefault();
      event.stopImmediatePropagation();
    }, true);
    document.addEventListener('keydown', event => {
      if (!['KeyD', 'KeyF'].includes(event.code) || event.isComposing ||
          event.altKey || event.ctrlKey || event.metaKey || (event.shiftKey && dragPointer === null)) return;
      const target = event.target;
      if (target instanceof Element && (target.closest('input,textarea,select') || target.isContentEditable)) return;
      const viewer = window.viewer;
      const position = viewer?.navigationState?.position;
      const names = position?.coordinateSpace?.value?.names;
      const axis = names?.indexOf('z');
      if (axis === undefined || axis < 0 || !Number.isFinite(position.value[axis])) return;
      const next = new Float32Array(position.value);
      const step = event.code === 'KeyF' ? 1 : -1;
      next[axis] = Math.min(config.max_z, Math.max(config.min_z, Math.floor(next[axis])+step))+.5;
      event.preventDefault();
      event.stopImmediatePropagation();
      position.value = next;
    }, true);
    document.documentElement.dataset.probeEmSectionNavigation = 'ready';
  }).catch(error => console.warn('Using server-side D/F navigation:', error));
})();
