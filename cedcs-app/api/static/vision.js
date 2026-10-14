/* Scene analysis: one photo, or a live camera. Pictures are shrunk and re-encoded in the browser (which also removes their
   metadata, such as GPS), sent once for analysis and then dropped. Nothing is recorded, uploaded as a file or stored. */
(() => {
  const $ = id => document.getElementById(id);
  if (!$('visCard')) return;
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const URG = { CRITICAL: '--bad', HIGH: '--warn', MODERATE: '--accent', LOW: '--ok' };
  const MAX_LIVE_MS = 180000; // a live session stops by itself after 3 minutes, to protect the free AI quota
  let mode = 'photo', stream = null, timer = null, busy = false, stopAt = 0, fails = 0, log = [];

  /* ---------- tabs ---------- */
  function setMode(m) {
    if (m !== 'video') stopCamera();
    mode = m;
    document.querySelectorAll('#visTabs button').forEach(b => b.classList.toggle('on', b.dataset.m === m));
    $('visPhoto').hidden = m !== 'photo';
    $('visVideo').hidden = m !== 'video';
    clearOut();
  }
  document.querySelectorAll('#visTabs button').forEach(b => b.onclick = () => setMode(b.dataset.m));

  function clearOut() { $('visOut').innerHTML = ''; log = []; }
  function msg(text, cls) { $('visOut').innerHTML = `<div class="hint ${cls || ''}" style="margin-top:8px">${esc(text)}</div>`; }

  /* ---------- pictures: shrink + re-encode in the browser ---------- */
  /* a picture that is (nearly) one flat colour has nothing to describe, and a model asked about it may invent a scene: never send it */
  function looksBlank(canvas) {
    const t = document.createElement('canvas'); t.width = t.height = 24;
    const g = t.getContext('2d', { willReadFrequently: true }); g.drawImage(canvas, 0, 0, 24, 24);
    const px = g.getImageData(0, 0, 24, 24).data; let sum = 0, sq = 0, n = 0;
    for (let i = 0; i < px.length; i += 4) { const l = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]; sum += l; sq += l * l; n++; }
    const mean = sum / n, sd = Math.sqrt(Math.max(0, sq / n - mean * mean));
    return sd < 7 || mean < 12;
  }
  function shrink(src, w, h, max, q) {
    const s = Math.min(1, max / Math.max(w, h)), c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w * s)); c.height = Math.max(1, Math.round(h * s));
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    if (looksBlank(c)) throw new Error('blank');
    return c.toDataURL('image/jpeg', q);
  }
  async function fromFile(f) {
    if (!f || !f.type.startsWith('image/')) throw new Error('that file is not a picture');
    let bmp = null;
    try { bmp = await createImageBitmap(f, { imageOrientation: 'from-image' }); } catch (e) { /* fall back to an <img> */ }
    if (bmp) { const u = shrink(bmp, bmp.width, bmp.height, 1024, 0.82); if (bmp.close) bmp.close(); return u; }
    return new Promise((res, rej) => {
      const url = URL.createObjectURL(f), im = new Image();
      im.onload = () => { const u = shrink(im, im.naturalWidth, im.naturalHeight, 1024, 0.82); URL.revokeObjectURL(url); res(u); };
      im.onerror = () => { URL.revokeObjectURL(url); rej(new Error('this picture could not be opened')); };
      im.src = url;
    });
  }
  async function analyze(dataUrl, source) {
    const r = await fetch('/vision/analyze', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ image: dataUrl, source }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.detail || ('the analysis failed (error ' + r.status + ')'));
    return j;
  }

  /* ---------- result ---------- */
  const chip = (text, color) => `<span class="vchip" style="color:var(${color})">${esc(text)}</span>`;
  const list = a => a && a.length ? a.map(esc).join(', ') : 'none seen';
  function view(j, source) {
    const a = j.analysis, urg = a.urgency_hint, col = URG[urg] || '--mute';
    if (a.image_quality === 'unusable') {
      return `<div class="vbox"><div class="vwarn">The picture could not be used (too dark, blurred or empty). Nothing was concluded from it: try again with better light and the scene in view.</div>
        <div class="hint" style="margin-top:6px">${esc(j.provider)} · ${j.ms} ms · ${source === 'video' ? 'frame discarded' : 'picture discarded'}.</div></div>`;
    }
    const heavy = a.visible_bleeding === 'heavy';
    return `<div class="vbox ${heavy || urg === 'CRITICAL' ? 'vpulse' : ''}">
      <div>${chip(urg === 'unclear' ? 'urgency unclear' : 'AI hint: ' + urg, col)} ${heavy ? chip('heavy bleeding visible', '--bad') : ''}</div>
      <p>${esc(a.summary) || 'No description.'}</p>
      <div class="vrow"><b>Person</b><span>${esc(a.person_position)}${a.people_visible != null ? ' · ' + a.people_visible + ' visible' : ''} · ${esc(a.movement_cues)}</span></div>
      <div class="vrow"><b>Bleeding</b><span>${esc(a.visible_bleeding)}${a.visible_bleeding_where ? ' (' + esc(a.visible_bleeding_where) + ')' : ''}</span></div>
      <div class="vrow"><b>Injuries</b><span>${list(a.visible_injuries)}</span></div>
      <div class="vrow"><b>Hazards</b><span>${list(a.hazards)}</span></div>
      ${a.image_quality !== 'good' ? `<div class="vwarn">Picture quality is ${esc(a.image_quality)}: treat this as unreliable.</div>` : ''}
      <div class="hint" style="margin-top:6px">${esc(a.limits)}</div>
      <div class="dbtns" style="margin-top:8px"><button type="button" class="btn ghost sm" id="visAdd">Add to report</button><span class="hint" id="visAdded"></span></div>
      <div class="hint" style="margin-top:6px">${esc(j.provider)} · ${j.ms} ms · ${source === 'video' ? 'frame discarded' : 'picture discarded'}. AI can be wrong and does not diagnose: you decide.</div></div>`;
  }
  function bindAdd(a) {
    const b = $('visAdd'); if (!b) return;
    b.onclick = () => {
      const t = $('report'), s = (a.report_sentence || a.summary || '').trim();
      if (!t || !s) return;
      t.value = (t.value.trim() ? t.value.trim() + '\n' : '') + s;
      t.dispatchEvent(new Event('input'));
      $('visAdded').textContent = 'Added to the report: check and edit it before running.';
    };
  }
  function logView() {
    if (!log.length) return '';
    return '<div class="vlog"><b>Recent frames</b>' + log.map(x => `<div>${esc(x.t)} · ${esc(x.u)} · ${esc(x.s)}</div>`).join('') + '</div>';
  }

  /* ---------- photo ---------- */
  async function photo(file) {
    clearOut();
    try {
      msg('Reading the picture…');
      const url = await fromFile(file);
      $('visOut').innerHTML = `<img id="visThumb" alt="the picture being analysed" src="${url}"><div class="hint" style="margin-top:6px">Analysing…</div>`;
      const j = await analyze(url, 'photo');
      $('visOut').innerHTML = `<img id="visThumb" alt="the picture that was analysed" src="${url}">` + view(j, 'photo') +
        '<div class="dbtns" style="margin-top:8px"><button type="button" class="btn ghost sm" id="visClear">Clear</button></div>';
      bindAdd(j.analysis);
      $('visClear').onclick = clearOut;
    } catch (e) { msg(e.message === 'blank' ? 'This picture is blank or very dark, so nothing was sent. Try again with better light.' : (e.message || 'Could not analyse that picture.'), 'st-FAILED'); }
  }
  $('visPick').onclick = () => $('visFile').click();
  $('visTake').onclick = () => $('visCam').click();
  for (const id of ['visFile', 'visCam']) $(id).onchange = e => { const f = e.target.files[0]; e.target.value = ''; if (f) photo(f); };

  /* ---------- live video ---------- */
  function stopCamera(note) {
    if (timer) clearInterval(timer);
    timer = null;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    const v = $('visVid'); if (v) v.srcObject = null;
    if ($('visStart')) { $('visStart').disabled = false; $('visStop').disabled = true; }
    if (note) $('visNote').textContent = note;
  }
  async function tick() {
    const v = $('visVid');
    if (!stream || busy || document.hidden || !v.videoWidth) return;
    if (Date.now() > stopAt) { stopCamera('Stopped after 3 minutes to save the free AI quota. Press Start to continue.'); return; }
    busy = true;
    try {
      const j = await analyze(shrink(v, v.videoWidth, v.videoHeight, 640, 0.6), 'video'); // the frame exists only in this call
      fails = 0;
      const a = j.analysis;
      log.unshift({ t: new Date().toLocaleTimeString(), u: a.urgency_hint, s: a.summary.slice(0, 90) });
      log = log.slice(0, 5);
      $('visOut').innerHTML = view(j, 'video') + logView();
      bindAdd(a);
      $('visNote').textContent = 'Live: analysing a frame every ' + $('visEvery').value + ' s.';
    } catch (e) {
      if (e.message === 'blank') { $('visNote').textContent = 'The camera sees a blank or very dark frame: nothing sent. Point it at the scene.'; return; }
      fails++;
      $('visNote').textContent = e.message;
      if (fails >= 3) stopCamera('Stopped after repeated errors: ' + e.message);
    } finally { busy = false; }
  }
  $('visStart').onclick = async () => {
    clearOut(); fails = 0;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { $('visNote').textContent = 'This browser cannot use the camera here (it needs https or localhost).'; return; }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
    } catch (e) {
      $('visNote').textContent = e.name === 'NotAllowedError' ? 'Camera permission was refused. Allow it in the browser address bar and try again.' : 'No camera could be opened (' + e.name + ').';
      return;
    }
    const v = $('visVid'); v.srcObject = stream; await v.play().catch(() => {});
    $('visStart').disabled = true; $('visStop').disabled = false;
    stopAt = Date.now() + MAX_LIVE_MS;
    $('visNote').textContent = 'Camera on. First analysis in a moment…';
    timer = setInterval(tick, Math.max(3, +$('visEvery').value) * 1000);
    setTimeout(tick, 1200);
  };
  $('visStop').onclick = () => stopCamera('Camera stopped. Nothing was recorded.');
  $('visEvery').onchange = () => { if (timer) { clearInterval(timer); timer = setInterval(tick, Math.max(3, +$('visEvery').value) * 1000); } };
  document.addEventListener('visibilitychange', () => { if (document.hidden && stream) stopCamera('Camera stopped because the tab was hidden.'); });
  window.addEventListener('pagehide', () => stopCamera());

  /* ---------- are any vision models available? ---------- */
  fetch('/vision/status').then(r => r.json()).then(s => {
    if (!s.models || !s.models.length) { $('visBadge').textContent = 'no vision model'; $('visBadge').style.color = 'var(--bad)'; msg('No vision model is configured (a Groq or NVIDIA key is needed).', 'st-FAILED'); }
  }).catch(() => {});
})();
