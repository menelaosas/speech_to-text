'use strict';
const $ = id => document.getElementById(id);
const audio = $('audio');

// ---- state -------------------------------------------------------------------------------------------------
let jobs = [], jobsSignature = '', current = null, documentData = null, edits = null, dirty = false;
let selectedFile = null, wordTarget = null, generation = 0, savedVersion = null;
let wordNodes = [], segmentNodes = [], segmentInputs = [], activeWordNodes = new Set(), lastActive = new Set();
let maxWordDuration = 0, animation = null, stopAt = null, connectionLost = false, playingSegment = -1;
const speakerColor = new Map();

const STEPS = [
  ['01_inspect', 'Inspect file'], ['02_prepare', 'Prepare audio'], ['03_vad', 'Detect speech'],
  ['04_diarization', 'Identify speakers'], ['05_asr', 'Transcribe'], ['05b_normalize', 'Restore final ς'],
  ['05c_align', 'Align words'], ['06_attribution', 'Assign speakers'], ['07_qc', 'Quality checks'],
  ['08_export', 'Export'], ['09_integrity', 'Verify original'],
];
// distinct hues, each dark enough for text on the light background (WCAG AA)
const PALETTE = ['#2f6f5e', '#b0532a', '#3a62b8', '#a23b84', '#5f6e10', '#b3263e', '#1b7590', '#7c5a28', '#6a4bc2', '#9a6a00',
                 '#3f7f3a', '#555f6e'];

// ---- helpers -----------------------------------------------------------------------------------------------
function notice(message, kind = 'error') {
  $('notice-text').textContent = message;
  $('notice').className = kind;
  $('notice').hidden = !message;
  clearTimeout(notice.timer);
  if (message && kind === 'info') notice.timer = setTimeout(() => { $('notice').hidden = true; }, 4000);
}
$('notice-close').onclick = () => { $('notice').hidden = true; };

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {'X-Review-Client': '1', ...options.headers}});
  let data = {};
  try { data = await response.json(); } catch { /* empty body */ }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function clock(value) {
  value = Math.max(0, Math.floor(value || 0));
  const h = Math.floor(value / 3600), m = Math.floor(value % 3600 / 60), s = value % 60;
  const mm = String(m).padStart(2, '0'), ss = String(s).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}
function bytes(n) { return n > 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.ceil(n / 1e3)} KB`; }
function show(view) {
  for (const id of ['welcome', 'editor', 'processing']) $(id).hidden = id !== view;
  $('crumb').textContent = {welcome: 'NEW RECORDING', editor: 'TRANSCRIPT EDITOR', processing: 'PROCESSING'}[view];
}
function confirmDialog(title, text, ok = 'Confirm') {
  $('confirm-title').textContent = title; $('confirm-text').textContent = text; $('confirm-ok').textContent = ok;
  $('confirm-dialog').showModal();
  return new Promise(resolve => $('confirm-dialog').addEventListener('close',
    () => resolve($('confirm-dialog').returnValue === 'ok'), {once: true}));
}
async function canLeave() {
  return !dirty || confirmDialog('Leave without saving?', 'Your corrections since the last saved version will be lost.', 'Leave');
}
function colorFor(label) {
  if (!label) return '#8a8f86';
  if (!speakerColor.has(label)) speakerColor.set(label, PALETTE[speakerColor.size % PALETTE.length]);
  return speakerColor.get(label);
}
function setDirty(value) {
  dirty = value;
  $('save').disabled = !dirty;
  if (dirty) $('save-state').textContent = 'Unsaved corrections';
}

// ---- upload ------------------------------------------------------------------------------------------------
function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  $('filename').textContent = file.name;
  $('filesize').textContent = `${bytes(file.size)} · ready to transcribe`;
  $('transcribe').disabled = false;
}
function resetUpload() {
  selectedFile = null; $('file').value = '';
  $('filename').textContent = 'Choose an audio recording';
  $('filesize').textContent = 'Drop a file here or click to browse';
  $('transcribe').disabled = true; $('upload-progress').hidden = true;
}
$('file').addEventListener('change', event => selectFile(event.target.files[0]));
for (const type of ['dragenter', 'dragover']) $('drop').addEventListener(type, e => { e.preventDefault(); $('drop').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('drop').addEventListener(type, e => { e.preventDefault(); $('drop').classList.remove('dragging'); });
$('drop').addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));

function uploadWithProgress(file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/jobs?name=${encodeURIComponent(file.name)}`);
    xhr.setRequestHeader('X-Review-Client', '1');
    xhr.upload.onprogress = e => {
      if (!e.lengthComputable) return;
      const pct = Math.round(100 * e.loaded / e.total);
      $('upload-bar').style.width = `${pct}%`;
      $('upload-text').textContent = `Uploading ${pct}% · ${bytes(e.loaded)} of ${bytes(e.total)}`;
    };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      xhr.status < 300 ? resolve(data) : reject(new Error(data.error || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('Upload failed: the local server did not respond'));
    xhr.send(file);
  });
}
$('upload-form').onsubmit = async event => {
  event.preventDefault();
  if (!selectedFile) return;
  if (selectedFile.size > 2 * 1024 ** 3 || !selectedFile.size) return notice('Choose an audio file between 1 byte and 2 GB.');
  $('transcribe').disabled = true; $('transcribe-label').textContent = 'Uploading recording…';
  $('upload-progress').hidden = false; $('upload-bar').style.width = '0%'; notice('');
  try {
    const job = await uploadWithProgress(selectedFile);
    resetUpload();
    await refreshJobs(); await openJob(job.id);
  } catch (error) { notice(error.message); $('transcribe').disabled = false; }
  finally { $('transcribe-label').textContent = 'Transcribe recording'; }
};

// ---- sidebar -----------------------------------------------------------------------------------------------
function statusText(job) {
  if (job.status === 'ready') return ['✓ Ready to review', 'ready'];
  if (job.status === 'failed') return ['Failed · open for details', 'failed'];
  if (job.status === 'queued') return ['Queued', 'busy'];
  if (job.status === 'uploading') return ['Uploading…', 'busy'];
  const done = STEPS.filter(([key]) => job.completed_steps.includes(key)).length;
  return [`Processing · ${done}/${STEPS.length}`, 'busy'];
}
function renderJobs() {
  const signature = JSON.stringify([current, jobs.map(j => [j.id, j.name, j.status, j.completed_steps.length])]);
  if (signature === jobsSignature) return;
  jobsSignature = signature;
  $('jobs').replaceChildren();
  if (!jobs.length) {
    const empty = document.createElement('p'); empty.className = 'jobs-empty'; empty.textContent = 'No recordings yet.';
    $('jobs').append(empty);
  }
  for (const job of jobs) {
    const button = document.createElement('button');
    button.className = `job${current === job.id ? ' selected' : ''}`;
    if (current === job.id) button.setAttribute('aria-current', 'page');
    const name = document.createElement('strong'); name.textContent = job.name; name.title = job.name;
    const [text, kind] = statusText(job);
    const meta = document.createElement('small');
    const badge = document.createElement('span'); badge.className = `badge ${kind}`; badge.textContent = text;
    const date = new Date(job.created_at);
    meta.append(badge, ` · ${date.toLocaleDateString()} ${date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`);
    button.append(name, meta);
    button.onclick = () => openJob(job.id).catch(error => notice(error.message));
    $('jobs').append(button);
  }
}
async function refreshJobs() { jobs = await api('/api/jobs'); renderJobs(); }
$('new').onclick = async () => {
  if (!await canLeave()) return;
  setDirty(false); current = null; documentData = null; generation++; audio.pause(); audio.removeAttribute('src');
  document.title = 'Speech Studio · Listen, review, refine';
  resetUpload(); show('welcome'); jobsSignature = ''; renderJobs(); notice('');
};

// ---- processing view ---------------------------------------------------------------------------------------
function processing(job) {
  show('processing');
  $('process-name').textContent = job.name;
  const failed = job.status === 'failed';
  $('process-eyebrow').textContent = failed ? 'TRANSCRIPTION FAILED' : job.status === 'queued' ? 'WAITING IN QUEUE' : 'TRANSCRIPTION IN PROGRESS';
  const elapsed = (Date.now() - new Date(job.created_at)) / 1000;
  $('process-state').textContent = failed ? ''
    : job.status === 'queued' ? 'Queued. Recordings are processed one at a time, using the GPU on this computer.'
    : `Processing on this computer · ${clock(elapsed)} elapsed. A one-hour recording takes about 5 minutes. You can leave this page and come back.`;
  const done = STEPS.filter(([key]) => job.completed_steps.includes(key)).length;
  $('process-bar').style.width = `${Math.round(100 * done / STEPS.length)}%`;
  $('process-bar').parentElement.classList.toggle('failed', failed);
  $('process-count').textContent = `${done} of ${STEPS.length} steps`;
  $('steps').replaceChildren();
  let currentMarked = false;
  for (const [key, label] of STEPS) {
    const li = document.createElement('li');
    const isDone = job.completed_steps.includes(key);
    let state = isDone ? 'done' : 'pending';
    if (!isDone && !currentMarked && job.status === 'running') { state = 'current'; currentMarked = true; }
    if (!isDone && !currentMarked && failed) { state = 'error'; currentMarked = true; }
    li.className = `step ${state}`;
    li.textContent = label;
    if (state === 'current') li.setAttribute('aria-current', 'step');
    $('steps').append(li);
  }
  $('failed-actions').hidden = !failed;
  $('failed-text').textContent = failed ? (job.error || 'Transcription failed.') : '';
  $('log-box').open = failed;
  const log = $('log'), atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
  log.textContent = job.log || 'Waiting for the pipeline…';
  if (atBottom) log.scrollTop = log.scrollHeight;
}
$('retry').onclick = async () => {
  try { const job = await api(`/api/jobs/${current}/retry`, {method: 'POST'}); await refreshJobs(); processing(job); }
  catch (error) { notice(error.message); }
};
async function deleteCurrent() {
  const job = jobs.find(j => j.id === current);
  if (!job) return;
  const ok = await confirmDialog('Delete this recording?',
    `“${job.name}” will be removed from this workspace: the uploaded copy, the transcript and all saved versions. This cannot be undone.`, 'Delete');
  if (!ok) return;
  try {
    await api(`/api/jobs/${current}/delete`, {method: 'POST'});
    setDirty(false); current = null; documentData = null; audio.pause(); audio.removeAttribute('src');
    resetUpload(); show('welcome'); jobsSignature = ''; await refreshJobs(); notice('Recording deleted.', 'info');
  } catch (error) { notice(error.message); }
}
$('delete-failed').onclick = deleteCurrent;
$('delete-ready').onclick = deleteCurrent;

// ---- editor ------------------------------------------------------------------------------------------------
async function openJob(id) {
  if (current === id && documentData && !$('editor').hidden) return;
  if (!await canLeave()) return;
  setDirty(false); current = id; documentData = null; audio.pause(); notice('');
  const ticket = ++generation; jobsSignature = ''; renderJobs();
  const job = await api(`/api/jobs/${id}`); if (ticket !== generation) return;
  document.title = `${job.name} · Speech Studio`;
  if (job.status !== 'ready') { processing(job); return; }
  const data = await api(`/api/jobs/${id}/transcript`); if (ticket !== generation) return;
  documentData = data;
  const segments = data.transcript.segments;
  edits = data.draft || {segments: segments.map(seg => ({
    id: seg.id, speaker: seg.primary_speaker || 'UNKNOWN', text: seg.text_normalized ?? seg.text,
    words: (seg.words || []).map(word => word.text_normalized ?? word.text)}))};
  speakerColor.clear();
  for (const label of [...new Set(segments.map(s => s.primary_speaker).filter(Boolean))].sort()) colorFor(label);
  $('recording-name').textContent = job.name;
  const speakers = new Set(segments.map(s => s.primary_speaker).filter(Boolean));
  const sigma = segments.some(s => s.text_normalized && s.text_normalized !== s.text);
  $('summary').textContent = `${clock(data.transcript.source.duration_s)} recording · ${speakers.size} speakers · `
    + `${data.transcript.language.toUpperCase()}${sigma ? ' · showing text with restored final ς' : ''}`;
  $('segment-count').textContent = segments.length;
  $('save-state').textContent = data.draft ? 'Saved corrections loaded' : 'Original transcript';
  setSavedVersion(data.latest_version);
  for (const kind of ['txt', 'srt', 'json']) {
    $(`orig-${kind}`).href = `/api/jobs/${id}/download?version=original&format=${kind}`;
    $(`orig-${kind}`).download = `${job.name.replace(/\.[^.]+$/, '')}.machine.${kind}`;
  }
  $('search').value = ''; $('review-filter').value = 'all'; $('match-count').textContent = '';
  audio.src = `/api/jobs/${id}/audio`; audio.playbackRate = Number($('speed').value);
  show('editor'); renderTranscript(); renderSpeakers(); window.scrollTo(0, 0);
}
function setSavedVersion(version) {
  savedVersion = version;
  for (const kind of ['txt', 'srt', 'json']) {
    const link = $(`download-${kind}`);
    if (version) { link.href = `/api/jobs/${current}/download?version=${version}&format=${kind}`; link.download = `transcript.edited.${kind}`; }
    else link.removeAttribute('href');
    link.classList.toggle('disabled', !version);
  }
  if (version) {
    const m = version.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/);
    const when = m ? new Date(Date.UTC(+m[1], m[2] - 1, +m[3], +m[4], +m[5], +m[6])) : null;
    $('saved-when').textContent = when ? `Saved ${when.toLocaleString()}` : 'Latest saved version';
  } else $('saved-when').textContent = 'Save a version to download it';
}

function flaggedSegment(seg) {
  return seg.speaker_uncertain || seg.overlap || (seg.flags || []).some(f => f.severity === 'warning');
}
// warnings other than overlap alone: these are the segments most likely to contain errors
function warnedSegment(seg) {
  return seg.speaker_uncertain || (seg.flags || []).some(f => f.severity === 'warning');
}
function reviewMatch(seg, mode) {
  return mode === 'flagged' ? flaggedSegment(seg) : mode === 'warnings' ? warnedSegment(seg) : true;
}
function segmentWarnings(seg) {
  const list = (seg.flags || []).filter(f => f.severity === 'warning').map(f => f.code.replaceAll('_', ' '));
  if (seg.overlap) list.push('overlapping speech');
  const speakerFlagged = (seg.flags || []).some(f => ['speaker_uncertain', 'no_diarized_speaker'].includes(f.code));
  if (seg.speaker_uncertain && !speakerFlagged) list.push('check speaker');
  return [...new Set(list)];
}
function baseline(word) { return word.text_normalized ?? word.text; }

function renderTranscript() {
  const root = $('transcript');
  root.replaceChildren();
  wordNodes = []; segmentNodes = []; segmentInputs = []; activeWordNodes = new Set(); lastActive = new Set(); maxWordDuration = 0;
  playingSegment = -1;
  const fragment = document.createDocumentFragment();
  documentData.transcript.segments.forEach((segment, si) => {
    const edit = edits.segments[si];
    const row = document.createElement('article'); row.className = 'segment';
    row.style.setProperty('--speaker', colorFor(segment.primary_speaker));
    segmentNodes.push(row);
    const meta = document.createElement('div'); meta.className = 'segment-meta';
    const speaker = document.createElement('input');
    speaker.className = 'speaker'; speaker.value = edit.speaker; speaker.maxLength = 100;
    speaker.setAttribute('list', 'speaker-names');
    speaker.setAttribute('aria-label', `Speaker for segment ${si + 1}`);
    speaker.oninput = () => { edit.speaker = speaker.value; setDirty(true); };
    speaker.onchange = () => { if (!speaker.value.trim()) { speaker.value = edit.speaker = segment.primary_speaker || 'UNKNOWN'; } renderSpeakers(); };
    segmentInputs.push(speaker);
    const timestamp = document.createElement('button');
    timestamp.className = 'timestamp'; timestamp.textContent = `${clock(segment.start)} – ${clock(segment.end)}`;
    timestamp.setAttribute('aria-label', `Play from ${clock(segment.start)}`);
    timestamp.onclick = () => { seek(segment.start); audio.play().catch(() => {}); };
    meta.append(speaker, timestamp);
    const others = (segment.speakers || []).filter(s => s !== segment.primary_speaker);
    if (others.length) {
      const also = document.createElement('div'); also.className = 'also';
      also.textContent = `with ${others.join(', ')}`; meta.append(also);
    }
    const warnings = segmentWarnings(segment);
    if (warnings.length) {
      const flags = document.createElement('div'); flags.className = 'flags';
      flags.textContent = `⚑ ${warnings.join(' · ')}`; meta.append(flags);
    }
    const words = document.createElement('div'); words.className = 'words';
    if (segment.words?.length) {
      segment.words.forEach((word, wi) => {
        const node = document.createElement('span');
        node.className = 'word'; node.tabIndex = -1; node.dataset.si = si; node.dataset.wi = wi;
        paintWord(node, segment, si, wi);
        words.append(node, ' ');
        wordNodes.push({node, start: word.start, end: word.end, si});
        maxWordDuration = Math.max(maxWordDuration, word.end - word.start);
      });
      words.firstChild.tabIndex = 0;  // one tab stop per segment; arrow keys move between words
    } else {
      const text = document.createElement('textarea');
      text.className = 'segment-text'; text.value = edit.text;
      text.setAttribute('aria-label', `Transcript for segment ${si + 1}`);
      text.oninput = () => { edit.text = text.value; setDirty(true); };
      const note = document.createElement('p'); note.className = 'hint';
      note.textContent = 'No word timing for this segment: playback highlights the whole segment.';
      words.append(text, note);
    }
    row.append(meta, words);
    fragment.append(row);
  });
  root.append(fragment);
  wordNodes.sort((a, b) => a.start - b.start);
  const segs = documentData.transcript.segments;
  const counts = {all: segs.length, flagged: segs.filter(flaggedSegment).length, warnings: segs.filter(warnedSegment).length};
  for (const option of $('review-filter').options) option.textContent = option.textContent.replace(/ · \d+$/, '') + ` · ${counts[option.value]}`;
  filter(); updateHighlight();
}
function paintWord(node, segment, si, wi) {
  const word = segment.words[wi], value = edits.segments[si].words[wi];
  node.textContent = value || '⟨removed⟩';
  node.classList.toggle('deleted', !value);
  node.classList.toggle('edited', value !== baseline(word));
  node.classList.toggle('estimated', !word.aligned);
  node.title = `${clock(word.start)} · ${word.aligned ? 'aligned timing' : 'estimated timing'}`
    + `${value !== baseline(word) ? ` · machine wrote “${baseline(word)}”` : ''}`;
}

// word interactions (event delegation: one listener for ~11,000 words)
$('transcript').addEventListener('click', event => {
  const node = event.target.closest('.word'); if (!node) return;
  const word = documentData.transcript.segments[node.dataset.si].words[node.dataset.wi];
  seek(word.start);
});
$('transcript').addEventListener('dblclick', event => {
  const node = event.target.closest('.word'); if (!node) return;
  editWord(+node.dataset.si, +node.dataset.wi);
});
$('transcript').addEventListener('keydown', event => {
  const node = event.target.closest('.word'); if (!node) return;
  if (event.key === 'Enter') { event.preventDefault(); editWord(+node.dataset.si, +node.dataset.wi); }
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    event.preventDefault(); event.stopPropagation();
    const sibling = event.key === 'ArrowRight' ? node.nextElementSibling : node.previousElementSibling;
    if (sibling?.classList.contains('word')) { node.tabIndex = -1; sibling.tabIndex = 0; sibling.focus(); }
  }
});

function seek(seconds) {
  if (!documentData) return;
  stopAt = null;
  audio.currentTime = Math.max(0, seconds - documentData.offset);
  updateHighlight();
}

function editWord(si, wi) {
  audio.pause(); wordTarget = {si, wi};
  const words = edits.segments[si].words, word = documentData.transcript.segments[si].words[wi];
  const context = $('word-context'); context.replaceChildren();
  words.slice(Math.max(0, wi - 6), wi).forEach(w => context.append(`${w} `));
  const mark = document.createElement('mark'); mark.textContent = words[wi] || '⟨removed⟩'; context.append(mark);
  words.slice(wi + 1, wi + 7).forEach(w => context.append(` ${w}`));
  $('word-time').textContent = `${clock(word.start)} · ${(word.end - word.start).toFixed(2)} s · speaker ${edits.segments[si].speaker}`;
  $('word-original').textContent = baseline(word);
  $('word-restore').hidden = words[wi] === baseline(word);
  $('word-input').value = words[wi];
  $('word-dialog').showModal(); $('word-input').focus(); $('word-input').select();
}
function applyWord(value) {
  const {si, wi} = wordTarget, segment = documentData.transcript.segments[si];
  const edit = edits.segments[si];
  if (edit.words[wi] === value) return;
  edit.words[wi] = value;
  edit.text = edit.words.filter(Boolean).join(' ');
  setDirty(true);
  const node = segmentNodes[si].querySelectorAll('.word')[wi];
  paintWord(node, segment, si, wi);
  node.classList.add('flash'); setTimeout(() => node.classList.remove('flash'), 900);
  filter();
}
// Enter in the input must apply (implicit submission would otherwise use the first button: Cancel)
$('word-input').addEventListener('keydown', event => {
  if (event.key === 'Enter') { event.preventDefault(); $('word-apply').click(); }
});
$('word-form').onsubmit = event => {
  if (event.submitter?.value !== 'save') return;
  applyWord($('word-input').value.trim().replace(/\s+/g, ' '));
};
$('word-dialog').addEventListener('close', () => {
  if (!wordTarget) return;
  const node = segmentNodes[wordTarget.si]?.querySelectorAll('.word')[wordTarget.wi];
  if (node) { node.tabIndex = 0; node.focus({preventScroll: true}); }
});
$('word-restore').onclick = () => {
  const {si, wi} = wordTarget;
  $('word-input').value = baseline(documentData.transcript.segments[si].words[wi]);
  $('word-input').focus();
};
$('word-listen').onclick = () => {
  const {si, wi} = wordTarget, word = documentData.transcript.segments[si].words[wi];
  audio.currentTime = Math.max(0, word.start - documentData.offset - 0.6);
  stopAt = word.end - documentData.offset + 0.4;
  audio.play().catch(error => notice(error.message));
};

// ---- speakers panel ----------------------------------------------------------------------------------------
function renderSpeakers() {
  if (!documentData) return;
  const counts = new Map(), colors = new Map();
  edits.segments.forEach((edit, si) => {
    counts.set(edit.speaker, (counts.get(edit.speaker) || 0) + 1);
    if (!colors.has(edit.speaker)) colors.set(edit.speaker, colorFor(documentData.transcript.segments[si].primary_speaker));
  });
  const list = $('speakers'); list.replaceChildren();
  const names = $('speaker-names'); names.replaceChildren();
  $('speaker-count').textContent = counts.size;
  for (const [label, count] of [...counts].sort((a, b) => b[1] - a[1])) {
    const option = document.createElement('option'); option.value = label; names.append(option);
    const row = document.createElement('div'); row.className = 'speaker-row';
    row.style.setProperty('--speaker', colors.get(label));
    const dot = document.createElement('span'); dot.className = 'dot';
    const input = document.createElement('input'); input.value = label; input.maxLength = 100;
    input.setAttribute('aria-label', `Rename ${label} in all segments`);
    const info = document.createElement('span'); info.className = 'muted'; info.textContent = `${count} segment${count === 1 ? '' : 's'}`;
    let applied = false;
    const apply = () => {
      if (applied) return;
      const value = input.value.trim().replace(/\s+/g, ' ');
      if (!value || value === label) { input.value = label; return; }
      applied = true;
      edits.segments.forEach((edit, si) => { if (edit.speaker === label) { edit.speaker = value; segmentInputs[si].value = value; } });
      setDirty(true); renderSpeakers(); filter();
      notice(`Renamed “${label}” to “${value}” in ${count} segment${count === 1 ? '' : 's'}.`, 'info');
    };
    input.onkeydown = e => { if (e.key === 'Enter') { e.preventDefault(); apply(); } if (e.key === 'Escape') input.value = label; };
    input.onblur = apply;
    row.append(dot, input, info); list.append(row);
  }
}

// ---- playback highlight ------------------------------------------------------------------------------------
function updateHighlight() {
  if (!documentData) return;
  const now = audio.currentTime + documentData.offset, active = new Set(), nextWords = new Set();
  if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; }
  let lo = 0, hi = wordNodes.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (wordNodes[mid].start <= now) lo = mid + 1; else hi = mid; }
  for (let i = lo - 1; i >= 0 && wordNodes[i].start >= now - maxWordDuration; i--) {
    const word = wordNodes[i]; if (now < word.end) { nextWords.add(word.node); active.add(word.si); }
  }
  for (const node of activeWordNodes) if (!nextWords.has(node)) node.classList.remove('active');
  for (const node of nextWords) if (!activeWordNodes.has(node)) node.classList.add('active');
  activeWordNodes = nextWords;
  const segments = documentData.transcript.segments;
  for (const si of lastActive) if (!active.has(si)) segmentNodes[si]?.classList.remove('playing');
  // segments without words (or between words) are matched by their span
  let s = 0, e = segments.length;
  while (s < e) { const mid = (s + e) >>> 1; if (segments[mid].start <= now) s = mid + 1; else e = mid; }
  for (let i = s - 1; i >= Math.max(0, s - 6); i--) if (now >= segments[i].start && now < segments[i].end) active.add(i);
  for (const si of active) segmentNodes[si].classList.add('playing');
  const first = [...active].sort((a, b) => a - b)[0];
  if (first !== undefined && first !== playingSegment) {
    playingSegment = first;
    $('now-speaker').textContent = `${edits.segments[first].speaker} · ${clock(now)}`;
    $('now-dot').style.background = colorFor(segments[first].primary_speaker);
  } else if (first === undefined && !audio.paused) {
    $('now-speaker').textContent = `${clock(now)} · no speech`;
  }
  if ($('follow').checked && !audio.paused && first !== undefined && !lastActive.has(first)) {
    const row = segmentNodes[first], box = row.getBoundingClientRect();
    const top = document.querySelector('.player').getBoundingClientRect().bottom + 8;
    if (!row.hidden && (box.top < top || box.bottom > window.innerHeight - 40)) {
      window.scrollTo({top: window.scrollY + box.top - top - 40, behavior: 'smooth'});
    }
  }
  lastActive = active;
}
function animatePlayback() { updateHighlight(); animation = audio.paused ? null : requestAnimationFrame(animatePlayback); }
audio.addEventListener('timeupdate', updateHighlight);
audio.addEventListener('seeked', updateHighlight);
audio.addEventListener('play', () => { if (animation === null) animation = requestAnimationFrame(animatePlayback); });
audio.addEventListener('pause', () => { if (animation !== null) cancelAnimationFrame(animation); animation = null; updateHighlight(); });
audio.addEventListener('error', () => { if (current && documentData) notice('Audio playback failed. Check that the prepared WAV exists in this recording’s job folder.'); });
$('speed').onchange = () => { audio.playbackRate = Number($('speed').value); };
function skip(delta) {
  const end = Number.isFinite(audio.duration) ? audio.duration : Infinity;
  stopAt = null; audio.currentTime = Math.min(end, Math.max(0, audio.currentTime + delta));
}
$('back').onclick = () => skip(-5);
$('forward').onclick = () => skip(5);

// ---- search and review filters -----------------------------------------------------------------------------
let filterTimer = null;
function filter() {
  if (!documentData) return;
  const query = $('search').value.trim().toLocaleLowerCase();
  const mode = $('review-filter').value;
  let visible = 0, matches = 0;
  documentData.transcript.segments.forEach((seg, si) => {
    const edit = edits.segments[si];
    const text = `${edit.speaker} ${edit.words?.length ? edit.words.join(' ') : edit.text}`.toLocaleLowerCase();
    const hit = !query || text.includes(query);
    const hidden = !hit || !reviewMatch(seg, mode);
    segmentNodes[si].hidden = hidden;
    if (!hidden) visible++;
    if (query && hit) matches++;
  });
  for (const {node} of wordNodes) node.classList.toggle('match', !!query && query.length > 1 && node.textContent.toLocaleLowerCase().includes(query));
  $('match-count').textContent = query ? `${matches} segment${matches === 1 ? '' : 's'}` : '';
  $('empty-results').hidden = visible > 0;
}
$('search').oninput = () => { clearTimeout(filterTimer); filterTimer = setTimeout(filter, 120); };
$('review-filter').onchange = filter;
$('next-review').onclick = () => {
  const top = document.querySelector('.player').getBoundingClientRect().bottom + 20;
  const segments = documentData?.transcript.segments || [];
  const mode = $('review-filter').value === 'flagged' ? 'flagged' : 'warnings';
  const next = segments.findIndex((seg, si) => reviewMatch(seg, mode) && !segmentNodes[si].hidden
    && segmentNodes[si].getBoundingClientRect().top > top + 5);
  if (next < 0) return notice('No more segments to review below this point.', 'info');
  const row = segmentNodes[next];
  window.scrollTo({top: window.scrollY + row.getBoundingClientRect().top - top, behavior: 'smooth'});
  row.classList.add('flash'); setTimeout(() => row.classList.remove('flash'), 1200);
  seek(segments[next].start);
};

// ---- save --------------------------------------------------------------------------------------------------
async function save() {
  if (!dirty || !current) return;
  const id = current, snapshot = JSON.stringify(edits);
  $('save').disabled = true; $('save-state').textContent = 'Saving…'; notice('');
  try {
    const result = await api(`/api/jobs/${id}/save`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: snapshot});
    if (current !== id) return;
    setSavedVersion(result.version);
    if (JSON.stringify(edits) === snapshot) { setDirty(false); $('save-state').textContent = '✓ Version saved'; }
    else { setDirty(true); $('save-state').textContent = 'Newer corrections still unsaved'; }
    notice('Version saved. Download it from the Download menu.', 'info');
  } catch (error) { notice(error.message); setDirty(true); }
}
$('save').onclick = save;
for (const link of document.querySelectorAll('#saved-group a')) link.addEventListener('click', e => {
  if (!savedVersion) { e.preventDefault(); notice('Save a version first.', 'info'); }
});
for (const link of document.querySelectorAll('#download-menu a')) link.addEventListener('click', () => { $('download-menu').open = false; });
document.addEventListener('click', e => { if (!e.target.closest('#download-menu')) $('download-menu').open = false; });
window.addEventListener('beforeunload', event => { if (dirty) { event.preventDefault(); event.returnValue = ''; } });

// ---- keyboard ----------------------------------------------------------------------------------------------
document.addEventListener('keydown', event => {
  const typing = ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName) || event.target.isContentEditable;
  const dialogOpen = $('word-dialog').open || $('confirm-dialog').open;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's' && !$('editor').hidden) {
    event.preventDefault(); save(); return;
  }
  if (typing || dialogOpen || $('editor').hidden) return;
  if (event.key === '/') { event.preventDefault(); $('search').focus(); return; }
  if (['BUTTON', 'A', 'AUDIO', 'SUMMARY'].includes(event.target.tagName)) return;
  if (event.code === 'Space') { event.preventDefault(); if (audio.paused) audio.play().catch(error => notice(error.message)); else audio.pause(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); skip(-5); }
  if (event.key === 'ArrowRight') { event.preventDefault(); skip(5); }
});

// ---- polling -----------------------------------------------------------------------------------------------
async function poll() {
  try {
    await refreshJobs();
    if (connectionLost) { connectionLost = false; notice('Connection to the local server restored.', 'info'); }
    if (current && !$('processing').hidden) {
      const job = jobs.find(j => j.id === current);
      if (job?.status === 'ready') { documentData = null; await openJob(current); }
      else if (job) processing(await api(`/api/jobs/${current}`));
    }
  } catch (error) {
    connectionLost = true;
    notice(`Connection to the local server lost (${error.message}). Is speech-pilot-web still running?`);
  }
  setTimeout(poll, 2500);
}
poll();
