'use strict';
const $ = id => document.getElementById(id);
let jobs = [], current = null, documentData = null, edits = null, dirty = false;
let selectedFile = null, wordTarget = null, generation = 0, wordNodes = [], segmentNodes = [], lastActive = new Set();
let activeWordNodes = new Set(), maxWordDuration = 0, animation = null;
const audio = $('audio');
const steps = [['01_inspect','Inspect'],['02_prepare','Prepare'],['03_vad','Detect speech'],['04_diarization','Speakers'],['05_asr','Transcribe'],['05c_align','Align words'],['09_integrity','Verify']];
function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {'X-Review-Client':'1', ...options.headers}});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Request failed');
  return data;
}
function time(value) { return `${Math.floor(value/60).toString().padStart(2,'0')}:${Math.floor(value%60).toString().padStart(2,'0')}`; }
function show(view) { for (const id of ['welcome','editor','processing']) $(id).hidden = id !== view; }
function markDirty() { dirty = true; $('save-state').textContent = 'Unsaved corrections'; $('downloads').hidden = true; }
function canLeave() { return !dirty || confirm('Leave without saving your corrections?'); }
function selectFile(file) {
  if (!file) return;
  selectedFile = file;
  $('filename').textContent = file.name;
  $('transcribe').disabled = false;
}
$('file').addEventListener('change', event => selectFile(event.target.files[0]));
for (const type of ['dragenter','dragover']) $('drop').addEventListener(type, event => {event.preventDefault(); $('drop').classList.add('dragging');});
for (const type of ['dragleave','drop']) $('drop').addEventListener(type, event => {event.preventDefault(); $('drop').classList.remove('dragging');});
$('drop').addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));
$('new').onclick = () => { if (!canLeave()) return; dirty = false; current = null; generation++; audio.pause(); show('welcome'); renderJobs(); notice(''); };
$('upload-form').onsubmit = async event => {
  event.preventDefault();
  if (!selectedFile) return;
  if (selectedFile.size > 2*1024**3 || !selectedFile.size) return notice('Choose an audio file between 1 byte and 2 GB.');
  $('transcribe').disabled = true; $('transcribe').textContent = 'Uploading recording…'; notice('');
  try {
    const job = await api(`/api/jobs?name=${encodeURIComponent(selectedFile.name)}`, {method:'POST', body:selectedFile});
    await refreshJobs(); await openJob(job.id);
  } catch (error) { notice(error.message); }
  finally { $('transcribe').disabled = false; $('transcribe').textContent = 'Transcribe recording ↗'; }
};
function renderJobs() {
  $('jobs').replaceChildren();
  for (const job of jobs) {
    const button = document.createElement('button'); button.className = `job${current === job.id ? ' selected' : ''}`;
    const name = document.createElement('strong'); name.textContent = job.name;
    const meta = document.createElement('small'); meta.textContent = `${job.status === 'ready' ? '✓ Ready to review' : job.status} · ${new Date(job.created_at).toLocaleDateString()}`;
    button.append(name,meta); button.onclick = () => openJob(job.id).catch(error => notice(error.message)); $('jobs').append(button);
  }
}
async function refreshJobs() { jobs = await api('/api/jobs'); renderJobs(); }
function processing(job) {
  show('processing'); $('process-name').textContent = job.name;
  $('process-state').textContent = job.status === 'failed' ? job.error : job.status === 'queued' ? 'Queued. Recordings are processed one at a time.' : 'Processing on this computer. Longer recordings can take several minutes. You can return to this page later.';
  $('log').textContent = job.log || 'Waiting for the pipeline…'; $('steps').replaceChildren();
  for (const [key,label] of steps) { const node = document.createElement('span'); const done = job.completed_steps.includes(key); node.className = `step${done?' done':''}`; node.textContent = `${done?'✓':'○'} ${label}`; $('steps').append(node); }
}
async function openJob(id) {
  if (current === id && documentData && !$('editor').hidden) return;
  if (!canLeave()) return;
  dirty = false; current = id; documentData = null; audio.pause(); notice(''); const ticket = ++generation; renderJobs();
  const job = await api(`/api/jobs/${id}`); if (ticket !== generation) return;
  if (job.status !== 'ready') { processing(job); return; }
  const data = await api(`/api/jobs/${id}/transcript`); if (ticket !== generation) return;
  documentData = data;
  edits = data.draft || {segments:data.transcript.segments.map(seg => ({id:seg.id, speaker:seg.primary_speaker || 'UNKNOWN', text:seg.text_normalized ?? seg.text, words:(seg.words || []).map(word => word.text_normalized ?? word.text)}))};
  $('recording-name').textContent = job.name;
  const speakers = new Set(data.transcript.segments.map(s => s.primary_speaker));
  $('summary').textContent = `${time(data.transcript.source.duration_s)} recording · ${speakers.size} speakers · ${data.transcript.language.toUpperCase()}`;
  $('segment-count').textContent = data.transcript.segments.length;
  $('save-state').textContent = data.draft ? 'Saved corrections loaded' : 'Original transcript'; $('downloads').hidden = true;
  $('search').value = ''; $('warnings').checked = false;
  audio.src = `/api/jobs/${id}/audio`; audio.playbackRate = Number($('speed').value); show('editor'); renderTranscript();
}
function seek(seconds) { audio.currentTime = Math.max(0, seconds - documentData.offset); updateHighlight(); }
function renderTranscript() {
  $('transcript').replaceChildren(); wordNodes = []; segmentNodes = []; lastActive = new Set(); activeWordNodes = new Set(); maxWordDuration = 0;
  documentData.transcript.segments.forEach((segment, si) => {
    const edit = edits.segments[si];
    const row = document.createElement('article'); row.className = 'segment'; segmentNodes.push(row);
    const meta = document.createElement('div'); meta.className = 'segment-meta';
    const speaker = document.createElement('input'); speaker.className = 'speaker'; speaker.value = edit.speaker; speaker.maxLength = 100; speaker.setAttribute('aria-label', `Speaker for segment ${si+1}`);
    speaker.oninput = () => { edit.speaker = speaker.value; markDirty(); };
    const timestamp = document.createElement('button'); timestamp.className = 'timestamp'; timestamp.textContent = `${time(segment.start)} — ${time(segment.end)}`; timestamp.onclick = () => seek(segment.start);
    meta.append(speaker,timestamp);
    const warnings = (segment.flags || []).filter(f => f.severity === 'warning').map(f => f.code.replaceAll('_',' '));
    if (segment.overlap) warnings.push('overlapping speech');
    if (segment.speaker_uncertain) warnings.push('check speaker');
    if (warnings.length) {const flags = document.createElement('div'); flags.className = 'flags'; flags.textContent = `⚑ ${[...new Set(warnings)].join(' · ')}`; meta.append(flags);}
    const words = document.createElement('div'); words.className = 'words';
    if (segment.words?.length) {
      segment.words.forEach((word, wi) => {
        const node = document.createElement('span'); node.className = 'word'; node.tabIndex = 0; node.setAttribute('role','button');
        node.textContent = edit.words[wi] || '⟨removed⟩';
        node.classList.toggle('deleted', !edit.words[wi]); node.classList.toggle('edited', edit.words[wi] !== (word.text_normalized ?? word.text)); node.classList.toggle('estimated', !word.aligned);
        node.title = `${time(word.start)} · ${word.aligned ? 'Aligned timing' : 'Estimated timing'} · Double-click to edit`;
        node.onclick = () => seek(word.start); node.ondblclick = () => editWord(si,wi);
        node.onkeydown = event => {if(event.key === 'Enter') {event.preventDefault(); editWord(si,wi);}};
        words.append(node,document.createTextNode(' ')); wordNodes.push({node,start:word.start,end:word.end,si}); maxWordDuration = Math.max(maxWordDuration,word.end-word.start);
      });
    } else {
      const text = document.createElement('textarea'); text.className = 'segment-text'; text.value = edit.text; text.setAttribute('aria-label',`Transcript for segment ${si+1}`); text.oninput = () => {edit.text = text.value; markDirty();};
      const note = document.createElement('p'); note.className='hint'; note.textContent='This segment has no word alignment. Playback highlights the segment.'; words.append(text,note);
    }
    row.append(meta,words); $('transcript').append(row);
  });
  wordNodes.sort((a,b)=>a.start-b.start); filter(); updateHighlight();
}
function editWord(si,wi) {
  audio.pause(); wordTarget = {si,wi};
  $('word-context').textContent = edits.segments[si].words.slice(Math.max(0,wi-4),wi+5).join(' ');
  $('word-input').value = edits.segments[si].words[wi]; $('word-dialog').showModal(); $('word-input').focus(); $('word-input').select();
}
$('word-form').onsubmit = event => {
  if (event.submitter?.value !== 'save') return;
  const {si,wi} = wordTarget; edits.segments[si].words[wi] = $('word-input').value.trim();
  edits.segments[si].text = edits.segments[si].words.filter(Boolean).join(' '); markDirty(); renderTranscript();
};
function updateHighlight() {
  if (!documentData) return;
  const now = audio.currentTime + documentData.offset, active = new Set();
  const nextWords = new Set();
  let lo = 0, hi = wordNodes.length;
  while(lo < hi) {const mid = (lo+hi) >>> 1; if(wordNodes[mid].start <= now) lo=mid+1; else hi=mid;}
  for(let i=lo-1;i>=0 && wordNodes[i].start >= now-maxWordDuration;i--) {
    const word=wordNodes[i]; if(now < word.end) {nextWords.add(word.node);active.add(word.si);}
  }
  for(const node of activeWordNodes) if(!nextWords.has(node))node.classList.remove('active');
  for(const node of nextWords) if(!activeWordNodes.has(node))node.classList.add('active');
  activeWordNodes=nextWords;
  documentData.transcript.segments.forEach((seg,si) => { const playing = now >= seg.start && now < seg.end; segmentNodes[si].classList.toggle('playing',playing); if (playing) active.add(si); });
  if ($('follow').checked && !audio.paused && active.size && [...active].some(i => !lastActive.has(i))) {
    const row = segmentNodes[[...active][0]]; if (!row.hidden) row.scrollIntoView({behavior:'smooth',block:'center'});
  }
  lastActive = active;
}
function filter() {
  if (!documentData) return;
  const query = $('search').value.toLocaleLowerCase(); let visible=0;
  documentData.transcript.segments.forEach((seg,si) => {
    const edit = edits.segments[si]; const flagged = seg.speaker_uncertain || seg.overlap || seg.flags.some(f => f.severity === 'warning');
    const hidden = (!`${edit.speaker} ${edit.text}`.toLocaleLowerCase().includes(query)) || ($('warnings').checked && !flagged);
    segmentNodes[si].hidden = hidden; if(!hidden)visible++;
  }); $('empty-results').hidden = visible > 0;
}
$('search').oninput = filter; $('warnings').onchange = filter;
audio.addEventListener('timeupdate',updateHighlight); audio.addEventListener('seeked',updateHighlight);
function animatePlayback() {updateHighlight(); animation=audio.paused ? null : requestAnimationFrame(animatePlayback);}
audio.addEventListener('play',()=>{if(animation===null) animation=requestAnimationFrame(animatePlayback);});
audio.addEventListener('pause',()=>{if(animation!==null)cancelAnimationFrame(animation);animation=null;updateHighlight();});
$('speed').onchange = () => {audio.playbackRate = Number($('speed').value);};
$('back').onclick = () => {audio.currentTime=Math.max(0,audio.currentTime-5);};
$('forward').onclick = () => {if(Number.isFinite(audio.duration)) audio.currentTime=Math.min(audio.duration,audio.currentTime+5);};
audio.addEventListener('error', () => {if(current && documentData) notice('Audio playback failed. Check that the prepared WAV exists in this recording’s job folder.');});
document.addEventListener('keydown',event => {if(event.code === 'Space' && !$('editor').hidden && !$('word-dialog').open && !['INPUT','TEXTAREA','SELECT','BUTTON','A','AUDIO'].includes(event.target.tagName)){event.preventDefault(); if(audio.paused)audio.play().catch(error=>notice(error.message));else audio.pause();}});
$('save').onclick = async () => {
  const id = current, snapshot = JSON.stringify(edits); $('save').disabled = true; notice('');
  try {
    const result = await api(`/api/jobs/${id}/save`,{method:'POST',headers:{'Content-Type':'application/json'},body:snapshot});
    if(current !== id) return;
    if(JSON.stringify(edits) === snapshot) {dirty=false; $('save-state').textContent='All corrections saved';} else {$('save-state').textContent='Newer corrections still unsaved';}
    for (const kind of ['txt','srt','json']) $('download-'+kind).href = `/api/jobs/${id}/download?version=${result.version}&format=${kind}`;
    $('downloads').hidden=false;
  } catch(error) {notice(error.message);} finally {$('save').disabled=false;}
};
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
async function poll() {
  try {
    await refreshJobs();
    if(current && !$('processing').hidden) {
      const job = jobs.find(j=>j.id===current);
      if(job?.status==='ready') await openJob(current); else if(job)processing(job);
    }
  } catch(error) {notice(`Connection to local server lost: ${error.message}`);}
  setTimeout(poll,2500);
}
poll();
