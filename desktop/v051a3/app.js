const $ = id => document.getElementById(id);
const intents = [
  'COPY_VALUE', 'SUPERSEDES', 'MAGNITUDE', 'NEGATE', 'REFERENCE',
  'ACTIVE_SELECTION', 'RUN_STATUS', 'RECEIPT_VALUE', 'EVIDENCE_CONSISTENCY',
  'GOAL', 'CONSTRAINT', 'OBSERVATION', 'PREDICTION'
];
$('intent').innerHTML = intents.map(value => `<option>${value}</option>`).join('');

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET',
    headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined
  });
  const payload = await response.json();
  if (!response.ok) throw Error(payload.message || payload.error || 'Request failed');
  return payload;
}

function add(role, text, meta) {
  const item = document.createElement('div');
  item.className = `msg ${role}`;
  item.textContent = text;
  if (meta) {
    const detail = document.createElement('div');
    detail.className = 'meta';
    detail.textContent = meta;
    item.appendChild(detail);
  }
  $('chat').appendChild(item);
  $('chat').scrollTop = $('chat').scrollHeight;
}

function proposalKinds(turn) {
  const proposals = turn.meta?.proposal?.proposals || [];
  return proposals.map(item => item.kind).join(', ');
}

async function refresh() {
  const status = await api('/api/status');
  $('status').textContent = `${status.facts} facts · ${status.typed_memory} frames · ${status.draft_lessons} drafts · ${status.reviewed_lessons} reviewed`;
  const history = await api('/api/history');
  $('chat').innerHTML = '';
  for (const item of history.messages) {
    const kinds = (item.meta?.frames || []).map(frame => frame.kind).join(', ');
    add(item.role, item.content, [item.meta?.route, kinds].filter(Boolean).join(' · '));
  }
}

$('form').onsubmit = async event => {
  event.preventDefault();
  const message = $('message').value.trim();
  if (!message) return;
  add('user', message);
  $('message').value = '';
  try {
    const turn = await api('/api/chat', {message});
    add('assistant', turn.response, [turn.route, proposalKinds(turn)].filter(Boolean).join(' · '));
    await refresh();
  } catch (error) {
    add('assistant', `Request failed: ${error.message}`, 'ERROR');
  }
};

$('newSession').onclick = async () => {
  await api('/api/new-session', {});
  await refresh();
};

$('teach').onclick = async () => {
  const text = $('lesson').value.trim();
  if (!text) return;
  const draft = await api('/api/teach', {intent: $('intent').value, text});
  $('lesson').value = '';
  $('lessonId').value = draft.id;
  $('lessonStatus').textContent = `Draft ${draft.id} created. It is not training data until reviewed.`;
  await refresh();
};

$('review').onclick = async () => {
  const lessonId = $('lessonId').value.trim();
  const reviewedBy = $('reviewer').value.trim();
  const permission = $('permission').value.trim();
  if (!lessonId || !reviewedBy || !permission) {
    $('lessonStatus').textContent = 'Lesson ID, reviewer, and permission provenance are required.';
    return;
  }
  const lesson = await api('/api/review', {
    lesson_id: lessonId,
    reviewed_by: reviewedBy,
    permission
  });
  $('lessonStatus').textContent = `${lesson.id} is REVIEWED and may now be exported deliberately.`;
  await refresh();
};

refresh().catch(error => add('assistant', `Unable to load runtime: ${error.message}`, 'ERROR'));
