// ─── COPD DATA ────────────────────────────────────────────
const COPD = {
  0: {
    sublabel: 'COPD0', name: 'Bajo Riesgo', icon: '✓',
    color: '#22c55e', bg: 'rgba(34,197,94,0.1)', border: 'rgba(34,197,94,0.3)',
    conf: '88%',
    fev1fvc: 'Normal (≥ 70%)', fev1: 'Normal (≥ 80%)', pattern: 'PFT normal',
    rec: 'Seguimiento normal — controles anuales',
    recBody: 'No se detectaron patrones de obstrucción pulmonar. Continúe con controles anuales de salud pulmonar. Se observan síntomas crónicos que deben ser monitoreados. Evite factores de riesgo como tabaco y exposición a contaminantes.',
    recColor: 'rgba(34,197,94,0.12)', recBorder: 'rgba(34,197,94,0.3)',
    recIconBg: 'rgba(34,197,94,0.15)', recIconColor: '#22c55e',
    emergency: false
  },
  1: {
    sublabel: 'COPD1', name: 'Nivel Leve', icon: '!',
    color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', border: 'rgba(245,158,11,0.3)',
    conf: '77%',
    fev1fvc: '< 70%', fev1: '≥ 80%', pattern: 'Obstrucción leve',
    rec: 'Observación recomendada',
    recBody: 'Se detectaron signos iniciales de obstrucción pulmonar leve. Se recomienda evaluación pulmonar formal mediante espirometría en los próximos 3 meses. Consulte su médico de cabecera para seguimiento.',
    recColor: 'rgba(245,158,11,0.08)', recBorder: 'rgba(245,158,11,0.3)',
    recIconBg: 'rgba(245,158,11,0.15)', recIconColor: '#f59e0b',
    emergency: false
  },
  2: {
    sublabel: 'COPD2', name: 'Nivel Moderado', icon: '2',
    color: '#f97316', bg: 'rgba(249,115,22,0.1)', border: 'rgba(249,115,22,0.3)',
    conf: '84%',
    fev1fvc: '< 70%', fev1: '50–80%', pattern: 'Obstrucción moderada',
    rec: 'Acudir al médico esta semana',
    recBody: 'Se detectaron patrones de obstrucción moderada en la auscultación pulmonar. Es necesaria una evaluación médica oportuna para confirmar el diagnóstico y definir un plan de tratamiento. Evite actividad física intensa hasta la consulta.',
    recColor: 'rgba(249,115,22,0.08)', recBorder: 'rgba(249,115,22,0.3)',
    recIconBg: 'rgba(249,115,22,0.15)', recIconColor: '#f97316',
    emergency: false
  },
  3: {
    sublabel: 'COPD3', name: 'Nivel Grave', icon: '!',
    color: '#ef4444', bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.3)',
    conf: '91%',
    fev1fvc: '< 70%', fev1: '30–50%', pattern: 'Obstrucción grave',
    rec: 'Consulta médica urgente — 24 a 48 horas',
    recBody: 'Patrón respiratorio grave detectado. Requiere atención médica urgente. Diríjase a urgencias o contacte a su médico tratante de inmediato. No realice esfuerzo físico. Si presenta disnea en reposo, llame a emergencias.',
    recColor: 'rgba(239,68,68,0.08)', recBorder: 'rgba(239,68,68,0.35)',
    recIconBg: 'rgba(239,68,68,0.15)', recIconColor: '#ef4444',
    emergency: false
  },
  4: {
    sublabel: 'COPD4', name: 'Nivel Muy Grave', icon: '⚠',
    color: '#dc2626', bg: 'rgba(153,27,27,0.15)', border: 'rgba(220,38,38,0.5)',
    conf: '96%',
    fev1fvc: '< 70%', fev1: '< 30% (o < 50% con I.R.)', pattern: 'Insuficiencia respiratoria crónica',
    rec: '🚨 Atención de emergencia inmediata',
    recBody: 'ESTADO CRÍTICO: Se detectó patrón de insuficiencia respiratoria severa. Acuda INMEDIATAMENTE a urgencias o active el número de emergencias 123. No conduzca. Permanezca sentado o semi-incorporado.',
    recColor: 'rgba(153,27,27,0.15)', recBorder: 'rgba(220,38,38,0.5)',
    recIconBg: 'rgba(220,38,38,0.2)', recIconColor: '#ef4444',
    emergency: true
  }
};

// ─── NAVIGATION ───────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('screen-' + id);
  if (el) el.classList.add('active');
}

function setActiveNav(btn) {
  document.querySelectorAll('.nav-links button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function scrollToAbout() {
  showScreen('home');
  setTimeout(() => document.getElementById('about').scrollIntoView({ behavior: 'smooth' }), 50);
}

// ─── ROLE LOGIC ───────────────────────────────────────────
let currentRole = 'medico';
let selectedFile = null;   // audio cargado para análisis real

function startAsRole(role) {
  currentRole = role;
  selectRole(role);
  showScreen('upload');
}

function selectRole(role) {
  currentRole = role;
  document.getElementById('pill-medico').classList.toggle('selected', role === 'medico');
  document.getElementById('pill-paciente').classList.toggle('selected', role === 'paciente');
}

// ─── FILE HANDLING ────────────────────────────────────────
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('audio')) handleFile(file);
}

function handleFile(file) {
  if (!file) return;
  selectedFile = file;
  const ext = file.name.split('.').pop().toUpperCase();
  const size = (file.size / 1024 / 1024).toFixed(1) + ' MB';
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileMeta').textContent = `${size} · ${ext}`;
  document.getElementById('filePreview').classList.add('show');
  drawMiniWave();
}

function removeFile() {
  selectedFile = null;
  document.getElementById('filePreview').classList.remove('show');
}

function drawMiniWave() {
  const canvas = document.getElementById('miniWave');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
  ctx.strokeStyle = '#14b8a6';
  ctx.lineWidth = 1.5;
  for (let x = 0; x < canvas.width; x++) {
    const y = canvas.height / 2
      + Math.sin(x * 0.15) * 10
      + Math.sin(x * 0.05) * 6
      + (Math.random() - 0.5) * 4;
    x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ─── PROCESSING SIMULATION ────────────────────────────────
let procTimer = null;

const STEPS_CONFIG = [
  { done: true,  active: false, badge: 'Completado', name: 'Carga de audio',               progress: 20  },
  { done: false, active: true,  badge: 'Procesando', name: 'Preprocesamiento de señal',    progress: 40  },
  { done: false, active: false, badge: 'En espera',  name: 'Extracción de features MFCC', progress: 60  },
  { done: false, active: false, badge: 'En espera',  name: 'Modelo de clasificación ML',  progress: 80  },
  { done: false, active: false, badge: 'En espera',  name: 'Determinación de triage EPOC', progress: 100 }
];

function updateResultMeta(name, age) {
  const meta = document.getElementById('resultMeta');
  if (!meta) return;
  const roleLabel = currentRole === 'paciente' ? 'Paciente' : 'Médico General';
  const fecha = new Date().toLocaleDateString('es-CO', { day: 'numeric', month: 'short', year: 'numeric' });
  meta.innerHTML =
    `<strong>Paciente:</strong> ${name || '—'} &nbsp;·&nbsp; ` +
    `<strong>Edad:</strong> ${age ? age + ' años' : '—'} &nbsp;·&nbsp; ` +
    `<strong>Fecha:</strong> ${fecha} &nbsp;·&nbsp; ` +
    `<strong>Rol:</strong> ${roleLabel}`;
}

function startProcessing() {
  // Exige un archivo real: sin audio NO hay resultado (evita el COPD2 de demo).
  if (!selectedFile) {
    alert('Primero sube un archivo de audio (.wav) de auscultación pulmonar.');
    return;
  }

  showScreen('processing');
  startWaveAnimation();

  // Datos del paciente (para enviar y para mostrar en el resultado)
  const pName = (document.getElementById('patientName') || {}).value || '';
  const pAge = (document.getElementById('patientAge') || {}).value || '';

  const fd = new FormData();
  fd.append('audio', selectedFile);
  fd.append('role', currentRole);
  if (pName.trim()) fd.append('patient_name', pName.trim());
  if (pAge) fd.append('patient_age', pAge);
  const symptoms = Array.from(document.querySelectorAll('.symptom-check input:checked'))
    .map(c => (c.parentElement.textContent || '').trim()).filter(Boolean);
  if (symptoms.length) fd.append('symptoms', symptoms.join(', '));

  const resultPromise = fetch('/api/v1/analyze', { method: 'POST', body: fd })
    .then(r => r.ok ? r.json() : r.json().then(e => Promise.reject(e)))
    .catch(err => { console.error('Error de análisis:', err); return null; });

  let step = 1;
  if (procTimer) clearInterval(procTimer);

  procTimer = setInterval(() => {
    advanceStep(step);
    step++;
    if (step >= 5) {
      clearInterval(procTimer);
      resultPromise.then(api => {
        setTimeout(() => {
          stopWaveAnimation();
          if (!api || !Number.isInteger(api.copd_level)) {
            alert('No se pudo analizar el audio. Verifica que sea una auscultación válida (.wav).');
            showScreen('upload');
            return;
          }
          updateResultMeta(pName, pAge);
          showResult(api.copd_level, api);  // SIEMPRE el resultado real del modelo
          showScreen('results');
        }, 600);
      });
    }
  }, 1400);
}

function advanceStep(activeStep) {
  const items = document.querySelectorAll('.step-item');
  const bar = document.getElementById('procBar');
  const pct = document.getElementById('procPercent');
  const prog = [20, 40, 60, 80, 100];

  items.forEach((item, i) => {
    item.classList.remove('active', 'done');
    const icon = item.querySelector('.step-icon');
    const badge = item.querySelector('.step-badge');
    if (i < activeStep) {
      item.classList.add('done');
      icon.textContent = '✓';
      badge.textContent = 'Completado';
    } else if (i === activeStep) {
      item.classList.add('active');
      icon.textContent = '⟳';
      badge.textContent = 'Procesando';
    } else {
      icon.textContent = i + 1;
      badge.textContent = 'En espera';
    }
  });

  const p = prog[activeStep] || 100;
  bar.style.width = p + '%';
  pct.textContent = p + '% completado';
}

// ─── WAVEFORM ANIMATION ───────────────────────────────────
let waveRaf = null;
let waveT = 0;

function startWaveAnimation() {
  const canvas = document.getElementById('waveCanvas');
  if (!canvas) return;
  canvas.width = canvas.offsetWidth || 640;

  function draw() {
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // gradient fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(20,184,166,0.3)');
    grad.addColorStop(1, 'rgba(20,184,166,0)');

    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const nx = x / w;
      const y = h / 2
        + Math.sin(nx * 12 + waveT) * 22
        + Math.sin(nx * 25 + waveT * 1.5) * 10
        + Math.sin(nx * 50 + waveT * 0.7) * 5
        + Math.sin(nx * 4 + waveT * 0.3) * 12;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    // fill under wave
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();

    // stroke
    ctx.beginPath();
    ctx.strokeStyle = '#14b8a6';
    ctx.lineWidth = 2;
    for (let x = 0; x < w; x++) {
      const nx = x / w;
      const y = h / 2
        + Math.sin(nx * 12 + waveT) * 22
        + Math.sin(nx * 25 + waveT * 1.5) * 10
        + Math.sin(nx * 50 + waveT * 0.7) * 5
        + Math.sin(nx * 4 + waveT * 0.3) * 12;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    waveT += 0.06;
    waveRaf = requestAnimationFrame(draw);
  }
  draw();
}

function stopWaveAnimation() {
  if (waveRaf) { cancelAnimationFrame(waveRaf); waveRaf = null; }
}

// ─── RESULT RENDERING ────────────────────────────────────
function showResult(level, api) {
  const d = COPD[level];

  // badge card
  const badge = document.getElementById('resultBadge');
  badge.style.background = d.bg;
  badge.style.borderColor = d.border;

  const icon = document.getElementById('resultIcon');
  icon.textContent = d.icon;
  icon.style.background = d.bg;
  icon.style.color = d.color;
  icon.style.border = `2px solid ${d.border}`;

  document.getElementById('resultSublabel').textContent = d.sublabel;
  document.getElementById('resultSublabel').style.color = d.color;
  document.getElementById('resultName').textContent = d.name;
  document.getElementById('resultDesc').textContent = d.recBody.substring(0, 80) + '…';
  document.getElementById('resultConf').textContent =
    (api && typeof api.confidence === 'number') ? (Math.round(api.confidence * 100) + '%') : d.conf;

  // metrics
  document.getElementById('metFev1fvc').textContent = d.fev1fvc;
  document.getElementById('metFev1').textContent = d.fev1;
  document.getElementById('metPattern').textContent = d.pattern;

  // gauge
  for (let i = 0; i <= 4; i++) {
    document.getElementById('seg' + i).classList.toggle('active', i === level);
    document.getElementById('tick' + i).classList.toggle('active', i === level);
  }

  // recommendation card
  const rec = document.getElementById('recCard');
  rec.style.background = d.recColor;
  rec.style.borderColor = d.recBorder;
  document.getElementById('recIcon').style.background = d.recIconBg;
  document.getElementById('recIcon').style.color = d.recIconColor;
  document.getElementById('recTitle').textContent = d.rec;
  document.getElementById('recBody').textContent = d.recBody;

  // emergency button
  document.getElementById('emergencyBtn').style.display = d.emergency ? 'flex' : 'none';

  // demo chips
  for (let i = 0; i <= 4; i++) {
    const chip = document.getElementById('chip' + i);
    const c = COPD[i];
    chip.classList.toggle('selected', i === level);
    chip.style.background = i === level ? c.color : 'transparent';
    chip.style.borderColor = c.color;
    chip.style.color = i === level ? '#0b1f3a' : c.color;
  }
}

// ─── VIDEO INTRO ──────────────────────────────────────────
(function initIntroVideo() {
  const video     = document.getElementById('introVideo');
  const fillBar   = document.getElementById('voProgressFill');
  const skipCount = document.getElementById('voSkipCount');

  if (!video) return;

  // Barra de progreso + contador regresivo mientras corre el video
  video.addEventListener('timeupdate', () => {
    if (!video.duration) return;
    const pct = (video.currentTime / video.duration) * 100;
    fillBar.style.width = pct + '%';
    const remaining = Math.ceil(video.duration - video.currentTime);
    skipCount.textContent = remaining > 0 ? remaining + 's' : '';
  });

  // Al terminar el video se cierra solo
  video.addEventListener('ended', closeIntro);
})();

function playIntroVideo() {
  const video   = document.getElementById('introVideo');
  const playOvl = document.getElementById('voPlayOverlay');
  if (!video) return;
  video.play();
  playOvl.classList.add('hidden');
}

function closeIntro() {
  const overlay = document.getElementById('videoOverlay');
  const video   = document.getElementById('introVideo');
  if (!overlay || overlay.classList.contains('vo-closing')) return;
  if (video) video.pause();
  overlay.classList.add('vo-closing');
  setTimeout(() => { if (overlay.parentNode) overlay.remove(); }, 750);
}

// ─── MOBILE SIDEBAR ───────────────────────────────────────
function openMobileMenu() {
  document.getElementById('mobSidebar').classList.add('open');
  document.getElementById('mobOverlay').classList.add('open');
  document.getElementById('hamburgerBtn').classList.add('open');
  document.body.style.overflow = 'hidden';
}

function closeMobileMenu() {
  document.getElementById('mobSidebar').classList.remove('open');
  document.getElementById('mobOverlay').classList.remove('open');
  document.getElementById('hamburgerBtn').classList.remove('open');
  document.body.style.overflow = '';
}

// Navega Y cierra el sidebar, actualizando el ítem activo
function navMobile(screenId) {
  showScreen(screenId);
  // Actualiza ítem activo en la sidebar
  document.querySelectorAll('.mob-nav-item').forEach(b => b.classList.remove('mob-active'));
  const btn = document.getElementById('mob-btn-' + screenId);
  if (btn) btn.classList.add('mob-active');
  closeMobileMenu();
}

// Cierra sidebar al presionar Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeMobileMenu();
});

// ─── INIT ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  showResult(2); // pre-render results at COPD2
});