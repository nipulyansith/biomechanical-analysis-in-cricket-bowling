/**
 * PitchIQ - shared UI helpers.
 * Loaded after api-client.js on every page. Consolidates the small
 * utility functions that were previously copy-pasted per-page.
 */

/** Navigate back to the results hub, carrying the session id along. */
function goBackToResults(event) {
  if (event) event.preventDefault();
  const sessionId = window.API?.sessionId || localStorage.getItem('lastSessionId');
  window.location.href = sessionId ? `results.html?session=${sessionId}` : 'results.html';
}

/** Show a transient toast message (error by default). */
function showToast(message, variant = 'error') {
  const toast = document.createElement('div');
  toast.className = `toast ${variant === 'error' ? 'toast--error' : ''}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/** Format a number to fixed decimals, tolerating null/undefined/NaN. */
function formatNumber(value, decimals = 2, fallback = '--') {
  if (value === null || value === undefined || isNaN(value)) return fallback;
  return parseFloat(value).toFixed(decimals);
}

/** Auto-init: fade '.reveal' elements in as they enter the viewport. */
function initScrollReveal() {
  const els = document.querySelectorAll('.reveal');
  if (!els.length) return;
  if (!('IntersectionObserver' in window)) {
    els.forEach(el => el.classList.add('is-visible'));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: '0px 0px -40px 0px' });
  els.forEach(el => io.observe(el));
}

/** Auto-init: subtle cursor-follow tilt on hero visualization art (desktop only). */
function initHeroParallax() {
  const layers = document.querySelectorAll('[data-depth]');
  if (!layers.length || window.matchMedia('(pointer: coarse)').matches) return;
  const hero = layers[0].closest('[data-parallax-root]') || layers[0].closest('.cine-hero') || document.body;
  hero.addEventListener('mousemove', (e) => {
    const rect = hero.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width - 0.5;
    const py = (e.clientY - rect.top) / rect.height - 0.5;
    layers.forEach(layer => {
      const depth = parseFloat(layer.dataset.depth || '14');
      layer.style.transform = `translate(${px * depth}px, ${py * depth}px)`;
    });
  });
  hero.addEventListener('mouseleave', () => {
    layers.forEach(layer => { layer.style.transform = ''; });
  });
}

/**
 * BowlingActionVisualization — the product's signature component.
 * Renders a technical cricket-pitch diagram (bowler's-eye perspective: crease and
 * stumps at both ends) with a looping ball-delivery animation down the pitch, plus
 * floating instrument-style data readouts. No human figure of any kind — pure
 * geometry + data. Numeric readouts only render when real data is supplied.
 *
 * opts: { elbow?: number, knee?: number, comDx?: number, comDy?: number, showBall?: boolean }
 */
function renderBowlingViz(container, opts = {}) {
  const el = typeof container === 'string' ? document.getElementById(container) : container;
  if (!el) return;

  const elbowLabel = (opts.elbow !== null && opts.elbow !== undefined && !isNaN(opts.elbow))
    ? `<g class="viz-label" transform="translate(48,180)">
         <rect x="0" y="0" width="88" height="40" rx="8" fill="#0b0f17" fill-opacity="0.85" stroke="#232a3a"/>
         <text x="10" y="16" class="viz-label__key">ELBOW</text>
         <text x="10" y="33" class="viz-label__val">${Math.round(opts.elbow)}°</text>
       </g>`
    : '';

  const kneeLabel = (opts.knee !== null && opts.knee !== undefined && !isNaN(opts.knee))
    ? `<g class="viz-label" transform="translate(344,180)">
         <rect x="0" y="0" width="82" height="40" rx="8" fill="#0b0f17" fill-opacity="0.85" stroke="#232a3a"/>
         <text x="10" y="16" class="viz-label__key">KNEE</text>
         <text x="10" y="33" class="viz-label__val">${Math.round(opts.knee)}°</text>
       </g>`
    : '';

  const hasCom = opts.comDx !== null && opts.comDx !== undefined && !isNaN(opts.comDx);
  const comBlock = `
    <g class="viz-label">
      <circle cx="150" cy="430" r="10" fill="none" stroke="#5ec8ff" stroke-width="1.5" stroke-dasharray="3 3"/>
      <circle cx="150" cy="430" r="2" fill="#5ec8ff"/>
      <text x="167" y="427" class="viz-label__key">COM</text>
      ${hasCom ? `<text x="167" y="443" class="viz-label__val" style="font-size:11px">${formatNumber(opts.comDx, 1)}cm</text>` : ''}
    </g>`;

  el.innerHTML = `
<svg class="bowling-viz__svg" viewBox="0 0 480 600" fill="none" xmlns="http://www.w3.org/2000/svg" data-depth="10">
  <defs>
    <radialGradient id="vizAura" cx="50%" cy="55%" r="55%">
      <stop offset="0%" stop-color="#2e8fff" stop-opacity="0.28"/>
      <stop offset="100%" stop-color="#2e8fff" stop-opacity="0"/>
    </radialGradient>
    <filter id="vizAuraBlur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="30"/></filter>
    <filter id="vizBallGlow" x="-120%" y="-120%" width="340%" height="340%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <ellipse class="viz-aura-pulse" cx="240" cy="340" rx="190" ry="230" fill="url(#vizAura)" filter="url(#vizAuraBlur)"/>

  <!-- Cricket pitch, bowler's-eye perspective: near crease (bottom) to far crease (top) -->
  <g class="viz-pitch">
    <path d="M70 548 L170 133 L310 133 L410 548 Z" stroke="#232a3a" stroke-width="1.5" opacity="0.7"/>
    <line x1="240" y1="548" x2="240" y2="133" stroke="#232a3a" stroke-width="1" stroke-dasharray="3 6" opacity="0.5"/>

    <!-- near (bowler's end) crease + stumps -->
    <line x1="70" y1="548" x2="410" y2="548" stroke="#4fd6ff" stroke-width="2" opacity="0.85"/>
    <g transform="translate(240,548)" opacity="0.95">
      <line x1="-16" y1="0" x2="-16" y2="-70" stroke="#8fa3c8" stroke-width="4"/>
      <line x1="0" y1="0" x2="0" y2="-70" stroke="#eef3fb" stroke-width="4"/>
      <line x1="16" y1="0" x2="16" y2="-70" stroke="#8fa3c8" stroke-width="4"/>
      <line x1="-19" y1="-73" x2="19" y2="-73" stroke="#eef3fb" stroke-width="4"/>
    </g>

    <!-- far (batsman's end) crease + stumps -->
    <line x1="170" y1="133" x2="310" y2="133" stroke="#4fd6ff" stroke-width="1.4" opacity="0.7"/>
    <g transform="translate(240,133)" opacity="0.85">
      <line x1="-7" y1="0" x2="-7" y2="-30" stroke="#8fa3c8" stroke-width="2"/>
      <line x1="0" y1="0" x2="0" y2="-30" stroke="#eef3fb" stroke-width="2"/>
      <line x1="7" y1="0" x2="7" y2="-30" stroke="#8fa3c8" stroke-width="2"/>
      <line x1="-8" y1="-32" x2="8" y2="-32" stroke="#eef3fb" stroke-width="2"/>
    </g>
  </g>

  ${opts.showBall !== false ? `
  <path class="viz-ball-trail" d="M240 500 C 230 400, 250 250, 240 148" stroke="#5ec8ff" stroke-width="1.5" stroke-dasharray="3 7" fill="none" opacity="0.5"/>
  <circle class="viz-ball" cx="240" cy="500" r="7" fill="#e5484d" filter="url(#vizBallGlow)"/>
  ` : ''}

  ${comBlock}
  ${elbowLabel}
  ${kneeLabel}
</svg>`;
}

document.addEventListener('DOMContentLoaded', () => {
  initScrollReveal();
  initHeroParallax();
});
