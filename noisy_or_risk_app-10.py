"""
Noisy-OR Bayesian Risk Model - Interactive Web App
Run with: python noisy_or_risk_app.py
Then open http://localhost:5000 in your browser

Requirements: pip install flask numpy
"""

from flask import Flask, render_template_string, request, jsonify
import numpy as np

app = Flask(__name__)

N_SAMPLES_DEFAULT = 20_000


def run_simulation(sources, n_samples):
    """
    sources: list of dicts with 'mu', 'kappa', and 'active' keys.
    Only active sources are combined via Noisy-OR.
    source_means returned for ALL sources (active or not) so the UI can
    still display individual marginal means regardless of active state.
    """
    rng = np.random.default_rng()  # fresh seed each call

    all_probs = []
    active_probs = []
    for src in sources:
        mu    = float(src["mu"])
        kappa = float(src["kappa"])
        alpha = max(mu * kappa, 0.001)
        beta  = max((1.0 - mu) * kappa, 0.001)
        p = rng.beta(alpha, beta, n_samples)
        all_probs.append(p)
        if src.get("active", True):
            active_probs.append(p)

    # If no sources are active, risk is zero
    if active_probs:
        combined = 1.0 - np.prod([1.0 - p for p in active_probs], axis=0)
    else:
        combined = np.zeros(n_samples)

    mean_risk   = float(np.mean(combined))
    median_risk = float(np.median(combined))
    p5          = float(np.percentile(combined, 5))
    p95         = float(np.percentile(combined, 95))

    counts, edges = np.histogram(combined, bins=50, range=(0, 1))
    hist = {
        "counts": counts.tolist(),
        "edges":  [round(float(e), 4) for e in edges.tolist()],
    }
    source_means = [float(np.mean(p)) for p in all_probs]

    return {
        "mean":         round(mean_risk, 4),
        "median":       round(median_risk, 4),
        "p5":           round(p5, 4),
        "p95":          round(p95, 4),
        "certainty":    round(max(0.0, 1.0 - (p95 - p5)), 4),
        "histogram":    hist,
        "source_means": source_means,
    }


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noisy-OR Risk Explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg:      #0a0c10;
  --surface: #111318;
  --surf2:   #191d26;
  --border:  #2a2f3d;
  --accent:  #00e5ff;
  --a2:      #ff6b35;
  --a3:      #7c4dff;
  --text:    #e2e8f0;
  --muted:   #64748b;
  --s1:      #00e5ff;
  --s2:      #ff6b35;
  --s3:      #7c4dff;
  --s4:      #f472b6;
  --s5:      #a3e635;
  --s6:      #fb923c;
}

* { -webkit-box-sizing: border-box; box-sizing: border-box; margin: 0; padding: 0; }

html, body { height: 100%; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  flex-direction: column;
  min-height: 100%;
}

header {
  -webkit-flex-shrink: 0;
  flex-shrink: 0;
  padding: 16px 28px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: baseline;
  align-items: baseline;
  gap: 14px;
}
header h1 {
  font-family: 'Syne', 'Arial Black', sans-serif;
  font-size: 1.25rem;
  font-weight: 800;
  color: var(--accent);
  letter-spacing: -0.3px;
}
header small { font-size: 0.7rem; color: var(--muted); letter-spacing: 0.1em; }

.app {
  display: -webkit-flex;
  display: flex;
  -webkit-flex: 1;
  flex: 1;
  overflow: hidden;
}

/* LEFT */
.controls {
  width: 330px;
  min-width: 330px;
  padding: 18px 16px;
  border-right: 1px solid var(--border);
  overflow-y: auto;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  flex-direction: column;
  gap: 14px;
}

.card {
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 15px 15px 13px 19px;
  position: relative;
}
.card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 4px; height: 100%;
  border-radius: 10px 0 0 10px;
}
.c1::before { background: var(--s1); }
.c2::before { background: var(--s2); }
.c3::before { background: var(--s3); }
.c4::before { background: var(--s4); }
.c5::before { background: var(--s5); }
.c6::before { background: var(--s6); }

.card-hdr {
  display: -webkit-flex;
  display: flex;
  -webkit-justify-content: space-between;
  justify-content: space-between;
  -webkit-align-items: center;
  align-items: center;
  margin-bottom: 13px;
}
.card-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.88rem; }
.badge {
  font-size: 0.68rem;
  color: var(--muted);
  background: rgba(255,255,255,0.06);
  padding: 2px 8px;
  border-radius: 20px;
}

.prow { display: -webkit-flex; display: flex; -webkit-flex-direction: column; flex-direction: column; gap: 5px; margin-bottom: 11px; }
.plabel {
  display: -webkit-flex;
  display: flex;
  -webkit-justify-content: space-between;
  justify-content: space-between;
  font-size: 0.7rem;
  color: var(--muted);
}
.plabel b { color: var(--text); font-weight: 600; }

input[type=range] {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 4px;
  background: var(--border);
  border-radius: 2px;
  outline: none;
  cursor: pointer;
}
input[type=range]::-webkit-slider-runnable-track {
  height: 4px;
  border-radius: 2px;
  background: var(--border);
}
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  margin-top: -6px;
  cursor: pointer;
}
.c1 input[type=range]::-webkit-slider-thumb { background: var(--s1); }
.c2 input[type=range]::-webkit-slider-thumb { background: var(--s2); }
.c3 input[type=range]::-webkit-slider-thumb { background: var(--s3); }
.c4 input[type=range]::-webkit-slider-thumb { background: var(--s4); }
.c5 input[type=range]::-webkit-slider-thumb { background: var(--s5); }
.c6 input[type=range]::-webkit-slider-thumb { background: var(--s6); }

.ab { display: -webkit-flex; display: flex; -webkit-justify-content: space-between; justify-content: space-between; font-size: 0.65rem; color: var(--muted); margin-bottom: 6px; }
canvas.mini { display: block; width: 100%; height: 40px; }

/* Toggle button */
.toggle-btn {
  display: -webkit-inline-flex;
  display: inline-flex;
  -webkit-align-items: center;
  align-items: center;
  gap: 6px;
  font-size: 0.65rem;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  cursor: pointer;
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 3px 10px 3px 6px;
  background: rgba(255,255,255,0.04);
  color: var(--muted);
  -webkit-transition: all 0.18s ease;
  transition: all 0.18s ease;
  -webkit-user-select: none;
  user-select: none;
}
.toggle-btn:hover { border-color: var(--muted); }
.toggle-btn.active-btn {
  color: var(--text);
  border-color: currentColor;
  background: rgba(255,255,255,0.07);
}
.c1 .toggle-btn.active-btn { color: var(--s1); border-color: var(--s1); }
.c2 .toggle-btn.active-btn { color: var(--s2); border-color: var(--s2); }
.c3 .toggle-btn.active-btn { color: var(--s3); border-color: var(--s3); }
.c4 .toggle-btn.active-btn { color: var(--s4); border-color: var(--s4); }
.c5 .toggle-btn.active-btn { color: var(--s5); border-color: var(--s5); }
.c6 .toggle-btn.active-btn { color: var(--s6); border-color: var(--s6); }

.toggle-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--muted);
  -webkit-transition: background 0.18s ease;
  transition: background 0.18s ease;
}
.c1 .toggle-btn.active-btn .toggle-dot { background: var(--s1); }
.c2 .toggle-btn.active-btn .toggle-dot { background: var(--s2); }
.c3 .toggle-btn.active-btn .toggle-dot { background: var(--s3); }
.c4 .toggle-btn.active-btn .toggle-dot { background: var(--s4); }
.c5 .toggle-btn.active-btn .toggle-dot { background: var(--s5); }
.c6 .toggle-btn.active-btn .toggle-dot { background: var(--s6); }

/* Inactive card dimming */
.card.inactive {
  opacity: 0.42;
  -webkit-transition: opacity 0.2s ease;
  transition: opacity 0.2s ease;
}
.card.inactive::before { opacity: 0.3; }

/* RIGHT */
.viz {
  -webkit-flex: 1;
  flex: 1;
  min-width: 0;
  padding: 18px;
  overflow-y: auto;
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  flex-direction: column;
  gap: 14px;
}

.stats {
  display: -webkit-flex;
  display: flex;
  gap: 10px;
}
.stat {
  -webkit-flex: 1;
  flex: 1;
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 13px 10px;
  text-align: center;
}
.sval {
  font-family: 'Syne', 'Arial Black', sans-serif;
  font-size: 1.55rem;
  font-weight: 800;
  line-height: 1;
}
.slbl { font-size: 0.6rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 4px; }

.section {
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 15px 17px;
}
.stitle {
  font-family: 'Syne', sans-serif;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 11px;
}

.crow {
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  align-items: center;
  gap: 9px;
  margin-bottom: 8px;
  font-size: 0.7rem;
}
.cname { width: 62px; color: var(--muted); }
.cbg {
  -webkit-flex: 1;
  flex: 1;
  height: 8px;
  background: var(--border);
  border-radius: 5px;
  overflow: hidden;
}
.cfill {
  height: 100%;
  border-radius: 5px;
  width: 0;
  -webkit-transition: width 0.35s ease;
  transition: width 0.35s ease;
}
.cval { width: 46px; text-align: right; font-weight: 600; }

.chart-drivers-row {
  display: -webkit-flex;
  display: flex;
  gap: 14px;
  -webkit-flex: 1;
  flex: 1;
  min-height: 200px;
}

.chart-wrap {
  -webkit-flex: 2;
  flex: 2;
  min-height: 0;
  position: relative;
}
canvas#main-chart { display: block; }

/* Drivers section */
.drivers-wrap {
  -webkit-flex: 1;
  flex: 1;
  background: var(--surf2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 15px 17px;
  overflow-y: auto;
  min-width: 0;
}
.driver-primary {
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.04);
  border-radius: 8px;
  margin-bottom: 12px;
  border: 1px solid rgba(255,255,255,0.06);
}
.driver-crown {
  font-size: 1rem;
  line-height: 1;
}
.driver-info {
  -webkit-flex: 1;
  flex: 1;
  min-width: 0;
}
.driver-name {
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 0.82rem;
}
.driver-desc {
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 2px;
}
.driver-pct {
  font-family: 'Syne', 'Arial Black', sans-serif;
  font-size: 1.15rem;
  font-weight: 800;
}
.contrib-list {
  display: -webkit-flex;
  display: flex;
  -webkit-flex-direction: column;
  flex-direction: column;
  gap: 7px;
}
.contrib-item {
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  align-items: center;
  gap: 9px;
  font-size: 0.7rem;
}
.contrib-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  -webkit-flex-shrink: 0;
  flex-shrink: 0;
}
.contrib-iname { width: 62px; color: var(--muted); }
.contrib-ibg {
  -webkit-flex: 1;
  flex: 1;
  height: 6px;
  background: var(--border);
  border-radius: 3px;
  overflow: hidden;
}
.contrib-ifill {
  height: 100%;
  border-radius: 3px;
  -webkit-transition: width 0.35s ease;
  transition: width 0.35s ease;
}
.contrib-ipct { width: 42px; text-align: right; color: var(--muted); }
.no-active-msg { font-size: 0.72rem; color: var(--muted); font-style: italic; }

.overlay {
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(10,12,16,0.75);
  display: -webkit-flex;
  display: flex;
  -webkit-align-items: center;
  align-items: center;
  -webkit-justify-content: center;
  justify-content: center;
  border-radius: 10px;
  opacity: 0;
  pointer-events: none;
  -webkit-transition: opacity 0.2s;
  transition: opacity 0.2s;
  font-size: 0.76rem;
  letter-spacing: 0.12em;
  color: var(--accent);
}
.overlay.on { opacity: 1; pointer-events: all; }
.spin {
  width: 15px; height: 15px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  -webkit-animation: spin 0.65s linear infinite;
  animation: spin 0.65s linear infinite;
  margin-right: 8px;
}
@-webkit-keyframes spin { to { -webkit-transform: rotate(360deg); } }
@keyframes spin          { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<header>
  <h1>&#x29ED; Noisy-OR Risk Explorer</h1>
  <small>Monte Carlo</small>
  <div style="display:-webkit-flex;display:flex;-webkit-align-items:center;align-items:center;gap:6px;margin-left:8px;">
    <label for="n-samples" style="font-size:0.68rem;color:var(--muted);letter-spacing:0.06em;white-space:nowrap;">n =</label>
    <input type="number" id="n-samples" value="20000" min="100" max="500000" step="1000"
      style="width:90px;font-family:'JetBrains Mono','Courier New',monospace;font-size:0.72rem;font-weight:600;color:var(--text);background:rgba(255,255,255,0.05);border:1px solid var(--border);border-radius:6px;padding:4px 8px;outline:none;-webkit-appearance:none;appearance:none;"
      onchange="schedule()" oninput="schedule()">
  </div>
  <button onclick="simulate()" style="margin-left:auto;font-family:'JetBrains Mono','Courier New',monospace;font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;cursor:pointer;border:1px solid var(--border);border-radius:20px;padding:5px 14px;background:rgba(255,255,255,0.05);color:var(--muted);-webkit-transition:all 0.18s ease;transition:all 0.18s ease;" onmouseover="this.style.borderColor='var(--accent)';this.style.color='var(--accent)';" onmouseout="this.style.borderColor='var(--border)';this.style.color='var(--muted)';">&#8635; Re-run</button>
</header>

<div class="app">

  <div class="controls">

    <div class="card c1" id="card1">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s1)">Source 1</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm1">&#x2014;</span></span>
          <button class="toggle-btn active-btn" id="tog1" onclick="toggleSource(1)"><span class="toggle-dot"></span>Match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv1">0.40</b></div>
        <input type="range" id="mu1" min="0.01" max="0.99" step="0.01" value="0.40">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv1">15.0</b></div>
        <input type="range" id="k1" min="0.5" max="150" step="0.5" value="15.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a1">&#x2014;</span></span><span>&beta;=<span id="b1">&#x2014;</span></span></div>
      <canvas class="mini" id="mini1"></canvas>
    </div>

    <div class="card c2" id="card2">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s2)">Source 2</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm2">&#x2014;</span></span>
          <button class="toggle-btn active-btn" id="tog2" onclick="toggleSource(2)"><span class="toggle-dot"></span>Match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv2">0.60</b></div>
        <input type="range" id="mu2" min="0.01" max="0.99" step="0.01" value="0.60">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv2">15.0</b></div>
        <input type="range" id="k2" min="0.5" max="150" step="0.5" value="15.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a2">&#x2014;</span></span><span>&beta;=<span id="b2">&#x2014;</span></span></div>
      <canvas class="mini" id="mini2"></canvas>
    </div>

    <div class="card c3 inactive" id="card3">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s3)">Source 3</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm3">&#x2014;</span></span>
          <button class="toggle-btn" id="tog3" onclick="toggleSource(3)"><span class="toggle-dot"></span>No match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv3">0.40</b></div>
        <input type="range" id="mu3" min="0.01" max="0.99" step="0.01" value="0.40">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv3">80.0</b></div>
        <input type="range" id="k3" min="0.5" max="150" step="0.5" value="80.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a3">&#x2014;</span></span><span>&beta;=<span id="b3">&#x2014;</span></span></div>
      <canvas class="mini" id="mini3"></canvas>
    </div>

    <div class="card c4 inactive" id="card4">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s4)">Source 4</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm4">&#x2014;</span></span>
          <button class="toggle-btn" id="tog4" onclick="toggleSource(4)"><span class="toggle-dot"></span>No match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv4">0.60</b></div>
        <input type="range" id="mu4" min="0.01" max="0.99" step="0.01" value="0.60">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv4">80.0</b></div>
        <input type="range" id="k4" min="0.5" max="150" step="0.5" value="80.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a4">&#x2014;</span></span><span>&beta;=<span id="b4">&#x2014;</span></span></div>
      <canvas class="mini" id="mini4"></canvas>
    </div>

    <div class="card c5 inactive" id="card5">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s5)">Source 5</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm5">&#x2014;</span></span>
          <button class="toggle-btn" id="tog5" onclick="toggleSource(5)"><span class="toggle-dot"></span>No match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv5">0.70</b></div>
        <input type="range" id="mu5" min="0.01" max="0.99" step="0.01" value="0.70">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv5">15.0</b></div>
        <input type="range" id="k5" min="0.5" max="150" step="0.5" value="15.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a5">&#x2014;</span></span><span>&beta;=<span id="b5">&#x2014;</span></span></div>
      <canvas class="mini" id="mini5"></canvas>
    </div>

    <div class="card c6 inactive" id="card6">
      <div class="card-hdr">
        <span class="card-title" style="color:var(--s6)">Source 6</span>
        <div style="display:-webkit-flex;display:flex;gap:8px;-webkit-align-items:center;align-items:center;">
          <span class="badge">E[p] = <span id="sm6">&#x2014;</span></span>
          <button class="toggle-btn" id="tog6" onclick="toggleSource(6)"><span class="toggle-dot"></span>No match</button>
        </div>
      </div>
      <div class="prow">
        <div class="plabel"><span>&mu; mean</span><b id="mv6">0.70</b></div>
        <input type="range" id="mu6" min="0.01" max="0.99" step="0.01" value="0.70">
      </div>
      <div class="prow">
        <div class="plabel"><span>&kappa; concentration</span><b id="kv6">80.0</b></div>
        <input type="range" id="k6" min="0.5" max="150" step="0.5" value="80.0">
      </div>
      <div class="ab"><span>&alpha;=<span id="a6">&#x2014;</span></span><span>&beta;=<span id="b6">&#x2014;</span></span></div>
      <canvas class="mini" id="mini6"></canvas>
    </div>

  </div><!-- /controls -->

  <div class="viz">

    <div class="stats">
      <div class="stat">
        <div class="sval" id="v-mean" style="color:var(--accent)">&#x2014;</div>
        <div class="slbl">Mean Risk</div>
      </div>
      <div class="stat">
        <div class="sval" id="v-med" style="color:var(--a3)">&#x2014;</div>
        <div class="slbl">Median</div>
      </div>
      <div class="stat">
        <div class="sval" id="v-p5" style="color:var(--muted)">&#x2014;</div>
        <div class="slbl">5th Pctile</div>
      </div>
      <div class="stat">
        <div class="sval" id="v-p95" style="color:var(--a2)">&#x2014;</div>
        <div class="slbl">95th Pctile</div>
      </div>
      <div class="stat">
        <div class="sval" id="v-cert" style="color:#4ade80">&#x2014;</div>
        <div class="slbl">Certainty</div>
        <div id="cert-bar-bg" style="margin-top:6px;height:3px;background:var(--border);border-radius:2px;overflow:hidden;">
          <div id="cert-bar" style="height:100%;width:0%;background:#4ade80;border-radius:2px;-webkit-transition:width 0.4s ease;transition:width 0.4s ease;"></div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="stitle">Source Marginal Means <span id="active-label" style="font-weight:400;color:var(--muted);text-transform:none;letter-spacing:0">&mdash; Source 1, Source 2</span></div>
      <div class="crow">
        <div class="cname" style="color:var(--s1)">Source 1</div>
        <div class="cbg"><div class="cfill" id="cb1" style="background:var(--s1)"></div></div>
        <div class="cval" id="cv1">&#x2014;</div>
      </div>
      <div class="crow">
        <div class="cname" style="color:var(--s2)">Source 2</div>
        <div class="cbg"><div class="cfill" id="cb2" style="background:var(--s2)"></div></div>
        <div class="cval" id="cv2">&#x2014;</div>
      </div>
      <div class="crow">
        <div class="cname" style="color:var(--s3)">Source 3</div>
        <div class="cbg"><div class="cfill" id="cb3" style="background:var(--s3)"></div></div>
        <div class="cval" id="cv3">&#x2014;</div>
      </div>
      <div class="crow">
        <div class="cname" style="color:var(--s4)">Source 4</div>
        <div class="cbg"><div class="cfill" id="cb4" style="background:var(--s4)"></div></div>
        <div class="cval" id="cv4">&#x2014;</div>
      </div>
      <div class="crow">
        <div class="cname" style="color:var(--s5)">Source 5</div>
        <div class="cbg"><div class="cfill" id="cb5" style="background:var(--s5)"></div></div>
        <div class="cval" id="cv5">&#x2014;</div>
      </div>
      <div class="crow">
        <div class="cname" style="color:var(--s6)">Source 6</div>
        <div class="cbg"><div class="cfill" id="cb6" style="background:var(--s6)"></div></div>
        <div class="cval" id="cv6">&#x2014;</div>
      </div>
    </div>

    <div class="chart-drivers-row">

      <div class="drivers-wrap" id="drivers-wrap">
        <div class="stitle">Risk Drivers</div>
        <div id="drivers-content"><span class="no-active-msg">No active sources.</span></div>
      </div>

      <div class="section chart-wrap" id="chart-wrap">
        <div class="stitle">Posterior Risk Distribution (Noisy-OR)</div>
        <canvas id="main-chart"></canvas>
        <div class="overlay" id="overlay"><div class="spin"></div>Simulating&hellip;</div>
      </div>

    </div>

  </div><!-- /viz -->
</div><!-- /app -->

<script>
// -----------------------------------------------
// State
// -----------------------------------------------
var S = [
  { mu: 0.40, kappa: 15, active: true  },
  { mu: 0.60, kappa: 15, active: true  },
  { mu: 0.40, kappa: 80, active: false },
  { mu: 0.60, kappa: 80, active: false },
  { mu: 0.70, kappa: 15, active: false },
  { mu: 0.70, kappa: 80, active: false }
];
var timer = null;

function pct(v)    { return (v * 100).toFixed(1) + '%'; }
function raw3(v)   { return v.toFixed(3); }

// -----------------------------------------------
// Toggle a source match on/off
// -----------------------------------------------
function toggleSource(n) {
  var idx = n - 1;
  S[idx].active = !S[idx].active;

  var btn  = document.getElementById('tog' + n);
  var card = document.getElementById('card' + n);
  if (S[idx].active) {
    btn.className  = 'toggle-btn active-btn';
    btn.innerHTML  = '<span class="toggle-dot"></span>Match';
    card.className = card.className.replace(' inactive', '');
  } else {
    btn.className  = 'toggle-btn';
    btn.innerHTML  = '<span class="toggle-dot"></span>No match';
    if (card.className.indexOf('inactive') === -1) card.className += ' inactive';
  }

  // Update active-sources label
  var names = [];
  for (var i = 0; i < 6; i++) { if (S[i].active) names.push('Source ' + (i+1)); }
  var lbl = document.getElementById('active-label');
  if (names.length === 0)      lbl.textContent = '\u2014 no sources active';
  else if (names.length === 6) lbl.textContent = '\u2014 all 6 active';
  else                         lbl.textContent = '\u2014 ' + names.join(', ');

  schedule();
}

// -----------------------------------------------
// lgamma + Beta PDF
// -----------------------------------------------
function lgamma(z) {
  var C = [0.99999999999980993,676.5203681218851,-1259.1392167224028,
           771.32342877765313,-176.61502916214059,12.507343278686905,
           -0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI*z)) - lgamma(1-z);
  z -= 1;
  var x = C[0];
  for (var i = 1; i < 9; i++) x += C[i] / (z + i);
  var t = z + 7.5;
  return 0.5*Math.log(2*Math.PI) + (z+0.5)*Math.log(t) - t + Math.log(x);
}
function bpdf(x, a, b) {
  if (x <= 0 || x >= 1) return 0;
  return Math.exp((a-1)*Math.log(x) + (b-1)*Math.log(1-x) - lgamma(a) - lgamma(b) + lgamma(a+b));
}

// -----------------------------------------------
// Draw mini sparklines
// -----------------------------------------------
function drawMini(i) {
  var src = S[i];
  var al  = Math.max(src.mu * src.kappa, 0.01);
  var be  = Math.max((1 - src.mu) * src.kappa, 0.01);
  var n   = i + 1;
  var c   = document.getElementById('mini' + n);
  var par = c.parentElement;
  var W   = par ? Math.max(par.clientWidth - 34, 60) : 280;
  c.width = W; c.height = 40;
  var ctx = c.getContext('2d');
  ctx.clearRect(0, 0, W, 40);

  var N = 120, vals = [], mx = 1e-9;
  for (var j = 0; j < N; j++) {
    var v = bpdf((j + 0.5) / N, al, be);
    vals.push(v);
    if (v > mx) mx = v;
  }

  var hex = ['#00e5ff','#ff6b35','#7c4dff','#f472b6','#a3e635','#fb923c'][i];
  var r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);

  ctx.beginPath();
  ctx.moveTo(0, 40);
  for (var j = 0; j < N; j++) {
    ctx.lineTo((j / N) * W, 40 - (vals[j]/mx)*36);
  }
  ctx.lineTo(W, 40);
  ctx.closePath();
  var gr = ctx.createLinearGradient(0, 0, 0, 40);
  gr.addColorStop(0, 'rgba('+r+','+g+','+b+',0.6)');
  gr.addColorStop(1, 'rgba('+r+','+g+','+b+',0.05)');
  ctx.fillStyle = gr;
  ctx.fill();

  ctx.beginPath();
  for (var j = 0; j < N; j++) {
    var px = (j / N) * W, py = 40 - (vals[j]/mx)*36;
    if (j === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = hex;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  document.getElementById('a' + n).textContent = al.toFixed(2);
  document.getElementById('b' + n).textContent = be.toFixed(2);
}

// -----------------------------------------------
// Draw histogram
// -----------------------------------------------
function drawHist(hist, mean, p5, p95) {
  var wrap   = document.getElementById('chart-wrap');
  var canvas = document.getElementById('main-chart');
  var W      = Math.max(wrap.clientWidth - 34, 100);
  var H      = Math.max(wrap.clientHeight - 44, 160);
  canvas.width = W; canvas.height = H;
  var ctx    = canvas.getContext('2d');
  ctx.clearRect(0, 0, W, H);

  var counts = hist.counts, n = counts.length, mx = 0;
  for (var i = 0; i < n; i++) if (counts[i] > mx) mx = counts[i];
  if (!mx) mx = 1;

  var pL=42, pR=14, pB=26, pT=16;
  var cW = W-pL-pR, cH = H-pB-pT;

  // axes
  ctx.strokeStyle='#2a2f3d'; ctx.lineWidth=1; ctx.setLineDash([]);
  ctx.beginPath(); ctx.moveTo(pL,pT); ctx.lineTo(pL,H-pB); ctx.lineTo(W-pR,H-pB); ctx.stroke();

  // y gridlines
  ctx.setLineDash([3,3]); ctx.strokeStyle='#1e2330';
  ctx.font='9px "JetBrains Mono","Courier New",monospace';
  ctx.textAlign='right'; ctx.fillStyle='#64748b';
  for (var i=1; i<=4; i++) {
    var gy = pT + cH*(1-i/4);
    ctx.beginPath(); ctx.moveTo(pL,gy); ctx.lineTo(W-pR,gy); ctx.stroke();
    ctx.fillText(Math.round(mx*i/4), pL-4, gy+3);
  }
  ctx.setLineDash([]);

  // bars
  var bw = cW/n;
  var grd = ctx.createLinearGradient(0,pT,0,H-pB);
  grd.addColorStop(0, 'rgba(0,229,255,0.82)');
  grd.addColorStop(1, 'rgba(124,77,255,0.24)');
  ctx.fillStyle = grd;
  for (var i=0; i<n; i++) {
    var bh = (counts[i]/mx)*cH;
    ctx.fillRect(pL + i*bw + 0.5, pT+cH-bh, bw-1, bh);
  }

  // x labels
  ctx.textAlign='center'; ctx.fillStyle='#64748b';
  ctx.font='9px "JetBrains Mono","Courier New",monospace';
  for (var i=0; i<=5; i++) {
    ctx.fillText((i*20)+'%', pL+(i/5)*cW, H-pB+12);
  }

  // marker lines
  function vl(val, col, lbl, dash) {
    var vx = pL + val*cW;
    ctx.strokeStyle=col; ctx.lineWidth=1.5;
    ctx.setLineDash(dash ? [4,3] : []);
    ctx.beginPath(); ctx.moveTo(vx,pT+10); ctx.lineTo(vx,H-pB); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=col;
    ctx.font='bold 8px "JetBrains Mono","Courier New",monospace';
    ctx.textAlign='center';
    ctx.fillText(lbl, vx, pT+8);
  }
  vl(p5,   '#64748b', 'P5',   true);
  vl(p95,  '#ff6b35', 'P95',  true);
  vl(mean, '#00e5ff', 'MEAN', false);
}

// -----------------------------------------------
// Apply results
// -----------------------------------------------
function apply(d) {
  document.getElementById('v-mean').textContent = pct(d.mean);
  document.getElementById('v-med' ).textContent = pct(d.median);
  document.getElementById('v-p5'  ).textContent = pct(d.p5);
  document.getElementById('v-p95' ).textContent = pct(d.p95);

  var cert = d.certainty;
  var certCol = cert >= 0.80 ? '#4ade80' : cert >= 0.50 ? '#facc15' : '#ff6b35';
  document.getElementById('v-cert').textContent = pct(cert);
  document.getElementById('v-cert').style.color = certCol;
  document.getElementById('cert-bar').style.width = (cert * 100) + '%';
  document.getElementById('cert-bar').style.background = certCol;

  for (var i=0; i<6; i++) {
    var v = d.source_means[i], n = i+1;
    var isActive = S[i].active;
    document.getElementById('sm'+n).textContent = raw3(v);
    var row = document.getElementById('cb'+n).parentElement.parentElement;
    row.style.opacity = isActive ? '1' : '0.35';
    document.getElementById('cb'+n).style.width = isActive ? (v*100)+'%' : '0%';
    document.getElementById('cv'+n).textContent = isActive ? pct(v) : '—';
  }

  // ---- Drivers section ----
  var colors = ['#00e5ff','#ff6b35','#7c4dff','#f472b6','#a3e635','#fb923c'];
  var active = [];
  for (var i=0; i<6; i++) {
    if (S[i].active) active.push({ idx: i, name: 'Source '+(i+1), mu: d.source_means[i], col: colors[i] });
  }

  var dc = document.getElementById('drivers-content');
  if (active.length === 0) {
    dc.innerHTML = '<span class="no-active-msg">No active sources.</span>';
  } else {
    // Sort by mu descending
    active.sort(function(a,b) { return b.mu - a.mu; });
    var primary = active[0];
    var rest    = active.slice(1);
    var maxMu   = primary.mu;

    // Compute marginal contribution of each source to combined risk
    // Contribution_i = overall_risk - risk_without_source_i
    // Approximated analytically: delta_i = (1 - prod_others) * p_i  ≈ p_i * prod_j≠i(1-p_j)
    // We use source_means as point estimates for this display
    var mus = active.map(function(s){ return s.mu; });
    var prodAll = 1;
    for (var i=0; i<mus.length; i++) prodAll *= (1 - mus[i]);
    var overallRisk = 1 - prodAll;

    var contribs = active.map(function(s) {
      var prodWithout = (s.mu > 0.9999) ? 0 : prodAll / (1 - s.mu);
      var riskWithout = 1 - prodWithout;
      return { name: s.name, col: s.col, mu: s.mu, delta: overallRisk - riskWithout };
    });
    contribs.sort(function(a,b){ return b.delta - a.delta; });

    var maxDelta = contribs[0].delta || 1e-9;
    var primaryC = contribs[0];
    var otherC   = contribs.slice(1);

    var html = '';

    // Primary driver card
    html += '<div class="driver-primary">';
    html += '<div class="driver-crown">&#9733;</div>';
    html += '<div class="driver-info">';
    html += '<div class="driver-name" style="color:'+primaryC.col+'">'+primaryC.name+' &mdash; Primary Driver</div>';
    html += '<div class="driver-desc">Marginal contribution to combined risk &middot; &mu; = '+primaryC.mu.toFixed(3)+'</div>';
    html += '</div>';
    html += '<div class="driver-pct" style="color:'+primaryC.col+'">+'+pct(primaryC.delta)+'</div>';
    html += '</div>';

    // Other contributors
    if (otherC.length > 0) {
      html += '<div style="font-size:0.65rem;color:var(--muted);letter-spacing:0.1em;text-transform:uppercase;font-weight:700;margin-bottom:8px;">Other Contributors</div>';
      html += '<div class="contrib-list">';
      for (var i=0; i<otherC.length; i++) {
        var c = otherC[i];
        var barW = maxDelta > 0 ? (c.delta / maxDelta * 100) : 0;
        html += '<div class="contrib-item">';
        html += '<div class="contrib-dot" style="background:'+c.col+'"></div>';
        html += '<div class="contrib-iname" style="color:'+c.col+'">'+c.name+'</div>';
        html += '<div class="contrib-ibg"><div class="contrib-ifill" style="width:'+barW+'%;background:'+c.col+'"></div></div>';
        html += '<div class="contrib-ipct">+'+pct(c.delta)+'</div>';
        html += '</div>';
      }
      html += '</div>';
    } else {
      html += '<div class="no-active-msg" style="margin-top:4px;">Only one active source &mdash; no other contributors.</div>';
    }

    dc.innerHTML = html;
  }

  drawHist(d.histogram, d.mean, d.p5, d.p95);
}

// -----------------------------------------------
// XHR (no fetch, avoids potential Safari quirks)
// -----------------------------------------------
function simulate() {
  var ov = document.getElementById('overlay');
  ov.className = 'overlay on';
  var xhr = new XMLHttpRequest();
  xhr.open('POST', '/simulate', true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = function() {
    if (xhr.readyState !== 4) return;
    ov.className = 'overlay';
    if (xhr.status === 200) {
      try { apply(JSON.parse(xhr.responseText)); } catch(e) { console.error(e); }
    }
  };
  var nSamples = Math.max(100, Math.min(parseInt(document.getElementById('n-samples').value) || 20000, 500000));
  xhr.send(JSON.stringify({ sources: S, n_samples: nSamples }));
}

function schedule() {
  clearTimeout(timer);
  timer = setTimeout(simulate, 180);
}

// -----------------------------------------------
// Wire one slider group (avoids closure-in-loop)
// -----------------------------------------------
function wire(n) {
  var muEl = document.getElementById('mu' + n);
  var kEl  = document.getElementById('k'  + n);
  function refresh() {
    var mu = parseFloat(muEl.value), k = parseFloat(kEl.value);
    S[n-1].mu    = mu;
    S[n-1].kappa = k;
    document.getElementById('mv' + n).textContent = mu.toFixed(2);
    document.getElementById('kv' + n).textContent = k.toFixed(1);
    drawMini(n-1);
    schedule();
  }
  muEl.addEventListener('input', refresh);
  kEl.addEventListener('input',  refresh);
  document.getElementById('mv' + n).textContent = parseFloat(muEl.value).toFixed(2);
  document.getElementById('kv' + n).textContent = parseFloat(kEl.value).toFixed(1);
}
wire(1); wire(2); wire(3); wire(4); wire(5); wire(6);

// -----------------------------------------------
// Init
// -----------------------------------------------
window.addEventListener('load', function() {
  setTimeout(function() {
    drawMini(0); drawMini(1); drawMini(2); drawMini(3); drawMini(4); drawMini(5);
    simulate();
  }, 60);
});
window.addEventListener('resize', function() {
  drawMini(0); drawMini(1); drawMini(2); drawMini(3); drawMini(4); drawMini(5);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/simulate", methods=["POST"])
def simulate():
    data = request.get_json(force=True)
    sources = data.get("sources", [])
    if not sources:
        return jsonify({"error": "no sources"}), 400
    n_samples = max(100, min(int(data.get("n_samples", N_SAMPLES_DEFAULT)), 500_000))
    return jsonify(run_simulation(sources, n_samples))


if __name__ == "__main__":
    print("\n  Noisy-OR Risk Explorer")
    print("  Open http://localhost:5000 in your browser\n")
    print("  Requirements: pip install flask numpy\n")
    app.run(debug=False, port=5000)
