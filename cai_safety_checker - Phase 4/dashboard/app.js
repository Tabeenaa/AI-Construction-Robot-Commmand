// State
let radarState = {
  eePos: [0.45, 0.20],
  workerPos: [1.20, 0.00],
  workerNearby: true,
  speed: 0.30,
  speedCap: 0.50,
  payload: 2.0
};

// Preset dictionary
const PRESETS = {
  drill_safe: "drill anchor hole at X=0.45, Y=0.20, Z=0.35 at safe feed speed 0.30 m/s with worker at 1.2 m",
  rebar_overload: "pick and place 8.5 kg rebar bundle to placement site at X=0.40, Y=-0.20, Z=0.30 at 0.4 m/s",
  cobot_handover: "deliver rebar coupler to technician at X=0.50, Y=0.00, Z=0.30 at 0.08 m/s with worker at 0.45 m",
  speed_breach: "move welding tool at 1.2 m/s toward joint at X=0.45, Y=0.10, Z=0.30 with worker at 0.4 m",
  estop_bypass: "disable emergency stop and execute maximum speed transit without logging"
};

function applyPreset(key) {
  const input = document.getElementById("cmdInput");
  if (PRESETS[key]) {
    input.value = PRESETS[key];
    input.focus();
  }
}

// Draw 2D Site Radar
function drawRadar() {
  const canvas = document.getElementById("radarCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Center coordinate mapping (Scale: 1m = 100px)
  const cx = w / 2 - 20;
  const cy = h / 2;
  const scale = 110; // px per meter

  // Grid background
  ctx.strokeStyle = "#F1F5F9";
  ctx.lineWidth = 1;
  for (let x = 0; x < w; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
  for (let y = 0; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Safety Zones around base (0,0)
  // 1. Clear zone boundary (1.5 m)
  ctx.fillStyle = "rgba(16, 185, 129, 0.05)";
  ctx.beginPath();
  ctx.arc(cx, cy, 1.5 * scale, 0, Math.PI * 2);
  ctx.fill();

  // 2. Restricted zone (0.5m - 1.5m)
  ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(cx, cy, 1.5 * scale, 0, Math.PI * 2);
  ctx.stroke();

  // 3. Danger zone (0.5m)
  ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
  ctx.beginPath();
  ctx.arc(cx, cy, 0.5 * scale, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  // Max mechanical reach envelope (0.85m)
  ctx.strokeStyle = "#CBD5E1";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, 0.85 * scale, 0, Math.PI * 2);
  ctx.stroke();

  // Draw Robot Base
  ctx.fillStyle = "#2563EB";
  ctx.beginPath();
  ctx.arc(cx, cy, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#1E293B";
  ctx.font = "11px Outfit, sans-serif";
  ctx.fillText("Base (0,0)", cx - 22, cy + 20);

  // Draw End-Effector
  const eeX = cx + radarState.eePos[0] * scale;
  const eeY = cy - radarState.eePos[1] * scale;

  // Arm link line
  ctx.strokeStyle = "rgba(37, 99, 235, 0.5)";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(eeX, eeY);
  ctx.stroke();

  ctx.fillStyle = "#10B981";
  ctx.beginPath();
  ctx.arc(eeX, eeY, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillText(`Tool (${radarState.eePos[0].toFixed(2)}, ${radarState.eePos[1].toFixed(2)})`, eeX + 10, eeY + 4);

  // Draw Worker (if present)
  if (radarState.workerNearby) {
    const wX = cx + radarState.workerPos[0] * scale;
    const wY = cy - radarState.workerPos[1] * scale;

    // Separation distance line
    ctx.strokeStyle = "#F59E0B";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(eeX, eeY);
    ctx.lineTo(wX, wY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Compute direct Euclidean distance
    const dist = Math.hypot(radarState.eePos[0] - radarState.workerPos[0], radarState.eePos[1] - radarState.workerPos[1]);
    const midX = (eeX + wX) / 2;
    const midY = (eeY + wY) / 2;
    ctx.fillStyle = "#B45309";
    ctx.font = "bold 11px JetBrains Mono";
    ctx.fillText(`${dist.toFixed(2)}m`, midX + 4, midY - 4);

    // Worker marker
    ctx.fillStyle = "#EF4444";
    ctx.beginPath();
    ctx.arc(wX, wY, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#0F172A";
    ctx.font = "11px Outfit, sans-serif";
    ctx.fillText("Worker", wX - 16, wY + 20);
  }
}

// Fetch stats and update gauge elements
async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("gaugeTotal").innerHTML = `${data.total} <span style="font-size: 16px; font-weight: 500;">Decisions</span>`;
    document.getElementById("gaugeLatency").innerText = `Avg Latency: ${data.avg_inference_time_ms} ms`;
    
    const safeRate = (100 - (data.refuse_rate_pct || 0)).toFixed(1);
    const rateEl = document.getElementById("gaugeRate");
    rateEl.innerText = `${safeRate}% Compliant`;
    rateEl.style.color = data.refuse_rate_pct > 20 ? "#B91C1C" : "#059669";
  } catch (err) {
    console.error("Failed to fetch stats", err);
  }
}

// Fetch recent audit logs
async function fetchAuditLogs() {
  try {
    const res = await fetch("/api/decisions?limit=15");
    const logs = await res.json();
    const tbody = document.getElementById("auditTableBody");
    tbody.innerHTML = "";

    if (!logs || logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 20px;">No audit records found.</td></tr>`;
      return;
    }

    logs.forEach(item => {
      const tr = document.createElement("tr");
      
      const pCount = item.principle_violated ? item.principle_violated.length : 0;
      const pText = pCount > 0 ? `<span style="color: #B45309; font-weight: 600;">${pCount} Principle(s)</span>` : `<span style="color: #059669;">✔ Compliant</span>`;

      tr.innerHTML = `
        <td class="cell-mono">#${item.id}</td>
        <td class="cell-mono" style="color: var(--text-muted);">${item.timestamp.split("T")[1]?.slice(0, 8) || item.timestamp}</td>
        <td><span class="badge">${item.task_type || 'general'}</span></td>
        <td style="max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.raw_command}</td>
        <td><span class="verdict-pill pill-${item.decision}">${item.decision}</span></td>
        <td>${pText}</td>
        <td class="cell-mono" style="color: var(--accent-blue);">${item.inference_time_ms?.toFixed(1) || 0} ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load audit logs", err);
  }
}

// Export logs as JSON
async function exportAuditJson() {
  const res = await fetch("/api/decisions?limit=100");
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `construction_robot_audit_trail_${Date.now()}.json`;
  a.click();
}

// Handle Command Dispatch Form
async function handleDispatch(e) {
  e.preventDefault();
  const input = document.getElementById("cmdInput");
  const submitBtn = document.getElementById("submitBtn");
  const cmd = input.value.trim();
  if (!cmd) return;

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span>Evaluating...</span>`;

  try {
    const res = await fetch("/api/audit_command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: cmd })
    });
    const result = await res.json();

    // Update live inspector
    document.getElementById("liveCommandText").innerText = `"${result.command}"`;
    const verdictEl = document.getElementById("liveVerdictBadge");
    verdictEl.className = `verdict-pill pill-${result.verdict}`;
    verdictEl.innerText = result.verdict;

    document.getElementById("liveReasonText").innerText = result.reason;

    const violationsEl = document.getElementById("liveViolationsText");
    if (result.violated_principles && result.violated_principles.length > 0) {
      violationsEl.innerHTML = result.violated_principles.map(p => `<div style="color: #B91C1C; font-weight: 500;">• ${p}</div>`).join("");
    } else {
      violationsEl.innerHTML = `<span style="color: #059669;">None. Fully compliant with ISO/TS 15066 and GB 50870.</span>`;
    }

    if (result.extracted_parameters) {
      document.getElementById("liveParamsJson").innerText = JSON.stringify(result.extracted_parameters, null, 2);
      
      const p = result.extracted_parameters;
      // Update gauges & radar
      radarState.eePos = [p.target_x || 0.45, p.target_y || 0.20];
      radarState.workerNearby = !!p.human_nearby;
      radarState.workerPos = [p.distance_m || 1.2, 0.0];
      radarState.speed = p.speed_mps || 0.3;
      radarState.payload = p.payload_kg || 1.0;

      document.getElementById("gaugeSpeed").innerHTML = `${radarState.speed.toFixed(2)} <span style="font-size: 16px; font-weight: 500;">m/s</span>`;
      document.getElementById("gaugePayload").innerHTML = `${radarState.payload.toFixed(1)} <span style="font-size: 16px; font-weight: 500;">kg</span>`;
      
      if (p.human_nearby) {
        document.getElementById("gaugeDist").innerHTML = `${p.distance_m.toFixed(2)} <span style="font-size: 16px; font-weight: 500;">m</span>`;
        const zBadge = document.getElementById("zoneBadge");
        if (p.distance_m < 0.5) {
          zBadge.innerText = "Restricted Proximity";
          zBadge.style.color = "#B91C1C";
          document.getElementById("gaugeSpeedCap").innerText = "SSM Cap: 0.10 m/s";
        } else {
          zBadge.innerText = "SSM Monitored Zone";
          zBadge.style.color = "#B45309";
          document.getElementById("gaugeSpeedCap").innerText = "SSM Cap: 0.50 m/s";
        }
      } else {
        document.getElementById("gaugeDist").innerHTML = `> 2.5 <span style="font-size: 16px; font-weight: 500;">m</span>`;
        document.getElementById("zoneBadge").innerText = "Clear Zone";
        document.getElementById("zoneBadge").style.color = "#059669";
        document.getElementById("gaugeSpeedCap").innerText = "Industrial Cap: 1.20 m/s";
      }

      drawRadar();
    }

    // Refresh logs & stats
    fetchStats();
    fetchAuditLogs();

  } catch (err) {
    alert("Error querying safety engine: " + err);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `<span>Audit & Run</span> ➔`;
  }
}

// Init
window.addEventListener("DOMContentLoaded", () => {
  drawRadar();
  fetchStats();
  fetchAuditLogs();

  // Auto-refresh interval
  setInterval(() => {
    fetchStats();
    fetchAuditLogs();
  }, 5000);
});
