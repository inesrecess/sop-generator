import os
import json
import uuid
import base64
import shutil
import subprocess
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic

app = FastAPI(title="SOP Generator")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/app/data"
SCREENSHOTS_DIR = f"{DATA_DIR}/screenshots"
SOPS_FILE = f"{DATA_DIR}/sops.json"

for d in [DATA_DIR, SCREENSHOTS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_sops() -> list:
    if os.path.exists(SOPS_FILE):
        with open(SOPS_FILE) as f:
            return json.load(f)
    return []


def save_sops(sops: list):
    with open(SOPS_FILE, "w") as f:
        json.dump(sops, f, indent=2)


# ── Models ─────────────────────────────────────────────────────────────────────
class SOPRequest(BaseModel):
    url: str
    title: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/sops")
async def list_sops():
    return load_sops()


@app.delete("/sops/{sop_id}")
async def delete_sop(sop_id: str):
    sops = load_sops()
    sops = [s for s in sops if s["id"] != sop_id]
    save_sops(sops)
    sop_dir = f"{SCREENSHOTS_DIR}/{sop_id}"
    if os.path.exists(sop_dir):
        shutil.rmtree(sop_dir)
    return {"ok": True}


@app.post("/generate-sop")
async def generate_sop(req: SOPRequest):
    sop_id = str(uuid.uuid4())[:8]
    sop_dir = f"{SCREENSHOTS_DIR}/{sop_id}"
    video_path = f"/tmp/vid_{sop_id}.mp4"
    os.makedirs(sop_dir, exist_ok=True)

    try:
        # ── 1. Download video ──────────────────────────────────────────────────
        dl = subprocess.run(
            [
                "yt-dlp", "--no-playlist",
                "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", video_path,
                "--no-warnings",
                req.url,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if not os.path.exists(video_path):
            # fallback: simplest format
            dl = subprocess.run(
                ["yt-dlp", "--no-playlist", "-o", video_path, req.url],
                capture_output=True, text=True, timeout=300,
            )
        if not os.path.exists(video_path):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not download the video. Make sure the Loom link is public "
                    f"and try again.\n\nDetails: {dl.stderr[:400]}"
                ),
            )

        # ── 2. Get video duration ──────────────────────────────────────────────
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
            capture_output=True, text=True,
        )
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            duration = 120.0

        # ── 3. Extract frames ──────────────────────────────────────────────────
        n_frames = min(15, max(6, int(duration / 8)))
        interval = duration / n_frames
        subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps=1/{interval:.2f},scale=1280:-2",
                "-q:v", "3",
                f"{sop_dir}/frame_%03d.jpg",
            ],
            capture_output=True, timeout=180,
        )

        frames = sorted(f for f in os.listdir(sop_dir) if f.endswith(".jpg"))
        if not frames:
            raise HTTPException(status_code=500, detail="Could not extract frames from video.")

        # ── 4. Build Claude message ────────────────────────────────────────────
        content = []
        screenshot_meta = []

        for i, fname in enumerate(frames[:15]):
            ts = i * interval
            time_str = f"{int(ts // 60):02d}:{int(ts % 60):02d}"
            rel_url = f"/screenshots/{sop_id}/{fname}"
            screenshot_meta.append(
                {"index": i + 1, "time": time_str, "url": rel_url, "filename": fname}
            )
            with open(f"{sop_dir}/{fname}", "rb") as fh:
                b64 = base64.standard_b64encode(fh.read()).decode()
            content += [
                {"type": "text", "text": f"Screenshot {i + 1} at {time_str}:"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            ]

        ref_table = json.dumps(screenshot_meta, indent=2)
        content.append(
            {
                "type": "text",
                "text": f"""
You have {len(frames)} screenshots from a Loom walkthrough video.
URL: {req.url}
Title hint: {req.title or '(determine from content)'}

Screenshot reference — use EXACT filenames shown here:
{ref_table}

Write a complete SOP in Markdown. Follow this structure exactly:

# [Clear Descriptive Title]

## Overview
[1-2 sentences on what this SOP covers and the outcome]

## Prerequisites
- [List what the user needs before starting]

## Step-by-Step Instructions

### Step 1: [Action Name]
[Clear description of what to do]
![Step 1](/screenshots/{sop_id}/EXACT_FILENAME_FROM_TABLE)

[Continue for every step, embedding the most relevant screenshot directly after each step]

## Tips & Notes
- [Important observations, warnings, or shortcuts]

Rules:
- Use the EXACT image paths from the reference table above
- Embed a screenshot after every major step using the exact filename
- Write in plain, simple language — assume the reader is brand new to this
- Number every step clearly
""",
            }
        )

        # ── 5. Call Claude ─────────────────────────────────────────────────────
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=(
                "You are an expert SOP writer. You analyse video screenshots and produce "
                "clear, actionable Standard Operating Procedures that anyone can follow. "
                "Always embed the provided screenshot images at the relevant steps using "
                "Markdown image syntax exactly as specified."
            ),
            messages=[{"role": "user", "content": content}],
        ) as stream:
            sop_text = stream.get_final_message().content[0].text

        # ── 6. Extract title ───────────────────────────────────────────────────
        title = req.title
        if not title:
            for line in sop_text.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
        if not title:
            title = f"SOP — {datetime.now().strftime('%b %d, %Y')}"

        # ── 7. Save ────────────────────────────────────────────────────────────
        sop = {
            "id": sop_id,
            "title": title,
            "url": req.url,
            "created_at": datetime.now().isoformat(),
            "content": sop_text,
            "screenshots": screenshot_meta,
        }
        sops = load_sops()
        sops.insert(0, sop)
        save_sops(sops)
        return sop

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(sop_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)


# ── Frontend HTML ──────────────────────────────────────────────────────────────
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SOP Generator</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
  <style>
    :root {
      --bg: #f8f9fa; --surface: #fff; --border: #e5e7eb;
      --primary: #6366f1; --primary-h: #4f46e5;
      --text: #1f2937; --muted: #6b7280;
      --green: #10b981; --red: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg); color: var(--text);
      height: 100vh; display: flex; flex-direction: column; overflow: hidden;
    }
    header {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 14px 24px; display: flex; align-items: center; flex-shrink: 0;
    }
    header h1 { font-size: 18px; font-weight: 700; color: var(--primary); }
    header .sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
    .input-bar {
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 12px 24px; display: flex; gap: 10px; align-items: center; flex-shrink: 0;
    }
    .input-bar input {
      padding: 9px 13px; border: 1px solid var(--border);
      border-radius: 8px; font-size: 14px; outline: none; transition: border-color .15s;
    }
    #urlInput { flex: 1; }
    #titleInput { width: 210px; }
    .input-bar input:focus { border-color: var(--primary); }
    .btn {
      padding: 9px 18px; border-radius: 8px; border: none;
      font-size: 13px; font-weight: 500; cursor: pointer; transition: all .15s; white-space: nowrap;
    }
    .btn-primary { background: var(--primary); color: #fff; }
    .btn-primary:hover:not(:disabled) { background: var(--primary-h); }
    .btn-primary:disabled { opacity: .5; cursor: not-allowed; }
    .btn-sm { padding: 6px 12px; font-size: 12px; }
    .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
    .btn-outline:hover { background: var(--bg); }
    .btn-green { background: var(--green); color: #fff; }
    .btn-green:hover { background: #059669; }
    .btn-red { background: transparent; border: 1px solid #fca5a5; color: var(--red); }
    .btn-red:hover { background: #fef2f2; }
    .main { flex: 1; display: flex; overflow: hidden; }
    .sidebar {
      width: 270px; border-right: 1px solid var(--border);
      background: var(--surface); display: flex; flex-direction: column;
      flex-shrink: 0; overflow: hidden;
    }
    .sidebar-header {
      padding: 12px 16px; font-size: 11px; font-weight: 600;
      color: var(--muted); text-transform: uppercase; letter-spacing: .06em;
      border-bottom: 1px solid var(--border); flex-shrink: 0;
    }
    .sop-list { overflow-y: auto; flex: 1; }
    .sop-item {
      padding: 13px 16px; border-bottom: 1px solid var(--border);
      cursor: pointer; transition: background .1s;
    }
    .sop-item:hover { background: var(--bg); }
    .sop-item.active { background: #eef2ff; border-left: 3px solid var(--primary); padding-left: 13px; }
    .sop-item .st { font-size: 13px; font-weight: 500; line-height: 1.4; }
    .sop-item .sm { font-size: 11px; color: var(--muted); margin-top: 3px; }
    .empty-list { padding: 24px 16px; text-align: center; color: var(--muted); font-size: 13px; }
    .viewer { flex: 1; overflow-y: auto; padding: 32px; }
    .viewer-empty {
      height: 100%; display: flex; flex-direction: column;
      align-items: center; justify-content: center; text-align: center;
      color: var(--muted); gap: 10px;
    }
    .viewer-empty .icon { font-size: 52px; }
    .viewer-empty strong { font-size: 16px; color: var(--text); }
    .viewer-empty p { font-size: 14px; line-height: 1.6; }
    .sop-actions { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }
    .sop-source { font-size: 12px; color: var(--muted); margin-bottom: 24px; }
    .sop-source a { color: var(--primary); text-decoration: none; }
    .sop-source a:hover { text-decoration: underline; }
    .md { max-width: 780px; margin: 0 auto; }
    .md h1 { font-size: 26px; font-weight: 700; margin-bottom: 6px; line-height: 1.25; }
    .md h2 { font-size: 18px; font-weight: 600; margin: 28px 0 10px; color: #374151; }
    .md h3 { font-size: 15px; font-weight: 600; margin: 20px 0 8px; }
    .md p { font-size: 14px; line-height: 1.75; margin-bottom: 12px; color: #4b5563; }
    .md ul, .md ol { padding-left: 22px; margin-bottom: 12px; }
    .md li { font-size: 14px; line-height: 1.75; margin-bottom: 3px; color: #4b5563; }
    .md img {
      max-width: 100%; border-radius: 10px; border: 1px solid var(--border);
      margin: 14px 0; box-shadow: 0 2px 12px rgba(0,0,0,.08); display: block;
    }
    .md hr { border: none; border-top: 1px solid var(--border); margin: 24px 0; }
    .md strong { font-weight: 600; }
    .md code { background: #f3f4f6; padding: 2px 5px; border-radius: 4px; font-size: 12px; }
    .overlay {
      position: fixed; inset: 0; background: rgba(255,255,255,.85);
      backdrop-filter: blur(5px); display: flex; flex-direction: column;
      align-items: center; justify-content: center; z-index: 50; gap: 18px;
    }
    .overlay.hidden { display: none; }
    .spinner {
      width: 48px; height: 48px; border: 4px solid var(--border);
      border-top-color: var(--primary); border-radius: 50%;
      animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .ov-text { font-size: 15px; font-weight: 500; }
    .ov-step { font-size: 13px; color: var(--muted); }
    .toast {
      position: fixed; bottom: 24px; right: 24px; background: #111827;
      color: #fff; padding: 11px 18px; border-radius: 8px; font-size: 13px;
      opacity: 0; transform: translateY(8px); transition: all .3s;
      z-index: 100; pointer-events: none;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
  </style>
</head>
<body>

<header>
  <div>
    <h1>&#127909; SOP Generator</h1>
    <div class="sub">Paste a Loom URL &rarr; get a full SOP with screenshots, ready to paste into Notion</div>
  </div>
</header>

<div class="input-bar">
  <input id="urlInput" type="text" placeholder="https://www.loom.com/share/..." />
  <input id="titleInput" type="text" placeholder="Title (optional)" />
  <button class="btn btn-primary" id="genBtn" onclick="generateSOP()">&#10024; Generate SOP</button>
</div>

<div class="main">
  <div class="sidebar">
    <div class="sidebar-header">My SOPs &nbsp;(<span id="sopCount">0</span>)</div>
    <div class="sop-list" id="sopList"></div>
  </div>
  <div class="viewer" id="viewer">
    <div class="viewer-empty">
      <div class="icon">&#128196;</div>
      <strong>No SOP open</strong>
      <p>Paste a Loom URL above and click <strong>Generate SOP</strong>.<br>Your saved SOPs appear in the sidebar.</p>
    </div>
  </div>
</div>

<div class="overlay hidden" id="overlay">
  <div class="spinner"></div>
  <div class="ov-text">Generating your SOP&hellip;</div>
  <div class="ov-step" id="ovStep">Downloading video &amp; extracting screenshots</div>
</div>

<div class="toast" id="toast"></div>

<script>
  let currentSOP = null, allSOPs = [];

  const STEPS = [
    'Downloading video & extracting screenshots...',
    'Analysing screenshots with Claude AI...',
    'Writing step-by-step instructions...',
    'Finalising your SOP document...',
  ];
  let stepTimer = null;
  function startSteps() {
    let i = 0;
    document.getElementById('ovStep').textContent = STEPS[0];
    stepTimer = setInterval(() => {
      i = (i + 1) % STEPS.length;
      document.getElementById('ovStep').textContent = STEPS[i];
    }, 9000);
  }
  function stopSteps() { clearInterval(stepTimer); }

  async function loadSOPs() {
    try { const r = await fetch('/sops'); allSOPs = await r.json(); renderList(); }
    catch (e) { console.error(e); }
  }

  function renderList() {
    document.getElementById('sopCount').textContent = allSOPs.length;
    const el = document.getElementById('sopList');
    if (!allSOPs.length) {
      el.innerHTML = '<div class="empty-list">No SOPs yet.<br>Generate your first one!</div>';
      return;
    }
    el.innerHTML = allSOPs.map(s => `
      <div class="sop-item ${currentSOP && currentSOP.id === s.id ? 'active' : ''}" onclick="viewSOP('${s.id}')">
        <div class="st">${esc(s.title)}</div>
        <div class="sm">${fmtDate(s.created_at)} &middot; ${(s.screenshots||[]).length} screenshots</div>
      </div>`).join('');
  }

  function viewSOP(id) {
    currentSOP = allSOPs.find(s => s.id === id);
    if (!currentSOP) return;
    renderList();
    const origin = window.location.origin;
    const absContent = currentSOP.content.replace(
      /!\\[([^\\]]*)\\]\\(\\/screenshots\\//g,
      `![$1](${origin}/screenshots/`
    );
    document.getElementById('viewer').innerHTML = `
      <div class="md">
        <div class="sop-actions">
          <button class="btn btn-sm btn-green" onclick="copyNotion()">&#128203; Copy for Notion</button>
          <button class="btn btn-sm btn-outline" onclick="copyRaw()">&#128221; Copy Markdown</button>
          <button class="btn btn-sm btn-red" onclick="deleteSOP('${currentSOP.id}')">&#128465; Delete</button>
        </div>
        <div class="sop-source">
          Generated ${fmtDate(currentSOP.created_at)} from
          <a href="${esc(currentSOP.url)}" target="_blank" rel="noreferrer">${esc(currentSOP.url)}</a>
        </div>
        ${marked.parse(absContent)}
      </div>`;
  }

  function copyNotion() {
    if (!currentSOP) return;
    const origin = window.location.origin;
    const text = currentSOP.content.replace(
      /!\\[([^\\]]*)\\]\\(\\/screenshots\\//g,
      `![$1](${origin}/screenshots/`
    );
    copy(text, 'Copied! Paste into Notion ✓');
  }

  function copyRaw() {
    if (!currentSOP) return;
    copy(currentSOP.content, 'Raw Markdown copied ✓');
  }

  function copy(text, msg) {
    navigator.clipboard.writeText(text)
      .then(() => toast(msg))
      .catch(() => {
        const t = document.createElement('textarea');
        t.value = text; document.body.appendChild(t); t.select();
        document.execCommand('copy'); document.body.removeChild(t);
        toast(msg);
      });
  }

  async function deleteSOP(id) {
    if (!confirm('Delete this SOP and its screenshots?')) return;
    await fetch(`/sops/${id}`, { method: 'DELETE' });
    allSOPs = allSOPs.filter(s => s.id !== id);
    currentSOP = null; renderList();
    document.getElementById('viewer').innerHTML = `
      <div class="viewer-empty">
        <div class="icon">&#128465;</div>
        <strong>SOP deleted</strong>
        <p>Select another SOP or generate a new one.</p>
      </div>`;
  }

  async function generateSOP() {
    const url = document.getElementById('urlInput').value.trim();
    const title = document.getElementById('titleInput').value.trim();
    if (!url) { toast('Please enter a Loom URL', true); return; }
    document.getElementById('genBtn').disabled = true;
    document.getElementById('overlay').classList.remove('hidden');
    startSteps();
    try {
      const r = await fetch('/generate-sop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, title }),
      });
      if (!r.ok) { const e = await r.json(); throw new Error(e.detail || 'Failed'); }
      const sop = await r.json();
      allSOPs.unshift(sop); currentSOP = sop;
      renderList(); viewSOP(sop.id);
      document.getElementById('urlInput').value = '';
      document.getElementById('titleInput').value = '';
      toast('SOP generated! ✓');
    } catch (e) {
      toast('Error: ' + e.message, true);
    } finally {
      stopSteps();
      document.getElementById('overlay').classList.add('hidden');
      document.getElementById('genBtn').disabled = false;
    }
  }

  function toast(msg, isErr = false) {
    const el = document.getElementById('toast');
    el.textContent = msg; el.style.background = isErr ? '#ef4444' : '#111827';
    el.classList.add('show'); setTimeout(() => el.classList.remove('show'), 3500);
  }

  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  function fmtDate(iso) {
    return new Date(iso).toLocaleDateString('en-US', {month:'short',day:'numeric',year:'numeric'});
  }

  document.getElementById('urlInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') generateSOP();
  });

  loadSOPs();
</script>
</body>
</html>
"""
