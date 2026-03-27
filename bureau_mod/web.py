"""Web control server with SSE real-time updates and SPA dashboard."""

from __future__ import annotations

import asyncio
import json
import logging

from aiohttp import web

from bureau_mod.git_utils import git_log_oneline, git_ls_tree, git_show_diff, read_repo_file
from bureau_mod.state import (
    ACTIVE_CLIENTS,
    PAUSE_EVENT,
    SSE_CLIENTS,
    STATE,
    TaskStatus,
    emit_stats,
)

log = logging.getLogger("bureau")


# ═══════════════════════════════════════════════════════════════════════════
# API Handlers
# ═══════════════════════════════════════════════════════════════════════════

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(text=DASHBOARD_HTML, content_type="text/html")


async def handle_api_state(request: web.Request) -> web.Response:
    return web.json_response(STATE.to_dict())


async def handle_api_skip(request: web.Request) -> web.Response:
    """Skip a task. If running, attempt to interrupt the agent."""
    try:
        data = await request.json()
        task_id = data.get("task_id")
        if not task_id:
            return web.json_response({"error": "task_id required"}, status=400)

        task = STATE.get_task(task_id)
        if not task:
            return web.json_response({"error": "task not found"}, status=404)

        if task.status == TaskStatus.RUNNING:
            # Try to interrupt the active SDK client
            client = ACTIVE_CLIENTS.get(task_id)
            if client:
                try:
                    await client.interrupt()
                except Exception:
                    pass

        STATE.update_task_status(task_id, TaskStatus.SKIPPED)
        log.info(f"Task {task_id} skipped via API")
        return web.json_response({"status": "ok", "task_id": task_id})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_stop_revising(request: web.Request) -> web.Response:
    """Tell a task to stop revising and accept current version."""
    try:
        data = await request.json()
        task_id = data.get("task_id")
        if not task_id:
            return web.json_response({"error": "task_id required"}, status=400)

        task = STATE.get_task(task_id)
        if not task:
            return web.json_response({"error": "task not found"}, status=404)

        task.stop_revising = True
        log.info(f"Task {task_id}: stop_revising set via API")
        return web.json_response({"status": "ok", "task_id": task_id})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_pause(request: web.Request) -> web.Response:
    STATE.paused = True
    PAUSE_EVENT.clear()
    log.info("Execution paused via API")
    emit_stats()
    return web.json_response({"status": "paused"})


async def handle_api_resume(request: web.Request) -> web.Response:
    STATE.paused = False
    PAUSE_EVENT.set()
    log.info("Execution resumed via API")
    emit_stats()
    return web.json_response({"status": "resumed"})


async def handle_api_reduce_rounds(request: web.Request) -> web.Response:
    try:
        data = await request.json()
        rounds = data.get("rounds", 1)
        count = 0
        for task in STATE.tasks.values():
            if task.status == TaskStatus.PENDING:
                task.max_revision_rounds = rounds
                count += 1
        log.info(f"Set max_revision_rounds={rounds} for {count} pending tasks")
        return web.json_response({"status": "ok", "affected": count})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_checkpoint(request: web.Request) -> web.Response:
    try:
        from pathlib import Path
        config_dir = Path(STATE.config_dir)
        checkpoint_path = config_dir / "checkpoint.json"
        STATE.save_checkpoint(checkpoint_path)
        return web.json_response({"status": "ok", "path": str(checkpoint_path)})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_api_stop(request: web.Request) -> web.Response:
    from bureau_mod.state import request_stop
    log.info("Stopping bureau via API")
    request_stop()
    return web.json_response({"status": "stopping"})


async def handle_api_gitlog(request: web.Request) -> web.Response:
    """Return git log --oneline."""
    entries = git_log_oneline(STATE.work_dir)
    return web.json_response(entries)


async def handle_api_gitdiff(request: web.Request) -> web.Response:
    """Return diff for a specific commit."""
    commit_hash = request.query.get("hash", "")
    if not commit_hash:
        return web.json_response({"error": "hash required"}, status=400)
    diff = git_show_diff(STATE.work_dir, commit_hash)
    return web.Response(text=diff, content_type="text/plain")


async def handle_api_task_output(request: web.Request) -> web.Response:
    """Return full output for a task."""
    task_id = request.query.get("task_id", "")
    task = STATE.get_task(task_id)
    if not task:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"task_id": task_id, "lines": task.output_lines})


async def handle_api_task_prompt(request: web.Request) -> web.Response:
    """Return the full prompt for a task."""
    task_id = request.query.get("task_id", "")
    task = STATE.get_task(task_id)
    if not task:
        return web.json_response({"error": "not found"}, status=404)
    return web.json_response({"task_id": task_id, "prompt": task.prompt or ""})


async def handle_api_filetree(request: web.Request) -> web.Response:
    """Return file tree of current HEAD."""
    files = git_ls_tree(STATE.work_dir)
    return web.json_response(files)


async def handle_api_file(request: web.Request) -> web.Response:
    """Serve file content from work directory or any active worktree."""
    path = request.query.get("path", "")
    if not path:
        return web.json_response({"error": "path required"}, status=400)
    # Collect worktree paths from task nodes
    worktree_roots = []
    for t in STATE.tasks.values():
        if t.worktree_path:
            worktree_roots.append(t.worktree_path)
    content = read_repo_file(STATE.work_dir, path, extra_roots=worktree_roots)
    if content is None:
        return web.json_response({"error": "not found"}, status=404)
    return web.Response(text=content, content_type="text/plain")


# ═══════════════════════════════════════════════════════════════════════════
# SSE Endpoint
# ═══════════════════════════════════════════════════════════════════════════

async def handle_sse(request: web.Request) -> web.StreamResponse:
    """Server-Sent Events endpoint for real-time updates."""
    resp = web.StreamResponse()
    resp.headers["Content-Type"] = "text/event-stream"
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    await resp.prepare(request)

    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    SSE_CLIENTS.add(q)

    try:
        # Send initial full state
        init_data = json.dumps({"type": "init", "data": STATE.to_dict()})
        await resp.write(f"data: {init_data}\n\n".encode())

        # Send initial git log
        git_entries = git_log_oneline(STATE.work_dir)
        git_data = json.dumps({"type": "gitlog", "data": git_entries})
        await resp.write(f"data: {git_data}\n\n".encode())

        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=15)
                await resp.write(f"data: {payload}\n\n".encode())
            except asyncio.TimeoutError:
                # Send keepalive
                await resp.write(b": keepalive\n\n")
            except (ConnectionResetError, ConnectionError):
                break
    finally:
        SSE_CLIENTS.discard(q)

    return resp


# ═══════════════════════════════════════════════════════════════════════════
# Web Server Setup
# ═══════════════════════════════════════════════════════════════════════════

async def start_web_server(port: int) -> web.AppRunner:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/state", handle_api_state)
    app.router.add_get("/api/events", handle_sse)
    app.router.add_get("/api/gitlog", handle_api_gitlog)
    app.router.add_get("/api/gitdiff", handle_api_gitdiff)
    app.router.add_get("/api/task_output", handle_api_task_output)
    app.router.add_get("/api/task_prompt", handle_api_task_prompt)
    app.router.add_get("/api/filetree", handle_api_filetree)
    app.router.add_get("/api/file", handle_api_file)
    app.router.add_post("/api/skip", handle_api_skip)
    app.router.add_post("/api/stop_revising", handle_api_stop_revising)
    app.router.add_post("/api/pause", handle_api_pause)
    app.router.add_post("/api/resume", handle_api_resume)
    app.router.add_post("/api/reduce_rounds", handle_api_reduce_rounds)
    app.router.add_post("/api/checkpoint", handle_api_checkpoint)
    app.router.add_post("/api/stop", handle_api_stop)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    log.info(f"Control server: http://localhost:{port}/")
    return runner


# ═══════════════════════════════════════════════════════════════════════════
# SPA Dashboard HTML
# ═══════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bureau Control</title>
<style>
:root {
  --bg: #12121e; --surface: #1e1e32; --surface2: #2a2a48;
  --text: #e0e0f0; --muted: #8888aa; --accent: #4a9eff;
  --green: #4caf50; --orange: #ffa500; --red: #f44336; --gray: #666;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text);
       padding: 12px; font-size: 13px; line-height: 1.4; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }

/* Header */
.header { display: flex; justify-content: space-between; align-items: center;
          padding: 8px 12px; background: var(--surface); border-radius: 6px;
          margin-bottom: 10px; flex-wrap: wrap; gap: 8px; }
.header h1 { font-size: 16px; white-space: nowrap; }
.stats { display: flex; gap: 12px; flex-wrap: wrap; }
.stat { text-align: center; }
.stat-val { font-size: 16px; font-weight: bold; color: var(--accent); }
.stat-lbl { font-size: 10px; color: var(--muted); text-transform: uppercase; }
.controls { display: flex; gap: 6px; flex-wrap: wrap; }
.controls button { padding: 5px 12px; border: none; border-radius: 4px;
                   cursor: pointer; font-size: 12px; font-family: var(--font); }
.btn-pause { background: var(--orange); color: #000; }
.btn-resume { background: var(--green); color: #fff; }
.btn-save { background: var(--accent); color: #fff; }
.btn-stop { background: var(--red); color: #fff; }
.btn-sm { padding: 2px 8px; font-size: 11px; border: 1px solid #555;
          background: var(--surface2); color: var(--text); border-radius: 3px;
          cursor: pointer; margin-left: 4px; }
.btn-sm:hover { background: #444; }

/* Banners */
.banner { padding: 8px 12px; border-radius: 4px; margin-bottom: 8px;
          font-weight: bold; text-align: center; }
.banner-paused { background: var(--orange); color: #000; }
.banner-rate { background: #553300; color: var(--orange); }
.banner-cap { background: #553300; color: var(--red); }

/* Phase */
.phase-bar { padding: 6px 12px; background: var(--surface); border-radius: 4px;
             margin-bottom: 8px; color: var(--accent); font-weight: bold; }

/* Task tree */
.task-tree { margin-bottom: 10px; }
.task { margin: 2px 0; padding: 4px 8px; background: var(--surface);
        border-radius: 4px; border-left: 3px solid var(--gray); }
.task-row { display: flex; align-items: center; gap: 6px; cursor: pointer;
            min-height: 22px; flex-wrap: wrap; }
.task-dot { font-size: 14px; flex-shrink: 0; }
.task-label { font-weight: bold; }
.task-status { color: var(--muted); font-size: 12px; }
.task-model { color: #7a7a9a; font-size: 11px; font-style: italic; }
.task-wt { color: var(--orange); font-size: 11px; }
.task-rounds { color: var(--gray); font-size: 11px; }
.task-actions { margin-left: auto; display: flex; gap: 4px; flex-shrink: 0; }
.task-detail { display: none; padding: 4px 8px 4px 24px; font-size: 12px; }
.task-detail.open { display: block; }
.task-detail-collapsed { padding: 2px 8px 2px 24px; font-size: 12px; }
.task-desc { color: var(--muted); margin-bottom: 4px; }
.task-desc-short { color: var(--muted); }
.task-deps { color: #667; font-size: 11px; }
.task-output { margin-top: 4px; }
.task-output-line { color: #9a9abb; font-size: 11px; font-family: monospace;
                    white-space: pre-wrap; word-break: break-all; }
.task-output-line a { color: var(--accent); }
.task-output-last { color: #9a9abb; font-size: 11px; font-family: monospace; }
.task-output-last a { color: var(--accent); }
.task-children { padding-left: 18px; }
.dot-pending { color: var(--gray); }
.dot-running { color: var(--accent); }
.dot-paused { color: var(--orange); }
.dot-completed { color: var(--green); }
.dot-failed { color: var(--red); }
.dot-skipped { color: #555; }
.task-error { color: var(--red); font-size: 11px; margin-top: 2px; }

/* Task prompt */
.task-prompt { margin-top: 6px; padding: 8px; background: #111; border-radius: 4px;
               font-family: monospace; font-size: 11px; white-space: pre-wrap;
               word-break: break-all; max-height: 500px; overflow: auto;
               color: #b0b0cc; border: 1px solid #333; }

/* Bottom panels */
.bottom-panels { display: flex; gap: 10px; }
.bottom-panels > div { flex: 1; min-width: 0; }

/* Git log */
.git-section { background: var(--surface); border-radius: 6px; padding: 8px 12px; }
.git-section h3 { font-size: 13px; margin-bottom: 6px; color: var(--muted); }
.git-entry { padding: 2px 0; cursor: pointer; font-family: monospace;
             font-size: 12px; display: flex; align-items: flex-start; gap: 6px; }
.git-entry:hover { color: var(--accent); }
.git-hash { color: var(--orange); flex-shrink: 0; }
.git-msg { color: var(--text); }
.git-diff { display: none; padding: 6px; margin: 4px 0; background: #111;
            border-radius: 4px; font-family: monospace; font-size: 11px;
            white-space: pre-wrap; word-break: break-all; max-height: 400px;
            overflow: auto; color: #ccc; }
.git-diff.open { display: block; }
.diff-add { color: #4caf50; }
.diff-del { color: #f44336; }
.diff-hdr { color: #4a9eff; }

/* File tree */
.file-tree-section { background: var(--surface); border-radius: 6px; padding: 8px 12px; }
.file-tree-section h3 { font-size: 13px; margin-bottom: 6px; color: var(--muted); }
.ft-dir { font-size: 12px; color: var(--muted); padding: 1px 0; }
.ft-file { font-size: 12px; padding: 1px 0; }
.ft-file a { color: var(--text); font-family: monospace; }
.ft-file a:hover { color: var(--accent); }

/* File preview popover */
.file-preview { position: fixed; z-index: 1000; max-width: 600px; max-height: 400px;
                overflow: auto; background: #1a1a2e; border: 1px solid #444;
                border-radius: 6px; padding: 8px 10px; font-family: monospace;
                font-size: 11px; white-space: pre-wrap; word-break: break-all;
                color: #ddd; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                pointer-events: none; }

/* Utilization bar */
.util-bar { height: 6px; background: #333; border-radius: 3px; margin: 4px 0;
            overflow: hidden; }
.util-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }

/* Connection indicator */
.conn { position: fixed; top: 4px; right: 8px; font-size: 10px; }
.conn-ok { color: var(--green); }
.conn-lost { color: var(--red); }

/* File link */
.file-link { color: var(--accent) !important; cursor: pointer; }
</style>
</head>
<body>

<div class="conn" id="conn">● connected</div>

<div class="header">
  <h1>🏢 Bureau</h1>
  <div class="stats">
    <div class="stat"><div class="stat-val" id="s-agents">0</div><div class="stat-lbl">Agents</div></div>
    <div class="stat"><div class="stat-val" id="s-cost">$0</div><div class="stat-lbl">Cost</div></div>
    <div class="stat"><div class="stat-val" id="s-tokens">0</div><div class="stat-lbl">Tokens</div></div>
    <div class="stat"><div class="stat-val" id="s-commits">0</div><div class="stat-lbl">Commits</div></div>
    <div class="stat"><div class="stat-val" id="s-util">—</div><div class="stat-lbl">Utilization</div></div>
  </div>
  <div class="controls">
    <button class="btn-pause" id="btn-pause" onclick="api('pause')">⏸ Pause</button>
    <button class="btn-resume" id="btn-resume" onclick="api('resume')" style="display:none">▶ Resume</button>
    <button class="btn-sm" onclick="reduceRounds()">↓ Rounds</button>
    <button class="btn-save" onclick="api('checkpoint')">💾</button>
    <button class="btn-stop" onclick="if(confirm('Stop?'))api('stop')">⏹</button>
  </div>
</div>

<div id="banner-paused" class="banner banner-paused" style="display:none">⏸ PAUSED</div>
<div id="banner-rate" class="banner banner-rate" style="display:none"></div>
<div id="phase-bar" class="phase-bar">Phase: —</div>

<div id="task-tree" class="task-tree"></div>

<div class="bottom-panels">
  <div class="git-section">
    <h3>Git Log</h3>
    <div id="git-log"></div>
  </div>
  <div class="file-tree-section">
    <h3>Files <span class="btn-sm" onclick="loadFileTree()" style="font-weight:normal">↻</span></h3>
    <div id="file-tree"><div style="color:var(--muted)">Loading...</div></div>
  </div>
</div>

<div id="file-preview-el" class="file-preview" style="display:none"></div>

<script>
// State
let S = {tasks:{}, root_task_ids:[], paused:false};
let gitLog = [];
let fileTree = [];
let expandedTasks = new Set();
let expandedOutputs = new Set();
let expandedPrompts = {};
let expandedDiffs = {};
let renderRAF = null;
let previewTimer = null;
let previewCache = {};

function scheduleRender() {
  if (renderRAF) return;
  renderRAF = requestAnimationFrame(() => { renderRAF = null; renderTasks(); });
}

// SSE
let es;
function connectSSE() {
  es = new EventSource('/api/events');
  document.getElementById('conn').className = 'conn conn-ok';
  document.getElementById('conn').textContent = '● connected';

  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    switch(msg.type) {
      case 'init': S = msg.data; renderAll(); loadFileTree(); break;
      case 'task': updateTask(msg.data); break;
      case 'output': appendOutput(msg.data.task_id, msg.data.line); break;
      case 'stats': updateStats(msg.data); break;
      case 'git_commit': addGitEntry(msg.data); loadFileTree(); break;
      case 'gitlog': gitLog = msg.data; renderGit(); break;
      case 'phase': updatePhase(msg.data.name); break;
      case 'rate_limit': updateRateLimit(msg.data); break;
    }
  };

  es.onerror = () => {
    document.getElementById('conn').className = 'conn conn-lost';
    document.getElementById('conn').textContent = '● disconnected';
    es.close();
    setTimeout(connectSSE, 3000);
  };
}

// API
async function api(endpoint, data) {
  const res = await fetch('/api/' + endpoint, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data || {})
  });
  return res.json();
}

function reduceRounds() {
  const r = prompt('Set max revision rounds for pending tasks:', '1');
  if (r) api('reduce_rounds', {rounds: parseInt(r)});
}

// Render
function renderAll() {
  updateStats(S);
  renderTasks();
  updatePhase(S.current_phase);
  document.getElementById('banner-paused').style.display = S.paused ? '' : 'none';
  document.getElementById('btn-pause').style.display = S.paused ? 'none' : '';
  document.getElementById('btn-resume').style.display = S.paused ? '' : 'none';
}

function updateStats(d) {
  if (d.total_agents !== undefined) document.getElementById('s-agents').textContent = d.total_agents;
  if (d.total_cost !== undefined) document.getElementById('s-cost').textContent = '$' + d.total_cost.toFixed(4);
  if (d.total_input_tokens !== undefined || d.total_output_tokens !== undefined) {
    const inp = d.total_input_tokens || S.total_input_tokens || 0;
    const out = d.total_output_tokens || S.total_output_tokens || 0;
    document.getElementById('s-tokens').textContent = ((inp+out)/1000).toFixed(0) + 'k';
  }
  if (d.total_commits !== undefined) document.getElementById('s-commits').textContent = d.total_commits;
  if (d.rate_limit_utilization != null) {
    document.getElementById('s-util').textContent = (d.rate_limit_utilization * 100).toFixed(0) + '%';
  }
  if (d.paused !== undefined) {
    S.paused = d.paused;
    document.getElementById('banner-paused').style.display = d.paused ? '' : 'none';
    document.getElementById('btn-pause').style.display = d.paused ? 'none' : '';
    document.getElementById('btn-resume').style.display = d.paused ? '' : 'none';
  }
}

function updatePhase(name) {
  S.current_phase = name;
  document.getElementById('phase-bar').textContent = 'Phase: ' + (name || '—');
}

function updateRateLimit(d) {
  const el = document.getElementById('banner-rate');
  if (d.status === 'rejected' || (d.utilization && d.utilization > 0.8)) {
    let text = '⏳ Rate limit: ';
    if (d.utilization != null) text += (d.utilization * 100).toFixed(0) + '% utilized';
    if (d.type) text += ' (' + d.type + ')';
    if (d.status === 'rejected') text += ' — PAUSED';
    el.textContent = text;
    el.style.display = '';
  } else {
    el.style.display = 'none';
  }
}

function updateTask(td) {
  if (!S.tasks) S.tasks = {};
  const existing = S.tasks[td.id];
  // Preserve output_lines if not in update
  if (existing && (!td.output_lines || td.output_lines.length === 0)) {
    td.output_lines = existing.output_lines || [];
  }
  // Merge children arrays (server may send parent updates + new children)
  if (existing && existing.children && td.children) {
    for (const cid of existing.children) {
      if (!td.children.includes(cid)) td.children.push(cid);
    }
  }
  S.tasks[td.id] = td;
  if (!td.parent_id && S.root_task_ids && !S.root_task_ids.includes(td.id)) {
    S.root_task_ids.push(td.id);
  }
  // Link child to parent (belt-and-suspenders with server-side)
  if (td.parent_id && S.tasks[td.parent_id]) {
    const parent = S.tasks[td.parent_id];
    if (!parent.children) parent.children = [];
    if (!parent.children.includes(td.id)) parent.children.push(td.id);
  }
  scheduleRender();
}

function appendOutput(taskId, line) {
  if (!S.tasks[taskId]) return;
  if (!S.tasks[taskId].output_lines) S.tasks[taskId].output_lines = [];
  S.tasks[taskId].output_lines.push(line);
  scheduleRender();
}

// ── File path linkification ────────────────────────────────────────────
// Build a set of known file paths from task read/write sets and the file
// tree, then match only those exact paths (and their basenames) in output.
function getKnownFiles() {
  const known = new Set();
  if (fileTree) fileTree.forEach(f => known.add(f));
  for (const t of Object.values(S.tasks || {})) {
    (t.file_reads || []).forEach(f => known.add(f));
    (t.file_writes || []).forEach(f => known.add(f));
  }
  return known;
}

function linkifyLine(text) {
  const known = getKnownFiles();
  if (known.size === 0) return esc(text);
  // Build a single regex that matches any known path found in the text.
  // Use longest-match-first ordering so "/a/b/c.rs" matches before "c.rs".
  const sorted = [...known].sort((a, b) => b.length - a.length);
  // Escape regex special chars in the file paths
  const escaped = sorted.map(p => p.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  const re = new RegExp('(' + escaped.join('|') + ')', 'g');
  // Split on matches, escape each segment, wrap matches in links
  const parts = text.split(re);
  let html = '';
  for (const part of parts) {
    if (known.has(part)) {
      html += '<a href="/api/file?path=' + encodeURIComponent(part) + '" class="file-link" data-path="' + escAttr(part) + '" target="_blank" onmouseenter="startPreview(event)" onmouseleave="cancelPreview(event)" onclick="event.stopPropagation()">' + esc(part) + '</a>';
    } else {
      html += esc(part);
    }
  }
  return html;
}

// ── File preview popover ───────────────────────────────────────────────
function startPreview(event) {
  cancelPreview();
  const el = event.target.closest('.file-link');
  if (!el) return;
  const path = el.dataset.path;
  previewTimer = setTimeout(async () => {
    try {
      let content = previewCache[path];
      if (content === undefined) {
        const res = await fetch('/api/file?path=' + encodeURIComponent(path));
        if (!res.ok) return;
        content = await res.text();
        previewCache[path] = content;
      }
      const prev = document.getElementById('file-preview-el');
      prev.textContent = content.substring(0, 8000);
      const rect = el.getBoundingClientRect();
      // Position below or above the link depending on viewport space
      const spaceBelow = window.innerHeight - rect.bottom;
      if (spaceBelow > 200) {
        prev.style.top = (rect.bottom + 4) + 'px';
        prev.style.bottom = '';
      } else {
        prev.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
        prev.style.top = '';
      }
      prev.style.left = Math.min(rect.left, window.innerWidth - 620) + 'px';
      prev.style.display = 'block';
    } catch(e) {}
  }, 400);
}

function cancelPreview() {
  if (previewTimer) { clearTimeout(previewTimer); previewTimer = null; }
  document.getElementById('file-preview-el').style.display = 'none';
}

// ── Task rendering ─────────────────────────────────────────────────────
function renderTasks() {
  const el = document.getElementById('task-tree');
  // Save scroll positions of open prompt/output containers before rebuild
  const savedScrolls = {};
  el.querySelectorAll('.task-prompt, .task-output').forEach(c => {
    if (c.id && c.scrollTop > 0) savedScrolls[c.id] = c.scrollTop;
  });
  let html = '';
  for (const rid of (S.root_task_ids || [])) {
    html += renderTaskNode(rid, 0);
  }
  el.innerHTML = html || '<div style="color:var(--muted);padding:20px">No tasks yet...</div>';
  // Restore scroll positions
  for (const [id, top] of Object.entries(savedScrolls)) {
    const c = document.getElementById(id);
    if (c) c.scrollTop = top;
  }
}

function renderTaskNode(taskId, depth) {
  const t = S.tasks[taskId];
  if (!t) return '';

  const dotClass = 'dot-' + t.status;
  const elapsed = t.started_at ?
    ((t.finished_at || Date.now()/1000) - t.started_at).toFixed(1) + 's' : '';
  const isExpanded = expandedTasks.has(taskId);
  const isOutputExpanded = expandedOutputs.has(taskId);
  const statusText = t.status === 'running' && t.task_type ? t.task_type : t.status;
  const nLines = t.output_lines ? t.output_lines.length : 0;
  const lastLine = nLines > 0 ? t.output_lines[nLines - 1] : '';
  const wt = t.worktree_path ? '⎇wt' : '⎇main';
  const wtClass = t.worktree_path ? 'task-wt' : 'task-model';
  const modelStr = t.model ? t.model.replace('claude-', '').substring(0, 15) : '';
  const effortStr = t.effort || '';
  const thinkStr = t.thinking || '';
  const configStr = [modelStr, effortStr, thinkStr].filter(Boolean).join('/');
  const nChildren = (t.children && t.children.length > 0) ? t.children.length : 0;
  const desc = t.description || '';
  const descShort = desc.length > 80 ? desc.substring(0, 80) + '…' : desc;
  const showRounds = t.max_revision_rounds > 0;

  let actions = '';
  if (t.status === 'pending') {
    actions += `<button class="btn-sm" onclick="event.stopPropagation();api('skip',{task_id:'${t.id}'})">Skip</button>`;
  }
  if (t.status === 'running') {
    actions += `<button class="btn-sm" onclick="event.stopPropagation();api('stop_revising',{task_id:'${t.id}'})">Finish</button>`;
    actions += `<button class="btn-sm" onclick="event.stopPropagation();api('skip',{task_id:'${t.id}'})">Skip</button>`;
  }

  let detail = '';
  if (isExpanded) {
    // Expanded: full description, file deps, error, output toggle, prompt toggle
    const reads = (t.file_reads || []).join(', ');
    const writes = (t.file_writes || []).join(', ');
    const grp = t.concurrency_group != null ? 'Group ' + t.concurrency_group : '';
    const err = t.error ? `<div class="task-error">${esc(t.error)}</div>` : '';
    let depsLine = '';
    if (reads || writes || grp) {
      const parts = [];
      if (writes) parts.push('W: ' + esc(writes));
      if (reads) parts.push('R: ' + esc(reads));
      if (grp) parts.push(grp);
      depsLine = `<div class="task-deps">${parts.join(' | ')}</div>`;
    }

    let outputHtml = '';
    if (nLines > 0) {
      if (isOutputExpanded) {
        outputHtml = `<div style="margin-top:4px"><span class="btn-sm" onclick="event.stopPropagation();toggleOutput('${t.id}')">▾ Output (${nLines})</span></div>` +
          `<div class="task-output" id="out-${t.id}">` +
          t.output_lines.map(l => `<div class="task-output-line">${linkifyLine(l)}</div>`).join('') +
          `</div>`;
      } else {
        outputHtml = `<div style="margin-top:4px"><span class="btn-sm" onclick="event.stopPropagation();toggleOutput('${t.id}')">▸ Output (${nLines})</span>` +
          ` <span class="task-output-last">▸ ${linkifyLine(lastLine)}</span></div>`;
      }
    }

    // Prompt toggle
    let promptHtml = '';
    if (t.has_prompt) {
      const isPromptOpen = expandedPrompts[t.id];
      if (isPromptOpen) {
        promptHtml = `<div style="margin-top:4px"><span class="btn-sm" onclick="event.stopPropagation();togglePrompt('${t.id}')">▾ Full Prompt</span></div>` +
          `<div class="task-prompt" id="prompt-${t.id}">${expandedPrompts[t.id] === true ? 'Loading...' : esc(expandedPrompts[t.id])}</div>`;
      } else {
        promptHtml = `<div style="margin-top:4px"><span class="btn-sm" onclick="event.stopPropagation();togglePrompt('${t.id}')">▸ Full Prompt</span></div>`;
      }
    }

    detail = `<div class="task-detail open">
      <div class="task-desc">${esc(desc)}</div>
      ${depsLine}${err}
      ${outputHtml}
      ${promptHtml}
    </div>`;
  } else {
    // Collapsed: first 80 chars of description + last output line
    detail = `<div class="task-detail-collapsed">`
      + `<span class="task-desc-short">${esc(descShort)}</span>`
      + (lastLine ? ` <span class="task-output-last">▸ ${linkifyLine(lastLine)}</span>` : '')
      + `</div>`;
  }

  let childrenHtml = '';
  if (t.children && t.children.length > 0) {
    childrenHtml = '<div class="task-children">';
    for (const cid of t.children) {
      childrenHtml += renderTaskNode(cid, depth + 1);
    }
    childrenHtml += '</div>';
  }

  return `<div class="task" style="margin-left:${depth*12}px" data-id="${t.id}">
    <div class="task-row" onclick="toggleExpand('${t.id}')">
      <span class="task-dot ${dotClass}">●</span>
      <span class="task-label">${esc(t.label)}</span>
      ${nChildren ? `<span class="task-model">(${nChildren})</span>` : ''}
      <span class="task-status">[${esc(statusText)}]${elapsed ? ' ' + elapsed : ''}</span>
      ${showRounds ? `<span class="task-rounds">r${t.revision_round}/${t.max_revision_rounds}</span>` : ''}
      ${configStr ? `<span class="task-model">${esc(configStr)}</span>` : ''}
      <span class="${wtClass}">${wt}</span>
      <span class="task-actions">${actions}</span>
    </div>
    ${detail}
    ${childrenHtml}
  </div>`;
}

function toggleExpand(taskId) {
  if (expandedTasks.has(taskId)) expandedTasks.delete(taskId);
  else expandedTasks.add(taskId);
  renderTasks();
}

function toggleOutput(taskId) {
  if (expandedOutputs.has(taskId)) expandedOutputs.delete(taskId);
  else expandedOutputs.add(taskId);
  renderTasks();
}

async function togglePrompt(taskId) {
  if (expandedPrompts[taskId]) {
    delete expandedPrompts[taskId];
    renderTasks();
    return;
  }
  expandedPrompts[taskId] = true; // loading state
  renderTasks();
  try {
    const res = await fetch('/api/task_prompt?task_id=' + encodeURIComponent(taskId));
    const data = await res.json();
    expandedPrompts[taskId] = data.prompt || '(no prompt stored)';
  } catch(e) {
    expandedPrompts[taskId] = '(error loading prompt)';
  }
  renderTasks();
}

// ── Git log ────────────────────────────────────────────────────────────
function renderGit() {
  const el = document.getElementById('git-log');
  let html = '';
  for (const entry of gitLog) {
    const h = entry.hash;
    html += `<div>
      <div class="git-entry" onclick="toggleDiff('${h}')">
        <span class="git-hash">${esc(h)}</span>
        <span class="git-msg">${esc(entry.message)}</span>
      </div>
      <div class="git-diff" id="diff-${h}"></div>
    </div>`;
  }
  el.innerHTML = html || '<div style="color:var(--muted)">No commits yet</div>';
}

function addGitEntry(entry) {
  gitLog.unshift(entry);
  renderGit();
}

async function toggleDiff(hash) {
  const el = document.getElementById('diff-' + hash);
  if (!el) return;
  if (el.classList.contains('open')) {
    el.classList.remove('open');
    return;
  }
  if (!expandedDiffs[hash]) {
    el.textContent = 'Loading...';
    el.classList.add('open');
    const res = await fetch('/api/gitdiff?hash=' + encodeURIComponent(hash));
    const text = await res.text();
    expandedDiffs[hash] = text;
  }
  el.innerHTML = colorDiff(expandedDiffs[hash]);
  el.classList.add('open');
}

function colorDiff(text) {
  return text.split('\n').map(line => {
    const e = esc(line);
    if (line.startsWith('+') && !line.startsWith('+++'))
      return `<span class="diff-add">${e}</span>`;
    if (line.startsWith('-') && !line.startsWith('---'))
      return `<span class="diff-del">${e}</span>`;
    if (line.startsWith('@@') || line.startsWith('diff ') || line.startsWith('index '))
      return `<span class="diff-hdr">${e}</span>`;
    return e;
  }).join('\n');
}

// ── File tree ──────────────────────────────────────────────────────────
async function loadFileTree() {
  try {
    const res = await fetch('/api/filetree');
    const files = await res.json();
    fileTree = files;
    renderFileTree();
  } catch(e) {}
}

function renderFileTree() {
  const el = document.getElementById('file-tree');
  if (!fileTree || fileTree.length === 0) {
    el.innerHTML = '<div style="color:var(--muted)">No files yet</div>';
    return;
  }
  // Build hierarchical tree
  const tree = {};
  for (const f of fileTree) {
    const parts = f.split('/');
    let node = tree;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (i === parts.length - 1) {
        node['\x00' + p] = f; // prefix with \0 to separate files from dirs
      } else {
        if (!node[p] || typeof node[p] === 'string') node[p] = {};
        node = node[p];
      }
    }
  }
  el.innerHTML = renderTreeNode(tree, 0);
}

function renderTreeNode(node, depth) {
  let html = '';
  const entries = Object.entries(node).sort(([a], [b]) => {
    const aFile = a.startsWith('\x00');
    const bFile = b.startsWith('\x00');
    if (aFile !== bFile) return aFile ? 1 : -1; // dirs first
    return a.replace('\x00','').localeCompare(b.replace('\x00',''));
  });
  for (const [key, val] of entries) {
    if (typeof val === 'string') {
      // file leaf
      const name = key.substring(1); // strip \0 prefix
      html += `<div class="ft-file" style="padding-left:${depth*14+4}px"><a href="/api/file?path=${encodeURIComponent(val)}" class="file-link" data-path="${escAttr(val)}" target="_blank" onmouseenter="startPreview(event)" onmouseleave="cancelPreview(event)">📄 ${esc(name)}</a></div>`;
    } else {
      // directory
      html += `<div class="ft-dir" style="padding-left:${depth*14+4}px">📁 ${esc(key)}</div>`;
      html += renderTreeNode(val, depth + 1);
    }
  }
  return html;
}

// ── Utilities ──────────────────────────────────────────────────────────
function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function escAttr(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Start
connectSSE();
</script>
</body>
</html>
"""
