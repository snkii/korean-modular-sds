#!/usr/bin/env python3
"""
English Modular SDS – integrated voice dialogue server

STT (port 8081) + TTS (port 8082) + LLM (port 8083) run as subprocesses;
integrated UI served on port 8080.

Usage: python run_eng.py [--port 8080]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

STT_URL = "http://127.0.0.1:8081"
TTS_URL = "http://127.0.0.1:8082"
LLM_URL = "http://127.0.0.1:8083"

app = Flask(__name__)
_procs: list[subprocess.Popen] = []

_ready = {"stt": False, "tts": False, "llm": False}


# ── Module start / stop ────────────────────────────────────────────────────────

def _start_module(conda_env: str, subdir: str, script: str = "server.py") -> subprocess.Popen:
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", conda_env, "python", script],
        cwd=os.path.join(ROOT_DIR, subdir),
    )
    _procs.append(proc)
    return proc


def _wait_ready(url: str, label: str, key: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                print(f"  ✓ {label} ready")
                _ready[key] = True
                return
        except Exception:
            pass
        time.sleep(3)
    print(f"  ✗ {label} failed to start ({timeout}s timeout)", file=sys.stderr)


def start_modules() -> None:
    print("=" * 52)
    print("  English Modular SDS starting...")
    print("=" * 52)
    print("\n[1/3] STT module (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "stt_module", "server_en.py")
    print("[2/3] TTS module (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "tts_module")
    print("[3/3] LLM module (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "llm_module")

    print("\nWaiting for modules (parallel)...")
    threads = [
        threading.Thread(target=_wait_ready, args=(STT_URL, "STT", "stt"), daemon=True),
        threading.Thread(target=_wait_ready, args=(TTS_URL, "TTS", "tts"), daemon=True),
        threading.Thread(target=_wait_ready, args=(LLM_URL, "LLM", "llm"), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\n" + "=" * 52)
    print("  All modules ready!")
    print("  UI: http://0.0.0.0:8080")
    print("=" * 52)


def cleanup() -> None:
    for p in _procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in _procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass


# ── Proxy routes ───────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    return jsonify(_ready)


@app.route("/stt/stream")
def stt_stream():
    def gen():
        while True:
            try:
                with requests.get(f"{STT_URL}/stream", stream=True, timeout=None) as r:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                return
            except Exception:
                yield ": waiting\n\n"
                time.sleep(3)
    return Response(
        stream_with_context(gen()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/stt/transcribe", methods=["POST"])
def stt_transcribe():
    if not _ready["stt"]:
        return jsonify(error="STT module not ready"), 503
    try:
        f = request.files["audio"]
        resp = requests.post(
            f"{STT_URL}/transcribe",
            files={"audio": (f.filename or "chunk.webm", f.stream, f.content_type)},
            timeout=15,
        )
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        return jsonify(error=str(e)), 503


@app.route("/llm/chat", methods=["POST"])
def llm_chat():
    if not _ready["llm"]:
        def err():
            yield f"data: {json.dumps({'type': 'error', 'text': 'LLM module not ready'})}\n\n"
        return Response(stream_with_context(err()), content_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    body = request.get_json()
    def gen():
        try:
            with requests.post(f"{LLM_URL}/chat", json=body, stream=True, timeout=None) as r:
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
    return Response(
        stream_with_context(gen()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/llm/reset", methods=["POST"])
def llm_reset():
    if not _ready["llm"]:
        return jsonify(ok=False, error="LLM module not ready"), 503
    try:
        resp = requests.post(f"{LLM_URL}/reset", json=request.get_json(), timeout=10)
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        return jsonify(error=str(e)), 503


@app.route("/tts/synthesize", methods=["POST"])
def tts_synthesize():
    if not _ready["tts"]:
        return jsonify(error="TTS module not ready"), 503
    try:
        resp = requests.post(f"{TTS_URL}/synthesize", json=request.get_json(), timeout=10)
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        return jsonify(error=str(e)), 503


@app.route("/tts/stream/<job_id>")
def tts_stream(job_id):
    def gen():
        try:
            with requests.get(f"{TTS_URL}/stream/{job_id}", stream=True, timeout=None) as r:
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as e:
            yield f"event: error\ndata: {e}\n\n"
    return Response(
        stream_with_context(gen()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>English Modular SDS</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #f0f2f5;
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 230px;
  background: #1e1e2e;
  color: #cdd6f4;
  display: flex;
  flex-direction: column;
  padding: 18px 14px;
  gap: 14px;
  flex-shrink: 0;
  overflow-y: auto;
}
.sidebar h3 {
  font-size: 11px;
  color: #6c7086;
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: 2px;
}
.sidebar label {
  font-size: 12px;
  color: #a6adc8;
  display: block;
  margin-bottom: 4px;
}
.sidebar select,
.sidebar textarea {
  width: 100%;
  background: #313244;
  border: 1px solid #45475a;
  color: #cdd6f4;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
  font-family: inherit;
}
.sidebar textarea { resize: vertical; min-height: 72px; }
.range-row { display: flex; align-items: center; gap: 8px; }
.range-row input[type=range] { flex: 1; }
.val-lbl { font-size: 12px; color: #a6adc8; min-width: 34px; text-align: right; }
.new-chat-btn {
  background: #89b4fa;
  color: #1e1e2e;
  border: none;
  border-radius: 8px;
  padding: 9px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  margin-top: auto;
}
.new-chat-btn:hover { opacity: .85; }

/* ── Chat area ── */
.chat-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

.chat-header {
  padding: 13px 18px;
  background: white;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  align-items: center;
  gap: 10px;
}
.header-title { font-size: 15px; font-weight: 600; color: #333; }
.status-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  background: #ccc;
  flex-shrink: 0;
  transition: background .3s;
}
.status-dot.listening { background: #a6e3a1; animation: pulse 1.2s infinite; }
.status-dot.thinking  { background: #89b4fa; animation: pulse .8s  infinite; }
.status-dot.speaking  { background: #fab387; animation: pulse .6s  infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
.status-txt { font-size: 12px; color: #888; }

/* ── Messages ── */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.msg { display: flex; flex-direction: column; max-width: 80%; }
.msg.user      { align-self: flex-end;   align-items: flex-end; }
.msg.assistant { align-self: flex-start; align-items: flex-start; }

.role-label {
  font-size: 11px;
  color: #aaa;
  margin-bottom: 3px;
  padding: 0 4px;
}

.bubble {
  padding: 10px 14px;
  border-radius: 18px;
  font-size: 15px;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
}
.msg.user .bubble {
  background: #89b4fa;
  color: #1e1e2e;
  border-bottom-right-radius: 4px;
}
.msg.assistant .bubble {
  background: white;
  color: #333;
  border: 1px solid #e4e4e4;
  border-bottom-left-radius: 4px;
  box-shadow: 0 1px 4px rgba(0,0,0,.05);
}
.cursor {
  display: inline-block;
  width: 2px; height: 1em;
  background: #89b4fa;
  animation: blink .7s infinite;
  vertical-align: text-bottom;
  margin-left: 1px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── Loading banner ── */
.loading-banner {
  display: none;
  background: #fab387;
  color: #1e1e2e;
  font-size: 13px;
  font-weight: 600;
  padding: 7px 18px;
  text-align: center;
  gap: 8px;
  align-items: center;
  justify-content: center;
}
.loading-banner.show { display: flex; }
.module-pills { display: flex; gap: 6px; margin-left: 8px; }
.pill {
  font-size: 11px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 10px;
  background: rgba(0,0,0,.15);
  color: #1e1e2e;
}
.pill.ready { background: #a6e3a1; }

/* ── Bottom bar ── */
.bottom-bar {
  padding: 10px 18px 14px;
  background: white;
  border-top: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}
.partial-area {
  width: 100%;
  min-height: 20px;
  font-size: 13px;
  color: #aaa;
  font-style: italic;
  text-align: center;
}
.mic-btn {
  width: 62px; height: 62px;
  border-radius: 50%;
  border: none;
  background: #89b4fa;
  color: #1e1e2e;
  font-size: 26px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all .2s;
  box-shadow: 0 3px 14px rgba(137,180,250,.4);
  user-select: none;
}
.mic-btn.recording {
  background: #f38ba8;
  box-shadow: 0 3px 18px rgba(243,139,168,.5);
  animation: micPulse 1s infinite;
}
@keyframes micPulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.07)} }
.mic-btn:hover { opacity: .88; }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <h3>Settings</h3>

  <div>
    <label>AI Voice</label>
    <select id="ttsVoice">
      <optgroup label="Male">
        <option value="M1" selected>M1</option>
        <option value="M2">M2</option>
        <option value="M3">M3</option>
        <option value="M4">M4</option>
        <option value="M5">M5</option>
      </optgroup>
      <optgroup label="Female">
        <option value="F1">F1</option>
        <option value="F2">F2</option>
        <option value="F3">F3</option>
        <option value="F4">F4</option>
        <option value="F5">F5</option>
      </optgroup>
    </select>
  </div>

  <div>
    <label>TTS Speed <span class="val-lbl" id="speedLbl">1.05</span></label>
    <div class="range-row">
      <input type="range" id="ttsSpeed" min="0.7" max="1.5" step="0.05" value="1.05"
             oninput="document.getElementById('speedLbl').textContent=parseFloat(this.value).toFixed(2)">
    </div>
  </div>

  <div>
    <label>Temperature <span class="val-lbl" id="tempLbl">0.70</span></label>
    <div class="range-row">
      <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7"
             oninput="document.getElementById('tempLbl').textContent=parseFloat(this.value).toFixed(2)">
    </div>
  </div>

  <div>
    <label>System Prompt</label>
    <textarea id="systemPrompt">You are a helpful, friendly AI assistant. Always respond in English with a single short sentence.</textarea>
  </div>

  <button class="new-chat-btn" onclick="newChat()">＋ New Chat</button>
</div>

<!-- Chat -->
<div class="chat-wrap">
  <div class="chat-header">
    <span class="status-dot" id="statusDot"></span>
    <span class="header-title">English Modular SDS</span>
    <span class="status-txt" id="statusTxt">Press the mic button to start</span>
  </div>

  <div class="loading-banner" id="loadingBanner">
    ⏳ Loading modules...
    <div class="module-pills">
      <span class="pill" id="pillStt">STT</span>
      <span class="pill" id="pillTts">TTS</span>
      <span class="pill" id="pillLlm">LLM</span>
    </div>
  </div>
  <div class="messages" id="messages"></div>

  <div class="bottom-bar">
    <div class="partial-area" id="partialArea"></div>
    <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="Toggle mic">🎤</button>
  </div>
</div>

<script>
// ── Global state ───────────────────────────────────────────────────────────────
const sessionId = Math.random().toString(36).slice(2);

// STT
let isRecording   = false;
let micStream     = null;
let recorder      = null;
let chunkInterval = null;
let sttEs         = null;

// LLM
let llmController = null;

// TTS
let audioCtx     = null;
let ttsNextStart = 0;
let ttsQueue     = [];
let isTTSBusy    = false;
let ttsJobSeq    = 0;
let ttsEsCurrent = null;
let ttsResolve   = null;

// Sentence accumulation
let sentenceBuf = '';


// ── Utils ──────────────────────────────────────────────────────────────────────
function setStatus(dotClass, txt) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot' + (dotClass ? ' ' + dotClass : '');
  document.getElementById('statusTxt').textContent = txt;
}

function scrollBottom() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg ' + role;

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role === 'user' ? 'You' : 'AI';
  wrapper.appendChild(label);

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (text) bubble.textContent = text;
  wrapper.appendChild(bubble);

  document.getElementById('messages').appendChild(wrapper);
  scrollBottom();
  return bubble;
}

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}

function getAudioCtx() {
  if (!audioCtx || audioCtx.state === 'closed') {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    ttsNextStart = 0;
  }
  return audioCtx;
}

function scheduleWav(arrayBuffer) {
  const ctx = getAudioCtx();
  ctx.resume().then(() => {
    ctx.decodeAudioData(arrayBuffer).then(buf => {
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      const start = Math.max(ttsNextStart, ctx.currentTime + 0.02);
      src.start(start);
      ttsNextStart = start + buf.duration;
    }).catch(e => console.warn('decodeAudioData:', e));
  });
}


// ── TTS queue ─────────────────────────────────────────────────────────────────

function stopTTS() {
  ttsJobSeq++;
  ttsQueue    = [];
  sentenceBuf = '';
  isTTSBusy   = false;
  if (ttsEsCurrent) { ttsEsCurrent.close(); ttsEsCurrent = null; }
  if (ttsResolve)   { ttsResolve(); ttsResolve = null; }
  if (audioCtx && audioCtx.state !== 'closed') {
    audioCtx.close();
    audioCtx = null;
  }
  ttsNextStart = 0;
}

function enqueueTTS(text) {
  text = text.trim();
  if (!text) return;
  ttsQueue.push(text);
  if (!isTTSBusy) {
    isTTSBusy = true;
    drainTTS(ttsJobSeq);
  }
}

async function drainTTS(mySeq) {
  while (ttsQueue.length > 0 && ttsJobSeq === mySeq) {
    const text = ttsQueue.shift();

    let job_id;
    try {
      const r = await fetch('/tts/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: document.getElementById('ttsVoice').value,
          speed: parseFloat(document.getElementById('ttsSpeed').value),
          lang: 'en',
        }),
      });
      if (ttsJobSeq !== mySeq) return;
      const data = await r.json();
      if (data.error) continue;
      job_id = data.job_id;
    } catch (e) {
      console.warn('TTS synthesize error:', e);
      continue;
    }
    if (ttsJobSeq !== mySeq) return;

    await new Promise(resolve => {
      ttsResolve = resolve;
      const es = new EventSource('/tts/stream/' + job_id);
      ttsEsCurrent = es;

      function done() {
        if (ttsEsCurrent === es) ttsEsCurrent = null;
        if (ttsResolve === resolve) ttsResolve = null;
        es.close();
        resolve();
      }

      es.addEventListener('audio', e => {
        if (ttsJobSeq !== mySeq) { done(); return; }
        scheduleWav(base64ToArrayBuffer(e.data));
        setStatus('speaking', 'Speaking...');
      });
      es.addEventListener('done',  done);
      es.addEventListener('error', done);
      es.onerror = () => { if (ttsJobSeq === mySeq) done(); };
    });
  }

  if (ttsJobSeq !== mySeq) return;

  isTTSBusy = false;
  const ctx = audioCtx;
  if (ctx && ttsNextStart > ctx.currentTime) {
    const remainMs = (ttsNextStart - ctx.currentTime) * 1000 + 300;
    setTimeout(() => {
      if (!isTTSBusy) {
        setStatus(isRecording ? 'listening' : '', isRecording ? 'Listening...' : 'Idle');
      }
    }, remainMs);
  } else {
    setStatus(isRecording ? 'listening' : '', isRecording ? 'Listening...' : 'Idle');
  }
}


// ── Sentence detection (LLM token → TTS) ──────────────────────────────────────
const SENT_RE = /^(.*?[.!?。！？\n])\s*/s;

function feedToken(token) {
  sentenceBuf += token;
  let m;
  while ((m = SENT_RE.exec(sentenceBuf)) !== null) {
    enqueueTTS(m[1]);
    sentenceBuf = sentenceBuf.slice(m[0].length);
  }
  if (sentenceBuf.length > 150) {
    const cut = sentenceBuf.lastIndexOf(' ');
    if (cut > 50) {
      enqueueTTS(sentenceBuf.slice(0, cut));
      sentenceBuf = sentenceBuf.slice(cut + 1);
    }
  }
}

function flushSentenceBuf() {
  if (sentenceBuf.trim()) {
    enqueueTTS(sentenceBuf);
    sentenceBuf = '';
  }
}


// ── LLM ───────────────────────────────────────────────────────────────────────
async function sendToLLM(userText) {
  if (llmController) { llmController.abort(); llmController = null; }
  stopTTS();

  setStatus('thinking', 'Thinking...');

  const bubble = addMessage('assistant');
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.appendChild(cursor);

  llmController = new AbortController();
  let replyText  = '';

  try {
    const res = await fetch('/llm/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message:       userText,
        session_id:    sessionId,
        system_prompt: document.getElementById('systemPrompt').value,
        temperature:   parseFloat(document.getElementById('temperature').value),
      }),
      signal: llmController.signal,
    });

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        let evt;
        try { evt = JSON.parse(line.slice(6)); } catch { continue; }

        if (evt.type === 'reply') {
          replyText += evt.text;
          cursor.remove();
          bubble.textContent = replyText;
          bubble.appendChild(cursor);
          scrollBottom();
          feedToken(evt.text);

        } else if (evt.type === 'done') {
          cursor.remove();
          flushSentenceBuf();
          if (!isTTSBusy) {
            setStatus(isRecording ? 'listening' : '', isRecording ? 'Listening...' : 'Idle');
          }

        } else if (evt.type === 'error') {
          cursor.remove();
          bubble.textContent = (replyText || '') + '\n[Error: ' + evt.text + ']';
          stopTTS();
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      cursor.remove();
      if (!replyText) bubble.closest('.msg').remove();
    } else {
      cursor.remove();
      bubble.textContent = replyText || '[Connection error: ' + e.message + ']';
      stopTTS();
    }
  } finally {
    llmController = null;
  }
}


// ── STT / Mic ─────────────────────────────────────────────────────────────────

function connectSTT() {
  if (sttEs) sttEs.close();
  sttEs = new EventSource('/stt/stream');

  sttEs.onerror = () => {
    sttEs.close();
    sttEs = null;
    setTimeout(connectSTT, 3000);
  };

  sttEs.addEventListener('partial', e => {
    document.getElementById('partialArea').textContent = e.data ? '⏳ ' + e.data : '';
  });

  sttEs.addEventListener('commit', e => {
    const text = (e.data || '').trim();
    document.getElementById('partialArea').textContent = '';
    if (!text) return;
    addMessage('user', text);
    sendToLLM(text);
  });
}

function startChunk(stream) {
  recorder = new MediaRecorder(stream);
  const chunks = [];
  recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: recorder.mimeType });
    if (blob.size >= 500) {
      const form = new FormData();
      form.append('audio', blob, 'chunk.webm');
      fetch('/stt/transcribe', { method: 'POST', body: form }).catch(() => {});
    }
    if (isRecording) startChunk(stream);
  };
  recorder.start();
}

async function toggleMic() {
  const btn = document.getElementById('micBtn');
  if (isRecording) {
    isRecording = false;
    clearInterval(chunkInterval);
    if (recorder && recorder.state !== 'inactive') recorder.stop();
    btn.classList.remove('recording');
    btn.textContent = '🎤';
    if (!isTTSBusy) setStatus('', 'Idle');
  } else {
    if (!micStream) {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        setStatus('', 'Mic error: ' + e.message);
        return;
      }
    }
    isRecording = true;
    btn.classList.add('recording');
    btn.textContent = '⏹';
    setStatus('listening', 'Listening...');
    startChunk(micStream);
    chunkInterval = setInterval(() => {
      if (isRecording && recorder && recorder.state !== 'inactive') recorder.stop();
    }, 200);
  }
}


// ── New chat ───────────────────────────────────────────────────────────────────
function newChat() {
  if (llmController) { llmController.abort(); llmController = null; }
  stopTTS();
  fetch('/llm/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  });
  document.getElementById('messages').innerHTML = '';
  document.getElementById('partialArea').textContent = '';
  setStatus(isRecording ? 'listening' : '', isRecording ? 'Listening...' : 'Press the mic button to start');
}


// ── Module status polling ──────────────────────────────────────────────────────
let _allReady = false;

async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    document.getElementById('pillStt').className = 'pill' + (s.stt ? ' ready' : '');
    document.getElementById('pillTts').className = 'pill' + (s.tts ? ' ready' : '');
    document.getElementById('pillLlm').className = 'pill' + (s.llm ? ' ready' : '');
    const allReady = s.stt && s.tts && s.llm;
    document.getElementById('loadingBanner').className = 'loading-banner' + (allReady ? '' : ' show');
    if (allReady && !_allReady) {
      _allReady = true;
      setStatus('', 'Press the mic button to start');
    }
    if (!allReady) setTimeout(pollStatus, 2000);
  } catch (e) {
    setTimeout(pollStatus, 3000);
  }
}

// ── Init ───────────────────────────────────────────────────────────────────────
document.getElementById('loadingBanner').className = 'loading-banner show';
pollStatus();
connectSTT();
</script>
</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="English Modular SDS – integrated server")
    parser.add_argument("--port", type=int, default=8080, help="UI server port (default: 8080)")
    args = parser.parse_args()

    threading.Thread(target=start_modules, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (cleanup(), sys.exit(0)))

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
