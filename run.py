#!/usr/bin/env python3
"""
Korean Modular SDS – 통합 음성 대화 서버

STT (port 8081) + TTS (port 8082) + LLM (port 8083) 를 subprocess로 실행하고
통합 UI를 port 8080 에서 서빙한다.

실행: python run.py [--port 8080]
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

# 각 모듈의 준비 상태 (False → True)
_ready = {"stt": False, "tts": False, "llm": False}


# ── 모듈 시작 / 종료 ───────────────────────────────────────────────────────────

def _start_module(conda_env: str, subdir: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", conda_env, "python", "server.py"],
        cwd=os.path.join(ROOT_DIR, subdir),
    )
    _procs.append(proc)
    return proc


def _wait_ready(url: str, label: str, key: str, timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                print(f"  ✓ {label} 준비 완료")
                _ready[key] = True
                return
        except Exception:
            pass
        time.sleep(3)
    print(f"  ✗ {label} 시작 실패 ({timeout}s 초과)", file=sys.stderr)


def start_modules() -> None:
    print("=" * 52)
    print("  Korean Modular SDS 시작 중...")
    print("=" * 52)
    print("\n[1/3] STT 모듈 (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "stt_module")
    print("[2/3] TTS 모듈 (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "tts_module")
    print("[3/3] LLM 모듈 (conda: korean-modular-sds) ...")
    _start_module("korean-modular-sds", "llm_module")

    print("\n모듈 준비 대기 중 (병렬)...")
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
    print("  모든 모듈 준비 완료!")
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


# ── 프록시 라우트 ──────────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """모듈 준비 상태 반환"""
    return jsonify(_ready)


@app.route("/stt/stream")
def stt_stream():
    """STT SSE 프록시 – 모듈이 준비될 때까지 연결 유지하며 재시도"""
    def gen():
        while True:
            try:
                with requests.get(f"{STT_URL}/stream", stream=True, timeout=None) as r:
                    for chunk in r.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                return  # 정상 종료
            except Exception:
                # STT 아직 미준비 – 3초 후 재시도 (heartbeat으로 연결 유지)
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
        return jsonify(error="STT 모듈 준비 중"), 503
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
            yield f"data: {json.dumps({'type': 'error', 'text': 'LLM 모듈 준비 중'})}\n\n"
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
        return jsonify(ok=False, error="LLM 모듈 준비 중"), 503
    try:
        resp = requests.post(f"{LLM_URL}/reset", json=request.get_json(), timeout=10)
        return Response(resp.content, status=resp.status_code, content_type="application/json")
    except Exception as e:
        return jsonify(error=str(e)), 503


@app.route("/tts/synthesize", methods=["POST"])
def tts_synthesize():
    if not _ready["tts"]:
        return jsonify(error="TTS 모듈 준비 중"), 503
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
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Korean Modular SDS</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #f0f2f5;
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ── 사이드바 ── */
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

/* ── 채팅 영역 ── */
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

/* ── 메시지 ── */
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

/* ── 로딩 배너 ── */
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

/* ── 하단 바 ── */
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

<!-- 사이드바 -->
<div class="sidebar">
  <h3>설정</h3>

  <div>
    <label>AI 목소리</label>
    <select id="ttsVoice">
      <optgroup label="남성">
        <option value="M1" selected>M1</option>
        <option value="M2">M2</option>
        <option value="M3">M3</option>
        <option value="M4">M4</option>
        <option value="M5">M5</option>
      </optgroup>
      <optgroup label="여성">
        <option value="F1">F1</option>
        <option value="F2">F2</option>
        <option value="F3">F3</option>
        <option value="F4">F4</option>
        <option value="F5">F5</option>
      </optgroup>
    </select>
  </div>

  <div>
    <label>TTS 속도 <span class="val-lbl" id="speedLbl">1.05</span></label>
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
    <label>시스템 프롬프트</label>
    <textarea id="systemPrompt">You are a helpful, friendly AI assistant. Always respond in Korean with a single short sentence.</textarea>
  </div>

  <button class="new-chat-btn" onclick="newChat()">＋ 새 대화</button>
</div>

<!-- 채팅 -->
<div class="chat-wrap">
  <div class="chat-header">
    <span class="status-dot" id="statusDot"></span>
    <span class="header-title">Korean Modular SDS</span>
    <span class="status-txt" id="statusTxt">마이크 버튼을 눌러 시작하세요</span>
  </div>

  <div class="loading-banner" id="loadingBanner">
    ⏳ 모듈 로딩 중...
    <div class="module-pills">
      <span class="pill" id="pillStt">STT</span>
      <span class="pill" id="pillTts">TTS</span>
      <span class="pill" id="pillLlm">LLM</span>
    </div>
  </div>
  <div class="messages" id="messages"></div>

  <div class="bottom-bar">
    <div class="partial-area" id="partialArea"></div>
    <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="마이크 켜기/끄기">🎤</button>
  </div>
</div>

<script>
// ── 전역 상태 ──────────────────────────────────────────────────────────────────
const sessionId = Math.random().toString(36).slice(2);

// STT
let isRecording   = false;
let micStream     = null;
let recorder      = null;
let chunkInterval = null;
let sttEs         = null;

// LLM
let llmController = null;   // AbortController

// TTS
let audioCtx     = null;
let ttsNextStart = 0;       // Web Audio 스케줄 타임
let ttsQueue     = [];      // 합성 대기 중인 문장 목록
let isTTSBusy    = false;   // drainTTS 실행 중 여부
let ttsJobSeq    = 0;       // 인터럽트 시퀀스 번호
let ttsEsCurrent = null;    // 현재 열린 TTS EventSource
let ttsResolve   = null;    // 현재 대기 중인 Promise resolve

// 문장 누적
let sentenceBuf = '';


// ── 유틸 ───────────────────────────────────────────────────────────────────────
function setStatus(dotClass, txt) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot' + (dotClass ? ' ' + dotClass : '');
  document.getElementById('statusTxt').textContent = txt;
}

function scrollBottom() {
  const m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

/** 채팅 버블 추가. bubble DOM 요소를 반환 */
function addMessage(role, text) {
  const wrapper = document.createElement('div');
  wrapper.className = 'msg ' + role;

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role === 'user' ? '나' : 'AI';
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


// ── TTS 큐 ────────────────────────────────────────────────────────────────────

/** 진행 중인 TTS + 오디오를 모두 중단하고 큐 비우기 */
function stopTTS() {
  ttsJobSeq++;                          // 진행 중인 drainTTS 무효화
  ttsQueue   = [];
  sentenceBuf = '';
  isTTSBusy  = false;
  // 열린 EventSource 닫기
  if (ttsEsCurrent) { ttsEsCurrent.close(); ttsEsCurrent = null; }
  // 대기 중인 Promise 해제 (드레인 루프 탈출)
  if (ttsResolve)   { ttsResolve(); ttsResolve = null; }
  // 오디오 즉시 중단: AudioContext 재생성
  if (audioCtx && audioCtx.state !== 'closed') {
    audioCtx.close();
    audioCtx = null;
  }
  ttsNextStart = 0;
}

/** 문장을 TTS 큐에 추가 */
function enqueueTTS(text) {
  text = text.trim();
  if (!text) return;
  ttsQueue.push(text);
  if (!isTTSBusy) {
    isTTSBusy = true;
    drainTTS(ttsJobSeq);
  }
}

/** TTS 큐를 순차 처리 (파이프라인: 합성 완료 즉시 다음 합성 시작, 재생은 겹침) */
async function drainTTS(mySeq) {
  while (ttsQueue.length > 0 && ttsJobSeq === mySeq) {
    const text = ttsQueue.shift();

    // 1. POST /tts/synthesize
    let job_id;
    try {
      const r = await fetch('/tts/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: document.getElementById('ttsVoice').value,
          speed: parseFloat(document.getElementById('ttsSpeed').value),
        }),
      });
      if (ttsJobSeq !== mySeq) return;
      const data = await r.json();
      if (data.error) continue;
      job_id = data.job_id;
    } catch (e) {
      console.warn('TTS synthesize 오류:', e);
      continue;
    }
    if (ttsJobSeq !== mySeq) return;

    // 2. GET /tts/stream/<job_id> - 오디오 청크 수신 및 스케줄
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
        setStatus('speaking', '말하는 중...');
      });
      es.addEventListener('done',  done);
      es.addEventListener('error', done);
      es.onerror = () => { if (ttsJobSeq === mySeq) done(); };
    });

    // 합성 완료 → 즉시 다음 문장 합성 시작 (재생은 ttsNextStart로 자동 순서 보장)
  }

  if (ttsJobSeq !== mySeq) return;

  // 큐 소진: 오디오 재생 완료 후 상태 복귀
  isTTSBusy = false;
  const ctx = audioCtx;
  if (ctx && ttsNextStart > ctx.currentTime) {
    const remainMs = (ttsNextStart - ctx.currentTime) * 1000 + 300;
    setTimeout(() => {
      if (!isTTSBusy) {
        setStatus(isRecording ? 'listening' : '', isRecording ? '듣는 중...' : '대기 중');
      }
    }, remainMs);
  } else {
    setStatus(isRecording ? 'listening' : '', isRecording ? '듣는 중...' : '대기 중');
  }
}


// ── 문장 감지 (LLM 토큰 → TTS 입력) ──────────────────────────────────────────
const SENT_RE = /^(.*?[.!?。！？\n])\s*/s;

function feedToken(token) {
  sentenceBuf += token;
  let m;
  while ((m = SENT_RE.exec(sentenceBuf)) !== null) {
    enqueueTTS(m[1]);
    sentenceBuf = sentenceBuf.slice(m[0].length);
  }
  // 너무 길어지면 공백 기준으로 강제 분리 (단락 등)
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
  // 이전 TTS / LLM 인터럽트
  if (llmController) { llmController.abort(); llmController = null; }
  stopTTS();

  setStatus('thinking', '생각 중...');

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
          // 버블 업데이트
          cursor.remove();
          bubble.textContent = replyText;
          bubble.appendChild(cursor);
          scrollBottom();
          // TTS 문장 누적
          feedToken(evt.text);

        } else if (evt.type === 'done') {
          cursor.remove();
          flushSentenceBuf();
          // TTS가 없으면 바로 상태 복귀
          if (!isTTSBusy) {
            setStatus(isRecording ? 'listening' : '', isRecording ? '듣는 중...' : '대기 중');
          }

        } else if (evt.type === 'error') {
          cursor.remove();
          bubble.textContent = (replyText || '') + '\n[오류: ' + evt.text + ']';
          stopTTS();
        }
      }
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      // 인터럽트에 의한 취소 - 정상
      cursor.remove();
      if (!replyText) bubble.closest('.msg').remove();
    } else {
      cursor.remove();
      bubble.textContent = replyText || '[연결 오류: ' + e.message + ']';
      stopTTS();
    }
  } finally {
    llmController = null;
  }
}


// ── STT / 마이크 ──────────────────────────────────────────────────────────────

/** STT SSE 스트림 연결 (자동 재연결 포함) */
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

/** MediaRecorder 기반 200ms 청크 녹음 (STT 모듈과 동일한 방식) */
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
    if (!isTTSBusy) setStatus('', '대기 중');
  } else {
    if (!micStream) {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch (e) {
        setStatus('', '마이크 오류: ' + e.message);
        return;
      }
    }
    isRecording = true;
    btn.classList.add('recording');
    btn.textContent = '⏹';
    setStatus('listening', '듣는 중...');
    startChunk(micStream);
    chunkInterval = setInterval(() => {
      if (isRecording && recorder && recorder.state !== 'inactive') recorder.stop();
    }, 200);
  }
}


// ── 새 대화 ────────────────────────────────────────────────────────────────────
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
  setStatus(isRecording ? 'listening' : '', isRecording ? '듣는 중...' : '마이크 버튼을 눌러 시작하세요');
}


// ── 모듈 준비 상태 폴링 ────────────────────────────────────────────────────────
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
      setStatus('', '마이크 버튼을 눌러 시작하세요');
    }
    if (!allReady) setTimeout(pollStatus, 2000);
  } catch (e) {
    setTimeout(pollStatus, 3000);
  }
}

// ── 초기화 ─────────────────────────────────────────────────────────────────────
document.getElementById('loadingBanner').className = 'loading-banner show';
pollStatus();
connectSTT();
</script>
</body>
</html>
"""


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean Modular SDS – 통합 서버")
    parser.add_argument("--port", type=int, default=8080, help="UI 서버 포트 (기본: 8080)")
    args = parser.parse_args()

    threading.Thread(target=start_modules, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (cleanup(), sys.exit(0)))

    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
