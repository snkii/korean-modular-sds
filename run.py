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

def _start_module(conda_env: str, subdir: str, script: str = "server.py") -> subprocess.Popen:
    proc = subprocess.Popen(
        ["conda", "run", "--no-capture-output", "-n", conda_env, "python", script],
        cwd=os.path.join(ROOT_DIR, subdir),
        preexec_fn=os.setsid,  # 새 프로세스 그룹 생성 → 자식까지 한번에 종료 가능
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


def _module_up(url: str) -> bool:
    try:
        return requests.get(url, timeout=2).status_code < 500
    except Exception:
        return False


def start_modules() -> None:
    print("=" * 52)
    print("  Korean Full-Duplex Audio-LM (HIL) 시작 중...")
    print("=" * 52)
    specs = [
        ("STT", "stt_module", STT_URL, "stt"),
        ("TTS", "tts_module", TTS_URL, "tts"),
        ("LLM", "llm_module", LLM_URL, "llm"),
    ]
    pending = []
    for label, subdir, url, key in specs:
        if _module_up(url):
            # 이미 다른 인스턴스가 이 포트를 서비스 중 → 재생성하지 않는다
            # (중복 spawn·llama-server 재기동으로 인한 포트/GPU 충돌 방지)
            print(f"  ✓ {label} 이미 실행 중 – 건너뜀 ({url})")
            _ready[key] = True
        else:
            print(f"  · {label} 모듈 기동 (conda: korean-modular-sds) ...")
            _start_module("korean-modular-sds", subdir)
            pending.append((label, url, key))

    if pending:
        print("\n모듈 준비 대기 중 (병렬)...")
        threads = [
            threading.Thread(target=_wait_ready, args=(url, label, key), daemon=True)
            for label, url, key in pending
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print("\n" + "=" * 52)
    print("  모든 모듈 준비 완료!")
    print("=" * 52)


def cleanup() -> None:
    for p in _procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    for p in _procs:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
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
<title>한국형 전이중 음성 대화 에이전트 · HIL</title>
<style>
:root {
  --bg-0:#05060c; --bg-1:#0a0d1a; --panel:rgba(255,255,255,.045);
  --panel-brd:rgba(255,255,255,.09); --ink:#eef1fb; --ink-soft:#aab3d0;
  --ink-mute:#6b7495; --field:rgba(255,255,255,.05); --field-brd:rgba(255,255,255,.12);
  --accent:#7c5cff; --accent-2:#22d3ee;
  /* 상태색 (body[data-state] 로 전환) */
  --state:#7c5cff; --state-soft:rgba(124,92,255,.55);
}
* { box-sizing:border-box; margin:0; padding:0; }
html,body { height:100%; }
body {
  font-family:'Pretendard','Apple SD Gothic Neo','Segoe UI',system-ui,sans-serif;
  color:var(--ink);
  background:var(--bg-0);
  display:flex; height:100vh; overflow:hidden;
  position:relative;
}
/* 상태별 강조색 */
body[data-state="idle"]      { --state:#7c5cff; --state-soft:rgba(124,92,255,.55); }
body[data-state="listening"] { --state:#34e5b0; --state-soft:rgba(52,229,176,.55); }
body[data-state="thinking"]  { --state:#4cc4ff; --state-soft:rgba(76,196,255,.55); }
body[data-state="speaking"]  { --state:#ffb454; --state-soft:rgba(255,180,84,.55); }

/* ── 배경 오로라 ── */
.aurora { position:fixed; inset:-20% -10% -10% -10%; z-index:0; pointer-events:none; overflow:hidden;
          background:radial-gradient(60% 50% at 50% 0%, #12173080 0%, transparent 70%); }
.aurora::before, .aurora::after {
  content:""; position:absolute; width:60vw; height:60vw; border-radius:50%;
  filter:blur(90px); opacity:.5; mix-blend-mode:screen;
}
.aurora::before { background:radial-gradient(circle,#5b3cff 0%,transparent 60%); top:-18vw; left:-8vw;
                  animation:float1 22s ease-in-out infinite; }
.aurora::after  { background:radial-gradient(circle,#12a7c9 0%,transparent 60%); bottom:-22vw; right:-6vw;
                  animation:float2 26s ease-in-out infinite; }
@keyframes float1 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(6vw,4vw) scale(1.12)} }
@keyframes float2 { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-5vw,-3vw) scale(1.08)} }
.grain { position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.5;
  background:
    radial-gradient(1px 1px at 20% 30%, #ffffff14, transparent),
    radial-gradient(1px 1px at 70% 60%, #ffffff10, transparent),
    radial-gradient(1px 1px at 40% 80%, #ffffff0d, transparent); }

/* ── 사이드바 ── */
.sidebar {
  position:relative; z-index:2;
  width:264px; flex-shrink:0;
  display:flex; flex-direction:column; gap:16px;
  padding:20px 16px;
  background:linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
  border-right:1px solid var(--panel-brd);
  backdrop-filter:blur(18px); -webkit-backdrop-filter:blur(18px);
  overflow-y:auto;
}
.brand-mini { display:flex; align-items:center; gap:11px; padding:2px 4px 10px;
              border-bottom:1px solid var(--panel-brd); }
.brand-logo { width:38px; height:38px; border-radius:11px; flex-shrink:0;
  display:flex; align-items:center; justify-content:center; font-weight:800; font-size:15px; letter-spacing:.02em;
  color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 6px 18px rgba(124,92,255,.45); }
.brand-mini .b-lab { font-size:12.5px; font-weight:700; color:var(--ink); line-height:1.25; }
.brand-mini .b-sub { font-size:10.5px; color:var(--ink-mute); margin-top:2px; }

.sidebar h3 { font-size:10.5px; color:var(--ink-mute); text-transform:uppercase;
              letter-spacing:.14em; margin-bottom:-4px; }
.field { display:flex; flex-direction:column; gap:6px; }
.sidebar label { font-size:12px; color:var(--ink-soft); display:flex; justify-content:space-between; align-items:center; }
.sidebar select, .sidebar textarea {
  width:100%; background:var(--field); border:1px solid var(--field-brd);
  color:var(--ink); border-radius:10px; padding:9px 10px; font-size:12.5px; font-family:inherit;
  transition:border-color .2s, box-shadow .2s;
}
.sidebar select:focus, .sidebar textarea:focus {
  outline:none; border-color:var(--state-soft); box-shadow:0 0 0 3px var(--state-soft); }
.sidebar textarea { resize:vertical; min-height:96px; line-height:1.55; }
.range-row { display:flex; align-items:center; }
input[type=range]{ -webkit-appearance:none; appearance:none; width:100%; height:5px; border-radius:99px;
  background:linear-gradient(90deg,var(--accent),var(--accent-2)); }
input[type=range]::-webkit-slider-thumb{ -webkit-appearance:none; width:16px; height:16px; border-radius:50%;
  background:#fff; border:3px solid var(--state); box-shadow:0 2px 8px rgba(0,0,0,.5); cursor:pointer; }
.val-lbl { font-size:11px; color:var(--ink); background:var(--field); border:1px solid var(--field-brd);
  padding:1px 8px; border-radius:99px; min-width:44px; text-align:center; }
.new-chat-btn {
  margin-top:auto; border:none; cursor:pointer; border-radius:12px; padding:12px;
  font-size:13px; font-weight:700; color:#0a0d1a; letter-spacing:.01em;
  background:linear-gradient(135deg,var(--accent),var(--accent-2));
  box-shadow:0 8px 22px rgba(124,92,255,.35); transition:transform .12s, filter .2s; }
.new-chat-btn:hover { filter:brightness(1.08); transform:translateY(-1px); }

/* ── 채팅 영역 ── */
.chat-wrap { position:relative; z-index:2; flex:1; display:flex; flex-direction:column; overflow:hidden; }

.chat-header {
  padding:16px 24px; display:flex; align-items:center; gap:16px;
  border-bottom:1px solid var(--panel-brd);
  background:linear-gradient(180deg, rgba(255,255,255,.04), transparent);
  backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
}
.title-block { display:flex; flex-direction:column; gap:3px; min-width:0; }
.title-ko { font-size:16.5px; font-weight:750; letter-spacing:-.01em;
  background:linear-gradient(90deg,#fff, #cbd3ff); -webkit-background-clip:text; background-clip:text; color:transparent; }
.title-en { font-size:11px; color:var(--ink-mute); letter-spacing:.06em; text-transform:uppercase; }
.title-en b { color:var(--ink-soft); font-weight:600; }
.header-status { margin-left:auto; display:flex; align-items:center; gap:9px;
  padding:7px 14px; border-radius:99px; background:var(--field); border:1px solid var(--field-brd); }
.status-dot { width:9px; height:9px; border-radius:50%; background:var(--ink-mute); flex-shrink:0;
  box-shadow:0 0 0 0 transparent; transition:background .3s, box-shadow .3s; }
.status-dot.listening { background:#34e5b0; box-shadow:0 0 10px 2px rgba(52,229,176,.7); animation:pulse 1.2s infinite; }
.status-dot.thinking  { background:#4cc4ff; box-shadow:0 0 10px 2px rgba(76,196,255,.7); animation:pulse .8s infinite; }
.status-dot.speaking  { background:#ffb454; box-shadow:0 0 10px 2px rgba(255,180,84,.7); animation:pulse .6s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.status-txt { font-size:12px; color:var(--ink-soft); white-space:nowrap; }

/* ── 메시지 ── */
.messages { flex:1; overflow-y:auto; padding:26px 24px 8px;
  display:flex; flex-direction:column; gap:16px; scroll-behavior:smooth; }
.messages::-webkit-scrollbar { width:8px; }
.messages::-webkit-scrollbar-thumb { background:rgba(255,255,255,.12); border-radius:99px; }

.msg { display:flex; flex-direction:column; max-width:78%; animation:rise .28s ease both; }
.msg.user { align-self:flex-end; align-items:flex-end; }
.msg.assistant { align-self:flex-start; align-items:flex-start; }
@keyframes rise { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
.role-label { font-size:10.5px; color:var(--ink-mute); margin-bottom:5px; padding:0 6px;
  letter-spacing:.08em; text-transform:uppercase; }
.bubble { padding:12px 16px; border-radius:18px; font-size:15px; line-height:1.7;
  white-space:pre-wrap; word-break:break-word; }
.msg.user .bubble { color:#fff;
  background:linear-gradient(135deg,var(--accent),#5b3cff);
  border-bottom-right-radius:6px; box-shadow:0 8px 24px rgba(91,60,255,.32); }
.msg.assistant .bubble { color:var(--ink);
  background:rgba(255,255,255,.055); border:1px solid var(--panel-brd);
  border-bottom-left-radius:6px; backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
.cursor { display:inline-block; width:2px; height:1em; background:var(--state);
  animation:blink .7s infinite; vertical-align:text-bottom; margin-left:2px; border-radius:2px; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* ── 로딩 배너 ── */
.loading-banner { display:none; align-items:center; justify-content:center; gap:10px;
  font-size:12.5px; font-weight:600; color:var(--ink-soft); padding:9px 18px;
  background:rgba(255,180,84,.08); border-bottom:1px solid var(--panel-brd); }
.loading-banner.show { display:flex; }
.module-pills { display:flex; gap:7px; margin-left:6px; }
.pill { font-size:10.5px; font-weight:700; letter-spacing:.05em; padding:3px 11px; border-radius:99px;
  color:var(--ink-mute); background:var(--field); border:1px solid var(--field-brd); transition:all .3s; }
.pill.ready { color:#0a0d1a; background:linear-gradient(135deg,#34e5b0,#22d3ee); border-color:transparent; }

/* ── 하단 스테이지 (오브) ── */
.stage { padding:14px 18px 30px; display:flex; flex-direction:column; align-items:center; gap:14px;
  background:linear-gradient(0deg, rgba(255,255,255,.035), transparent); }
.partial-area { width:100%; max-width:640px; min-height:22px; text-align:center;
  font-size:13.5px; color:var(--ink-soft); font-style:normal; }
.partial-area:empty::before { content:""; }

.orb-wrap { position:relative; width:132px; height:132px; display:flex; align-items:center; justify-content:center; }
/* 회전 링 */
.orb-wrap::before { content:""; position:absolute; inset:-6px; border-radius:50%;
  background:conic-gradient(from 0deg, var(--state), transparent 30%, var(--state) 60%, transparent 92%);
  opacity:.55; filter:blur(2px); animation:spin 6s linear infinite; }
@keyframes spin { to{ transform:rotate(360deg); } }
/* 대기 시 은은한 확장 */
.orb-wrap::after { content:""; position:absolute; width:96px; height:96px; border-radius:50%;
  border:1px solid var(--state-soft); animation:breathe 3.4s ease-in-out infinite; }
@keyframes breathe { 0%,100%{ transform:scale(1); opacity:.4 } 50%{ transform:scale(1.28); opacity:0 } }

.mic-btn { position:relative; z-index:1; width:96px; height:96px; border-radius:50%; border:none;
  cursor:pointer; user-select:none; font-size:34px; color:#fff;
  display:flex; align-items:center; justify-content:center;
  background:radial-gradient(120% 120% at 30% 25%, #ffffff30, transparent 40%),
             radial-gradient(circle at 50% 50%, var(--state), #1a1030 90%);
  box-shadow:0 0 0 1px rgba(255,255,255,.12) inset, 0 14px 40px var(--state-soft);
  transition:transform .16s, box-shadow .3s; }
.mic-btn:hover { transform:scale(1.05); }
.mic-btn.recording { animation:micPulse 1.1s ease-in-out infinite; }
@keyframes micPulse { 0%,100%{ box-shadow:0 0 0 1px rgba(255,255,255,.15) inset, 0 0 0 0 var(--state-soft); }
  50%{ box-shadow:0 0 0 1px rgba(255,255,255,.15) inset, 0 0 0 18px transparent; } }
/* 녹음 중 링 강조 */
body[data-rec="1"] .orb-wrap::before { animation-duration:2.4s; opacity:.85; }
.hint { font-size:11.5px; color:var(--ink-mute); letter-spacing:.02em; }

/* 반응형 */
@media (max-width:760px){ .sidebar{ display:none; } .title-en{ display:none; } }
</style>
</head>
<body data-state="idle" data-rec="0">
<div class="aurora"></div>
<div class="grain"></div>

<!-- 사이드바 -->
<div class="sidebar">
  <div class="brand-mini">
    <div class="brand-logo">HIL</div>
    <div>
      <div class="b-lab">서울대학교<br>휴먼인터페이스연구실</div>
      <div class="b-sub">Human Interface Laboratory</div>
    </div>
  </div>

  <h3>대화 설정</h3>

  <div class="field">
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

  <div class="field">
    <label>말하기 속도 <span class="val-lbl" id="speedLbl">1.05</span></label>
    <div class="range-row">
      <input type="range" id="ttsSpeed" min="0.7" max="1.5" step="0.05" value="1.05"
             oninput="document.getElementById('speedLbl').textContent=parseFloat(this.value).toFixed(2)">
    </div>
  </div>

  <div class="field">
    <label>답변 다양성 <span class="val-lbl" id="tempLbl">0.70</span></label>
    <div class="range-row">
      <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7"
             oninput="document.getElementById('tempLbl').textContent=parseFloat(this.value).toFixed(2)">
    </div>
  </div>

  <div class="field">
    <label>시스템 프롬프트</label>
    <textarea id="systemPrompt">당신은 서울대학교 휴먼인터페이스연구실(HIL)이 개발한 한국형 전이중 음성 대화 에이전트입니다. 사용자와 실시간으로 자연스럽게 대화하며, 항상 정중하고 친근한 한국어 구어체로 한두 문장 이내로 짧고 명확하게 대답합니다. 소리 내어 읽어 주는 음성 답변이므로 괄호나 특수문자, 이모지, 줄바꿈, 목록 기호는 절대 쓰지 않습니다. 모르는 내용은 아는 척하지 않고 솔직하게 말합니다.</textarea>
  </div>

  <button class="new-chat-btn" onclick="newChat()">＋ 새 대화 시작</button>
</div>

<!-- 채팅 -->
<div class="chat-wrap">
  <div class="chat-header">
    <div class="title-block">
      <div class="title-ko">한국형 전이중 음성 대화 에이전트</div>
      <div class="title-en">Korean <b>Full-Duplex</b> Audio-Language Model · <b>HIL</b> @ Seoul National University</div>
    </div>
    <div class="header-status">
      <span class="status-dot" id="statusDot"></span>
      <span class="status-txt" id="statusTxt">마이크 버튼을 눌러 시작하세요</span>
    </div>
  </div>

  <div class="loading-banner" id="loadingBanner">
    ⏳ 음성 엔진 준비 중
    <div class="module-pills">
      <span class="pill" id="pillStt">음성인식</span>
      <span class="pill" id="pillTts">음성합성</span>
      <span class="pill" id="pillLlm">언어모델</span>
    </div>
  </div>
  <div class="messages" id="messages"></div>

  <div class="stage">
    <div class="partial-area" id="partialArea"></div>
    <div class="orb-wrap">
      <button class="mic-btn" id="micBtn" onclick="toggleMic()" title="마이크 켜기/끄기">🎤</button>
    </div>
    <div class="hint">버튼을 누르고 편하게 말씀하세요 · 언제든 끼어들 수 있어요</div>
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
let audioCtx      = null;
let ttsNextStart  = 0;       // Web Audio 스케줄 타임
let ttsQueue      = [];      // 합성 대기 중인 문장 목록
let isTTSBusy     = false;   // drainTTS 실행 중 여부
let ttsJobSeq     = 0;       // 인터럽트 시퀀스 번호
let ttsEsCurrent  = null;    // 현재 열린 TTS EventSource
let ttsResolve    = null;    // 현재 대기 중인 Promise resolve
let _activeSrcs   = [];      // 재생 중인 AudioBufferSourceNode 목록 (중단용)

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

async function scheduleWav(arrayBuffer) {
  const ctx = getAudioCtx();
  await ctx.resume();
  const buf = await ctx.decodeAudioData(arrayBuffer);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  const start = Math.max(ttsNextStart, ctx.currentTime + 0.02);
  src.start(start);
  ttsNextStart = start + buf.duration;
  _activeSrcs.push(src);
  src.onended = () => { _activeSrcs = _activeSrcs.filter(s => s !== src); };
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
  // 재생 중인 소스 노드 즉시 중단 후 suspend (AudioContext는 재사용하여 autoplay unlock 유지)
  for (const s of _activeSrcs) { try { s.stop(); } catch (_) {} }
  _activeSrcs = [];
  if (audioCtx && audioCtx.state !== 'closed') audioCtx.suspend();
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
      let pending = 0;
      let streamDone = false;

      function done() {
        streamDone = true;
        if (ttsEsCurrent === es) ttsEsCurrent = null;
        if (ttsResolve === resolve) ttsResolve = null;
        es.close();
        if (pending === 0) resolve();
      }

      es.addEventListener('audio', async e => {
        if (ttsJobSeq !== mySeq) { done(); return; }
        pending++;
        try { await scheduleWav(base64ToArrayBuffer(e.data)); } catch (_) {}
        pending--;
        if (ttsJobSeq === mySeq) setStatus('speaking', '말하는 중...');
        if (streamDone && pending === 0) resolve();
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

  sttEs.addEventListener('speech_start', () => {
    setStatus('listening', '듣는 중...');
  });

  sttEs.addEventListener('partial', e => {
    document.getElementById('partialArea').textContent = e.data ? '⏳ ' + e.data : '';
    // ASR hypothesis가 생성되는 즉시 TTS/LLM 인터럽트 (VAD 오탐 방지: 실제 인식 텍스트 확인 후 중단)
    if (e.data) {
      if (llmController) { llmController.abort(); llmController = null; }
      stopTTS();
    }
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
    // await 전에 호출해야 user gesture context 안에서 AudioContext unlock이 보장됨
    getAudioCtx();
    audioCtx.resume().catch(() => {});
    if (!micStream) {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
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

<script>
// ── 중앙 오브 상태 연출 (비침습적 확장) ────────────────────────────────────────
// 기존 로직이 갱신하는 #statusDot / #micBtn 의 class 를 관찰해
// body[data-state], body[data-rec] 로 미러링한다. CSS 가 이 값으로 오브 색·링을 연출한다.
(function () {
  const dot = document.getElementById('statusDot');
  const mic = document.getElementById('micBtn');
  function sync() {
    const s = dot.classList.contains('listening') ? 'listening'
            : dot.classList.contains('thinking')  ? 'thinking'
            : dot.classList.contains('speaking')  ? 'speaking' : 'idle';
    document.body.dataset.state = s;
    document.body.dataset.rec = mic.classList.contains('recording') ? '1' : '0';
  }
  new MutationObserver(sync).observe(dot, { attributes: true, attributeFilter: ['class'] });
  new MutationObserver(sync).observe(mic, { attributes: true, attributeFilter: ['class'] });
  sync();
})();
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
