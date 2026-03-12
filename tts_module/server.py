"""
실시간 한국어 TTS 서버 (Supertonic)
실행: conda run -n supertonic python3 server.py
접속: http://localhost:8082
"""

import base64
import io
import logging
import queue
import re
import struct
import threading
import uuid
import wave

logging.getLogger("werkzeug").setLevel(logging.ERROR)

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request
from supertonic import TTS

# ── 모델 로드 ──────────────────────────────────────────────────────────────────
print("TTS 모델 로딩 중...")
tts = TTS(auto_download=True)
SAMPLE_RATE = tts.sample_rate  # 44100
print(f"TTS 로드 완료  |  sample_rate={SAMPLE_RATE}  |  voices={tts.voice_style_names}")

# ── 설정 ───────────────────────────────────────────────────────────────────────
DEFAULT_VOICE = "M1"
DEFAULT_LANG  = "ko"
DEFAULT_SPEED = 1.05

# ── 문장 분리 ──────────────────────────────────────────────────────────────────
_SENT_RE = re.compile(r'(?<=[.!?。！？\n])\s*')

def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── numpy float32 → WAV bytes (16-bit PCM) ────────────────────────────────────
def to_wav_bytes(wav_np: np.ndarray, sr: int) -> bytes:
    """wav_np: shape (1, T) or (T,), float32 [-1, 1]"""
    samples = wav_np.squeeze()
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── 잡 관리 ────────────────────────────────────────────────────────────────────
_jobs: dict[str, queue.Queue] = {}
_jobs_lock = threading.Lock()

_SENTINEL = None  # 스트림 종료 신호


def synthesis_worker(job_id: str, sentences: list[str], voice: str, lang: str, speed: float):
    q = _jobs[job_id]
    try:
        style = tts.get_voice_style(voice_name=voice)
        for sent in sentences:
            wav, _ = tts.synthesize(sent, voice_style=style, lang=lang, speed=speed)
            wav_bytes = to_wav_bytes(wav, SAMPLE_RATE)
            b64 = base64.b64encode(wav_bytes).decode()
            q.put(("audio", b64))
            print(f"[TTS] {sent[:30]}{'...' if len(sent)>30 else ''}")
    except Exception as e:
        q.put(("error", str(e)))
    finally:
        q.put(_SENTINEL)


# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(force=True)
    text  = (data.get("text") or "").strip()
    voice = data.get("voice", DEFAULT_VOICE)
    lang  = data.get("lang",  DEFAULT_LANG)
    speed = float(data.get("speed", DEFAULT_SPEED))

    if not text:
        return jsonify(error="텍스트 없음"), 400
    if voice not in tts.voice_style_names:
        return jsonify(error=f"알 수 없는 voice: {voice}"), 400

    sentences = split_sentences(text)
    job_id = str(uuid.uuid4())

    with _jobs_lock:
        _jobs[job_id] = queue.Queue()

    threading.Thread(
        target=synthesis_worker,
        args=(job_id, sentences, voice, lang, speed),
        daemon=True,
    ).start()

    return jsonify(job_id=job_id)


@app.route("/stream/<job_id>")
def stream(job_id: str):
    with _jobs_lock:
        q = _jobs.get(job_id)
    if q is None:
        return jsonify(error="잡 없음"), 404

    def generate():
        try:
            while True:
                item = q.get(timeout=60)
                if item is _SENTINEL:
                    yield "event: done\ndata: \n\n"
                    break
                kind, payload = item
                yield f"event: {kind}\ndata: {payload}\n\n"
        except queue.Empty:
            yield "event: error\ndata: timeout\n\n"
        finally:
            with _jobs_lock:
                _jobs.pop(job_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>실시간 한국어 TTS (Supertonic)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
           display: flex; justify-content: center; padding: 40px 16px; }
    .card { background: white; border-radius: 14px; padding: 36px;
            width: 100%; max-width: 680px; box-shadow: 0 4px 20px rgba(0,0,0,.1); }
    h2   { margin-bottom: 24px; font-size: 22px; }
    .row { display: flex; gap: 12px; margin-bottom: 14px; align-items: center; flex-wrap: wrap; }
    label { font-size: 13px; color: #555; }
    select, input[type=range] { font-size: 14px; border: 1px solid #ddd;
            border-radius: 6px; padding: 6px 10px; background: #fafafa; }
    textarea { width: 100%; height: 120px; font-size: 16px; padding: 12px;
               border: 1px solid #ddd; border-radius: 8px; resize: vertical;
               font-family: inherit; margin-bottom: 14px; }
    button { font-size: 15px; padding: 10px 28px; border-radius: 8px;
             cursor: pointer; border: none; transition: opacity .15s; }
    #speakBtn { background: #4CAF50; color: white; }
    #speakBtn:disabled { opacity: .5; cursor: default; }
    #stopBtn  { background: #f44336; color: white; display: none; }
    .status   { margin-top: 16px; font-size: 13px; color: #777; min-height: 18px; }
    .progress { margin-top: 10px; height: 4px; background: #eee; border-radius: 2px; }
    .progress-bar { height: 100%; background: #4CAF50; border-radius: 2px;
                    width: 0%; transition: width .3s; }
    #speedVal { min-width: 28px; font-size: 13px; color: #333; }
  </style>
</head>
<body>
<div class="card">
  <h2>🔊 실시간 한국어 TTS (Supertonic)</h2>

  <textarea id="text" placeholder="여기에 텍스트를 입력하세요. 문장 단위로 즉시 재생됩니다.">안녕하세요. 만나서 반갑습니다. 오늘 날씨가 정말 좋네요.</textarea>

  <div class="row">
    <label>목소리</label>
    <select id="voice">
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

    <label>언어</label>
    <select id="lang">
      <option value="ko" selected>한국어</option>
      <option value="en">English</option>
    </select>

    <label>속도</label>
    <input type="range" id="speed" min="0.7" max="1.5" step="0.05" value="1.05"
           oninput="document.getElementById('speedVal').textContent=this.value">
    <span id="speedVal">1.05</span>
  </div>

  <div class="row">
    <button id="speakBtn" onclick="speak()">▶ 읽기</button>
    <button id="stopBtn"  onclick="stopAll()">■ 중지</button>
  </div>

  <div class="progress"><div class="progress-bar" id="bar"></div></div>
  <div class="status" id="status">텍스트를 입력하고 읽기를 누르세요.</div>
</div>

<script>
let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
let nextStartAt = 0;
let totalSents = 0;
let doneSents  = 0;
let evtSource  = null;
let stopped    = false;

function setStatus(msg)  { document.getElementById("status").textContent = msg; }
function setProgress(v)  { document.getElementById("bar").style.width = (v*100).toFixed(1)+"%"; }
function setSpeaking(on) {
  document.getElementById("speakBtn").disabled = on;
  document.getElementById("stopBtn").style.display = on ? "" : "none";
}

async function speak() {
  const text = document.getElementById("text").value.trim();
  if (!text) { setStatus("텍스트를 입력해주세요."); return; }

  // 이전 재생 정리
  if (evtSource) { evtSource.close(); evtSource = null; }
  stopped = false;

  // 브라우저 autoplay 정책: 클릭 시점에 반드시 resume
  await audioCtx.resume();
  nextStartAt = audioCtx.currentTime + 0.05;

  setSpeaking(true);
  setProgress(0);
  setStatus("⏳ 합성 중...");

  // 1. POST → job_id
  let job;
  try {
    const res = await fetch("/synthesize", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        text,
        voice: document.getElementById("voice").value,
        lang:  document.getElementById("lang").value,
        speed: parseFloat(document.getElementById("speed").value),
      }),
    });
    job = await res.json();
  } catch(e) { setStatus("❌ 서버 연결 실패: " + e.message); setSpeaking(false); return; }

  if (job.error) { setStatus("❌ " + job.error); setSpeaking(false); return; }

  // 문장 수 추정 (진행률용)
  totalSents = (text.match(/[.!?。！？\n]/g) || []).length || 1;
  doneSents  = 0;

  // 2. SSE → 오디오 청크 수신 + 즉시 재생
  evtSource = new EventSource("/stream/" + job.job_id);

  evtSource.addEventListener("audio", async (e) => {
    if (stopped) return;
    try {
      const buf = await audioCtx.decodeAudioData(base64ToArrayBuffer(e.data));
      scheduleBuffer(buf);
      doneSents++;
      setProgress(doneSents / totalSents);
      setStatus(`🔊 재생 중... (${doneSents}/${totalSents} 문장)`);
    } catch(err) { console.warn("decode error", err); }
  });

  evtSource.addEventListener("done", () => {
    evtSource.close(); evtSource = null;
    setProgress(1);
    const remaining = nextStartAt - audioCtx.currentTime;
    setTimeout(() => { if (!stopped) { setSpeaking(false); setStatus("✅ 완료"); } },
               Math.max(0, remaining * 1000));
  });

  evtSource.addEventListener("error", (e) => {
    if (e.data) setStatus("❌ " + e.data);
    evtSource.close(); evtSource = null;
    setSpeaking(false);
  });

  evtSource.onerror = () => {
    if (!stopped) { setStatus("⚠️ 스트림 오류"); setSpeaking(false); }
  };
}

function scheduleBuffer(buf) {
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);
  const start = Math.max(nextStartAt, audioCtx.currentTime);
  src.start(start);
  nextStartAt = start + buf.duration;
}

function stopAll() {
  stopped = true;
  if (evtSource) { evtSource.close(); evtSource = null; }
  // AudioContext 재생성으로 즉시 정지
  audioCtx.close();
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  nextStartAt = 0;
  setSpeaking(false);
  setProgress(0);
  setStatus("중지됨");
}

function base64ToArrayBuffer(b64) {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return buf.buffer;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("접속: http://localhost:8082")
    app.run(host="0.0.0.0", port=8082, debug=False, threaded=True)
