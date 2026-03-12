"""
실시간 한국어 음성 전사 서버 (Korean Streaming ASR: Denoiser + Conformer-CTC + GPU)
실행: python3 server_nemo.py
접속: http://localhost:8081
"""

import argparse
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading

logging.getLogger("werkzeug").setLevel(logging.ERROR)

import torch
import numpy as np
import librosa
from flask import Flask, Response, request, jsonify, render_template_string

# ── Korean Streaming ASR 경로 추가 ───────────────────────────────────────────
REPO_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Korean-Streaming-ASR")
DENOISER_DIR = os.path.join(REPO_DIR, "src", "denoiser")
CKPT_DIR     = os.path.join(REPO_DIR, "checkpoint")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
sys.path.insert(0, DENOISER_DIR)

from denoiser import pretrained as denoiser_pretrained
import nemo.collections.asr as nemo_asr

# ── 설정 ─────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DENOISER_PTH = os.path.join(CKPT_DIR, "denoiser.th")
ASR_NEMO     = os.path.join(CKPT_DIR, "Conformer-CTC-BPE.nemo")

SILENCE_RMS_THRESH = 0.005
SILENCE_CHUNKS_END = 2      # 연속 묵음 청크 수 → 발화 종료
DENOISE_DRY        = 0.05

print(f"사용 장치: {DEVICE}")

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
print("Denoiser 로딩 중...")
_denoiser_args = argparse.Namespace(model_path=DENOISER_PTH, dns48=False, dns64=False,
                                     master64=False, valentini_nc=False)
denoiser_model = denoiser_pretrained.get_model(_denoiser_args).to(DEVICE)
denoiser_model.eval()
print("Denoiser 로드 완료")

print("Conformer-CTC ASR 모델 로딩 중...")
asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(ASR_NEMO, map_location=DEVICE)
asr_model.eval()
if DEVICE == "cuda":
    asr_model.cuda()
print("ASR 모델 로드 완료")

SAMPLE_RATE = asr_model.preprocessor._cfg["sample_rate"]
BLANK_ID    = len(asr_model.decoder.vocabulary)


@torch.no_grad()
def transcribe_buffer(buf: np.ndarray) -> str:
    """누적 버퍼 전체를 denoiser → CTC greedy decode."""
    audio = torch.from_numpy(buf).unsqueeze(0).to(DEVICE)
    audio_len = torch.tensor([buf.shape[0]], device=DEVICE)
    estimate = denoiser_model(audio)
    audio = ((1 - DENOISE_DRY) * estimate + DENOISE_DRY * audio).squeeze(1)
    _, _, predictions = asr_model(input_signal=audio, input_signal_length=audio_len)
    pred = predictions[0].cpu().numpy()
    out, prev = [], BLANK_ID
    for p in pred:
        if (p != prev or prev == BLANK_ID) and p != BLANK_ID:
            out.append(int(p))
        prev = p
    return asr_model.tokenizer.ids_to_text(out)


# ── 상태 ─────────────────────────────────────────────────────────────────────
_lock          = threading.Lock()
_audio_chunks  : list[np.ndarray] = []
_in_speech     = False
_silence_count = 0
_last_text     = ""

# ── Flask + SSE ───────────────────────────────────────────────────────────────
app = Flask(__name__)
_sse_queues: list[queue.Queue] = []
_sse_lock = threading.Lock()


def broadcast(text: str, kind: str = "partial"):
    msg = f"event: {kind}\ndata: {text}\n\n"
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>실시간 한국어 전사 (Conformer)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
           display: flex; justify-content: center; padding: 40px 16px; }
    .card { background: white; border-radius: 14px; padding: 36px;
            width: 100%; max-width: 700px; box-shadow: 0 4px 20px rgba(0,0,0,.1); }
    h2   { margin-bottom: 24px; font-size: 22px; }
    .controls { display: flex; gap: 12px; margin-bottom: 20px; align-items: center; }
    button { font-size: 15px; padding: 10px 24px; border-radius: 8px;
             cursor: pointer; border: none; transition: opacity .15s; }
    #stopBtn    { background: #f44336; color: white; }
    #restartBtn { background: #2196F3; color: white; display: none; }
    #indicator  { width: 14px; height: 14px; border-radius: 50%; background: #ccc; flex-shrink: 0; }
    #indicator.active { background: #f44336; animation: pulse 1s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
    #transcript { min-height: 200px; max-height: 420px; overflow-y: auto;
                  border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px;
                  font-size: 17px; line-height: 1.7; color: #222;
                  white-space: pre-wrap; word-break: break-all; }
    .partial { color: #aaa; font-style: italic; }
    .status    { margin-top: 12px; font-size: 13px; color: #777; min-height: 18px; }
    .clear-btn { margin-left: auto; background: #eee; color: #555;
                 font-size: 13px; padding: 6px 14px; }
  </style>
</head>
<body>
<div class="card">
  <h2>🎙️ 실시간 한국어 음성 전사 (Conformer-CTC)</h2>
  <div class="controls">
    <button id="stopBtn"    onclick="stopRecording()">중지</button>
    <button id="restartBtn" onclick="startRecording()">재시작</button>
    <span id="indicator"></span>
    <button class="clear-btn" onclick="clearTranscript()">지우기</button>
  </div>
  <div id="transcript"></div>
  <div class="status" id="status">마이크 권한 요청 중...</div>
</div>

<script>
  let recorder = null, chunkInterval = null, micStream = null, isRecording = false;

  const evtSrc = new EventSource("/stream");
  evtSrc.onerror = () => setStatus("⚠️ SSE 연결 끊김");

  async function startRecording() {
    if (!micStream) {
      try { micStream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
      catch(e) { setStatus("❌ 마이크 오류: " + e.message); return; }
    }
    isRecording = true;
    document.getElementById("stopBtn").style.display    = "";
    document.getElementById("restartBtn").style.display = "none";
    document.getElementById("indicator").className = "active";
    setStatus("🎤 듣는 중...");
    startChunk(micStream);
    chunkInterval = setInterval(() => {
      if (isRecording && recorder && recorder.state !== "inactive") recorder.stop();
    }, 200);
  }

  function startChunk(stream) {
    recorder = new MediaRecorder(stream);
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: recorder.mimeType });
      if (blob.size >= 500) {
        const form = new FormData();
        form.append("audio", blob, "chunk.webm");
        fetch("/transcribe", { method: "POST", body: form }).catch(() => {});
      }
      if (isRecording) startChunk(stream);
    };
    recorder.start();
  }

  function stopRecording() {
    isRecording = false;
    clearInterval(chunkInterval);
    if (recorder && recorder.state !== "inactive") recorder.stop();
    document.getElementById("stopBtn").style.display    = "none";
    document.getElementById("restartBtn").style.display = "";
    document.getElementById("indicator").className = "";
    setStatus("중지됨");
  }

  window.addEventListener("load", startRecording);

  let liveSpan = null;

  evtSrc.addEventListener("partial", e => { if (e.data) updateLive(e.data); });
  evtSrc.addEventListener("commit",  e => { if (e.data) commitText(e.data); });

  function updateLive(text) {
    const div = document.getElementById("transcript");
    if (!liveSpan) {
      liveSpan = document.createElement("span");
      liveSpan.className = "partial";
      div.appendChild(liveSpan);
    }
    liveSpan.textContent = text;
    div.scrollTop = div.scrollHeight;
    setStatus("⏳ 인식 중...");
  }

  function commitText(text) {
    const div = document.getElementById("transcript");
    if (liveSpan) { liveSpan.remove(); liveSpan = null; }
    div.appendChild(document.createTextNode(text + " "));
    div.scrollTop = div.scrollHeight;
    setStatus("✅ 전사 완료");
  }

  function clearTranscript() {
    document.getElementById("transcript").innerHTML = "";
    liveSpan = null;
  }
  function setStatus(msg) { document.getElementById("status").textContent = msg; }
</script>
</body>
</html>
"""


@app.route("/stream")
def stream():
    q = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_queues.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    global _audio_chunks, _in_speech, _silence_count, _last_text

    if "audio" not in request.files:
        return jsonify(error="audio 없음"), 400
    raw = request.files["audio"].read()
    if not raw:
        return jsonify(error="빈 데이터"), 400

    # webm → 16kHz mono wav
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    wav_path = tmp_path.replace(".webm", ".wav")
    try:
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True
        )
        if ret.returncode != 0:
            return jsonify(error="ffmpeg 실패"), 500
        audio_np, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    finally:
        for p in (tmp_path, wav_path):
            if os.path.exists(p):
                os.remove(p)

    if len(audio_np) < 800:
        return jsonify(message="too short"), 200

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    is_speech = rms > SILENCE_RMS_THRESH

    with _lock:
        if is_speech:
            if not _in_speech:
                # 새 발화 시작
                _audio_chunks = []
                _last_text = ""
                _in_speech = True
            _silence_count = 0
            _audio_chunks.append(audio_np)
            full_audio = np.concatenate(_audio_chunks)
        else:
            if not _in_speech:
                return jsonify(message="silence"), 200
            _silence_count += 1
            if _silence_count < SILENCE_CHUNKS_END:
                return jsonify(message="buffering"), 200
            # 발화 종료
            _in_speech = False
            _silence_count = 0
            final = _last_text
            _audio_chunks = []
            _last_text = ""
            if final:
                broadcast(final, kind="commit")
                print(f"[확정] {final}")
            return jsonify(message=final)

    # 음성 청크 전사
    text = transcribe_buffer(full_audio)
    text = text.strip() if text else ""
    print(f"[전사] {text}")

    with _lock:
        if text and text != _last_text:
            _last_text = text
            broadcast(text, kind="partial")

    return jsonify(message=text)


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print(f"접속: http://localhost:8081")
    app.run(host="0.0.0.0", port=8081, debug=False, threaded=True)
