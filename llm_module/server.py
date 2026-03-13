"""
Qwen3.5-35B-A3B 챗봇 서버
llama-server(llama.cpp) subprocess + OpenAI API 방식
논-띵킹 모드 기본

실행: python3 server.py [--port 8083] [--ctx 8192]
접속: http://localhost:8083
"""

import argparse
import atexit
import json
import logging
import re
import subprocess
import threading
import time

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, Response, jsonify, render_template_string, request
from openai import OpenAI

LLAMA_SERVER_BIN = (
    "/home/sukim/PROJECT_2026/LLAMA_CPP/upstream/build/bin/llama-server"
)
MODEL_PATH = (
    "/home/sukim/.cache/huggingface/hub/"
    "models--unsloth--Qwen3.5-35B-A3B-GGUF/blobs/"
    "ee93ceffed5ce4df8b09bcbaf59a286d531025a1ebde9cf204c74e800c47d57e"
)
LLAMA_PORT = 8180

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ctx",  type=int, default=8192, help="컨텍스트 길이")
parser.add_argument("--port", type=int, default=8083)
args = parser.parse_args()

# ── llama-server subprocess 시작 ───────────────────────────────────────────────
cmd = [
    LLAMA_SERVER_BIN,
    "--model",         MODEL_PATH,
    "--n-gpu-layers",  "-1",
    "--tensor-split",  "1,1,1,1,1,1,1,1",
    "--ctx-size",      str(args.ctx),
    "--batch-size",    "512",
    "--port",          str(LLAMA_PORT),
    "--host",          "127.0.0.1",
    "--jinja",                        # jinja chat template (enable_thinking 지원)
    "--no-mmap",
]
_logfile = open("/tmp/llama-server.log", "w")
print(f"llama-server 시작 중 (port {LLAMA_PORT})...")
print(f"로그: /tmp/llama-server.log")
_proc = subprocess.Popen(cmd, stdout=_logfile, stderr=_logfile)
atexit.register(_proc.terminate)

# 서버 준비 대기
_client = OpenAI(base_url=f"http://127.0.0.1:{LLAMA_PORT}/v1", api_key="none")
for _ in range(60):
    try:
        _client.models.list()
        break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("llama-server 시작 실패")
print("llama-server 준비 완료")

# ── 세션 관리 ──────────────────────────────────────────────────────────────────
DEFAULT_SYSTEM = "You are a helpful, friendly AI assistant. Please respond in the same language as the user."

_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()

def get_history(session_id: str) -> list[dict]:
    with _sessions_lock:
        return _sessions.setdefault(session_id, [])

def reset_history(session_id: str):
    with _sessions_lock:
        _sessions[session_id] = []

def strip_thinking(text: str) -> str:
    """<think>...</think> 블록 제거"""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).lstrip()

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    data          = request.get_json(force=True)
    message       = (data.get("message") or "").strip()
    session_id    = data.get("session_id", "default")
    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM)
    temperature   = float(data.get("temperature", 0.7))
    max_tokens    = int(data.get("max_tokens", 2048))

    if not message:
        return jsonify(error="메시지 없음"), 400

    history = get_history(session_id)
    history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": system_prompt}] + history

    def generate():
        reply_buf = ""
        in_think  = False
        try:
            stream = _client.chat.completions.create(
                model="local",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                extra_body={
                    "top_k": 20,
                    "repeat_penalty": 1.0,
                    "presence_penalty": 1.5,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                stream=True,
            )
            for chunk in stream:
                token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not token:
                    continue

                # <think> 태그 실시간 필터링 (혹시 모를 누락 대비)
                reply_buf += token
                if "<think>" in reply_buf:
                    in_think = True
                if in_think:
                    if "</think>" in reply_buf:
                        reply_buf = re.sub(r"<think>.*?</think>\s*", "", reply_buf, flags=re.DOTALL)
                        in_think = False
                    continue

                yield f"data: {json.dumps({'type': 'reply', 'text': token})}\n\n"

            # 스트림 종료 후 남은 버퍼 처리
            if reply_buf and not in_think:
                pass  # 이미 토큰 단위로 전송됨
            elif in_think:
                # 닫히지 않은 think 블록 제거 후 전송
                clean = re.sub(r"<think>.*", "", reply_buf, flags=re.DOTALL).strip()
                if clean:
                    yield f"data: {json.dumps({'type': 'reply', 'text': clean})}\n\n"

            # 히스토리에는 thinking 제거된 텍스트 저장
            final = strip_thinking(reply_buf).strip()
            history.append({"role": "assistant", "content": final})
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True) or {}
    reset_history(data.get("session_id", "default"))
    return jsonify(ok=True)


# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Qwen3.5-35B-A3B 챗봇</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
           display: flex; height: 100vh; overflow: hidden; }

    .sidebar { width: 260px; background: #1e1e2e; color: #cdd6f4; display: flex;
               flex-direction: column; padding: 20px 16px; gap: 16px; flex-shrink: 0; }
    .sidebar h3 { font-size: 13px; color: #6c7086; text-transform: uppercase;
                  letter-spacing: .08em; }
    .sidebar label { font-size: 12px; color: #a6adc8; margin-bottom: 4px; display: block; }
    .sidebar textarea { width: 100%; background: #313244; border: 1px solid #45475a;
                        color: #cdd6f4; border-radius: 6px; padding: 8px; font-size: 12px;
                        resize: vertical; min-height: 80px; }
    .sidebar input[type=range] { width: 100%; }
    .temp-val { font-size: 12px; color: #a6adc8; }
    .new-chat-btn { background: #89b4fa; color: #1e1e2e; border: none; border-radius: 8px;
                    padding: 10px; font-size: 14px; font-weight: 600; cursor: pointer;
                    margin-top: auto; }

    .chat-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    .chat-header { padding: 16px 24px; background: white; border-bottom: 1px solid #e0e0e0;
                   font-size: 16px; font-weight: 600; color: #333; }
    .messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex;
                flex-direction: column; gap: 16px; }
    .msg { display: flex; flex-direction: column; max-width: 75%; }
    .msg.user      { align-self: flex-end;  align-items: flex-end; }
    .msg.assistant { align-self: flex-start; align-items: flex-start; }
    .bubble { padding: 12px 16px; border-radius: 16px; font-size: 15px;
              line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
    .msg.user .bubble      { background: #89b4fa; color: #1e1e2e; border-bottom-right-radius: 4px; }
    .msg.assistant .bubble { background: white; color: #333; border: 1px solid #e0e0e0;
                              border-bottom-left-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .cursor { display: inline-block; width: 2px; height: 1em; background: #89b4fa;
              animation: blink .7s infinite; vertical-align: text-bottom; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    .input-area { padding: 16px 24px; background: white; border-top: 1px solid #e0e0e0;
                  display: flex; gap: 12px; align-items: flex-end; }
    .input-area textarea { flex: 1; border: 1px solid #ddd; border-radius: 12px; padding: 12px 16px;
                           font-size: 15px; resize: none; max-height: 160px; outline: none;
                           font-family: inherit; line-height: 1.5; }
    .input-area textarea:focus { border-color: #89b4fa; }
    .send-btn { background: #89b4fa; color: #1e1e2e; border: none; border-radius: 12px;
                padding: 12px 20px; font-size: 15px; font-weight: 600; cursor: pointer; }
    .send-btn:disabled { opacity: .4; cursor: default; }
  </style>
</head>
<body>

<div class="sidebar">
  <h3>설정</h3>
  <div>
    <label>시스템 프롬프트</label>
    <textarea id="systemPrompt">You are a helpful, friendly AI assistant. Please respond in the same language as the user.</textarea>
  </div>
  <div>
    <label>Temperature <span class="temp-val" id="tempVal">0.70</span></label>
    <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7"
           oninput="document.getElementById('tempVal').textContent=parseFloat(this.value).toFixed(2)">
  </div>
  <button class="new-chat-btn" onclick="newChat()">＋ 새 대화</button>
</div>

<div class="chat-wrap">
  <div class="chat-header">Qwen3.5-35B-A3B (non-thinking)</div>
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <textarea id="input" rows="1" placeholder="메시지를 입력하세요... (Shift+Enter: 줄바꿈)"
              oninput="autoResize(this)" onkeydown="handleKey(event)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">전송</button>
  </div>
</div>

<script>
const sessionId = Math.random().toString(36).slice(2);
let isStreaming = false;

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}
function newChat() {
  fetch("/reset", { method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({session_id: sessionId}) });
  document.getElementById("messages").innerHTML = "";
}
function scrollToBottom() {
  const m = document.getElementById("messages");
  m.scrollTop = m.scrollHeight;
}

function addMessage(role) {
  const div    = document.createElement("div");
  div.className = "msg " + role;
  const bubble  = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    bubble.appendChild(cursor);
  }
  div.appendChild(bubble);
  document.getElementById("messages").appendChild(div);
  scrollToBottom();
  return bubble;
}

async function sendMessage() {
  if (isStreaming) return;
  const input = document.getElementById("input");
  const text  = input.value.trim();
  if (!text) return;

  addMessage("user").textContent = text;
  input.value = "";
  input.style.height = "auto";

  const bubble = addMessage("assistant");
  isStreaming = true;
  document.getElementById("sendBtn").disabled = true;

  let replyText = "";
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        message: text,
        session_id: sessionId,
        system_prompt: document.getElementById("systemPrompt").value,
        temperature: parseFloat(document.getElementById("temperature").value),
      }),
    });

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === "reply") {
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
          replyText += evt.text;
          bubble.textContent = replyText;
          const c = document.createElement("span");
          c.className = "cursor";
          bubble.appendChild(c);
          scrollToBottom();
        } else if (evt.type === "done") {
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
        } else if (evt.type === "error") {
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
          bubble.textContent = "❌ " + evt.text;
          bubble.style.color = "#f38ba8";
        }
      }
    }
  } catch(e) {
    const cursor = bubble.querySelector(".cursor");
    if (cursor) cursor.remove();
    bubble.textContent = "❌ 연결 오류: " + e.message;
  } finally {
    isStreaming = false;
    document.getElementById("sendBtn").disabled = false;
    document.getElementById("input").focus();
  }
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print(f"접속: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
