"""
Gemma-3-27B (llama-cpp-python) 챗봇 서버
GGUF 파일을 직접 로드해서 서버 내에서 추론합니다.

실행: python3 server.py [--model /path/to/model.gguf] [--port 8083]
접속: http://localhost:8083
"""

import argparse
import json
import logging
import threading

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, Response, jsonify, render_template_string, request
from llama_cpp import Llama

GGUF_DEFAULT = (
    "/home/sukim/.cache/huggingface/hub/"
    "models--google--gemma-3-27b-it-qat-q4_0-gguf/snapshots/"
    "17cf0f6ad611f1a57a1640daa57eb427d6e67ed6/gemma-3-27b-it-q4_0.gguf"
)

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=GGUF_DEFAULT, help="GGUF 파일 경로")
parser.add_argument("--ctx",   type=int, default=8192,  help="컨텍스트 길이")
parser.add_argument("--port",  type=int, default=8083)
args = parser.parse_args()

# ── 모델 로드 (8 GPU에 분산) ───────────────────────────────────────────────────
print(f"모델 로딩 중: {args.model}")
llm = Llama(
    model_path=args.model,
    n_gpu_layers=-1,          # 모든 레이어를 GPU로
    tensor_split=[1]*8,       # 8 GPU 균등 분산
    n_ctx=args.ctx,
    n_batch=512,
    verbose=False,
)
print("모델 로드 완료")

# inference는 한 번에 하나만 (llama_cpp 내부 상태 보호)
_infer_lock = threading.Lock()

DEFAULT_SYSTEM = "You are a helpful, friendly AI assistant. Please respond in the same language as the user."

# ── 세션 관리 ──────────────────────────────────────────────────────────────────
_sessions: dict[str, list[dict]] = {}
_sessions_lock = threading.Lock()

def get_history(session_id: str) -> list[dict]:
    with _sessions_lock:
        return _sessions.setdefault(session_id, [])

def reset_history(session_id: str):
    with _sessions_lock:
        _sessions[session_id] = []

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
        try:
            with _infer_lock:
                stream = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=40,
                    repeat_penalty=1.1,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content") or ""
                    if token:
                        reply_buf += token
                        yield f"data: {json.dumps({'type': 'reply', 'text': token})}\n\n"

            history.append({"role": "assistant", "content": reply_buf.strip()})
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
  <title>Gemma-3-27B 챗봇</title>
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
  <div class="chat-header">Gemma-3-27B</div>
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
