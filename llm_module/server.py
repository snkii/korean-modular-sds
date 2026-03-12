"""
Gemma-3-27B (llama.cpp) 챗봇 서버
실행 전제: llama-server로 모델이 서빙 중이어야 합니다.

  GGUF=/path/to/gemma-3-27b-it-q4_0.gguf
  llama-server -m $GGUF --alias gemma-3-27b -ngl 99 --split-mode row -c 8192 --port 8000

실행: python3 server.py [--backend http://localhost:8000/v1] [--model gemma-3-27b]
접속: http://localhost:8083
"""

import argparse
import json
import logging
import threading
import uuid

logging.getLogger("werkzeug").setLevel(logging.ERROR)

from flask import Flask, Response, jsonify, render_template_string, request
from openai import OpenAI

# ── 인자 파싱 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="http://localhost:8000/v1",
                    help="OpenAI-compatible API base URL (vLLM / SGLang)")
parser.add_argument("--model", default="gemma-3-27b")
parser.add_argument("--port", type=int, default=8083)
args = parser.parse_args()

client = OpenAI(base_url=args.backend, api_key="EMPTY")

DEFAULT_SYSTEM = "You are a helpful, friendly AI assistant. Please respond in the same language as the user."

# ── 세션 관리 (메모리) ─────────────────────────────────────────────────────────
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
    data = request.get_json(force=True)
    message      = (data.get("message") or "").strip()
    session_id   = data.get("session_id", "default")
    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM)
    enable_think  = bool(data.get("enable_thinking", False))
    temperature   = float(data.get("temperature", 0.7))
    max_tokens    = int(data.get("max_tokens", 8192))

    if not message:
        return jsonify(error="메시지 없음"), 400

    history = get_history(session_id)
    history.append({"role": "user", "content": message})

    messages = [{"role": "system", "content": system_prompt}] + history

    def generate():
        full_reply   = ""
        think_buf    = ""
        reply_buf    = ""
        in_think     = False
        think_done   = False

        try:
            extra = {}
            if not enable_think:
                extra["chat_template_kwargs"] = {"enable_thinking": False}

            stream = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                presence_penalty=1.5,
                extra_body={"top_k": 20, **extra},
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", "") or ""
                if not token:
                    # reasoning_content 필드 지원 (일부 프레임워크)
                    token = getattr(delta, "reasoning_content", "") or ""
                    if token:
                        data_evt = json.dumps({"type": "think", "text": token})
                        yield f"data: {data_evt}\n\n"
                        think_buf += token
                        continue

                full_reply += token

                # <think>...</think> 파싱
                if not think_done:
                    think_buf += token
                    if not in_think and "<think>" in think_buf:
                        in_think = True
                        idx = think_buf.index("<think>") + len("<think>")
                        think_buf = think_buf[idx:]
                    if in_think:
                        if "</think>" in think_buf:
                            end_idx = think_buf.index("</think>")
                            think_text = think_buf[:end_idx]
                            if think_text:
                                data_evt = json.dumps({"type": "think", "text": think_text})
                                yield f"data: {data_evt}\n\n"
                            think_done = True
                            in_think = False
                            reply_start = think_buf[end_idx + len("</think>"):]
                            think_buf = ""
                            if reply_start.strip():
                                data_evt = json.dumps({"type": "reply", "text": reply_start})
                                yield f"data: {data_evt}\n\n"
                                reply_buf += reply_start
                        else:
                            data_evt = json.dumps({"type": "think", "text": token})
                            yield f"data: {data_evt}\n\n"
                    else:
                        # 생각 없이 바로 답변
                        think_done = True
                        data_evt = json.dumps({"type": "reply", "text": token})
                        yield f"data: {data_evt}\n\n"
                        reply_buf += token
                else:
                    data_evt = json.dumps({"type": "reply", "text": token})
                    yield f"data: {data_evt}\n\n"
                    reply_buf += token

            # 히스토리에 assistant 답변 저장 (thinking 제외)
            final_reply = reply_buf.strip() or full_reply.strip()
            history.append({"role": "assistant", "content": final_reply})
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
  <title>Qwen3.5 챗봇</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5;
           display: flex; height: 100vh; overflow: hidden; }

    /* 사이드바 */
    .sidebar { width: 260px; background: #1e1e2e; color: #cdd6f4; display: flex;
               flex-direction: column; padding: 20px 16px; gap: 16px; flex-shrink: 0; }
    .sidebar h3 { font-size: 13px; color: #6c7086; text-transform: uppercase;
                  letter-spacing: .08em; }
    .sidebar label { font-size: 12px; color: #a6adc8; margin-bottom: 4px; display: block; }
    .sidebar textarea { width: 100%; background: #313244; border: 1px solid #45475a;
                        color: #cdd6f4; border-radius: 6px; padding: 8px; font-size: 12px;
                        resize: vertical; min-height: 80px; }
    .sidebar select, .sidebar input[type=range] {
                        width: 100%; background: #313244; border: 1px solid #45475a;
                        color: #cdd6f4; border-radius: 6px; padding: 6px 8px; font-size: 13px; }
    .toggle-row { display: flex; align-items: center; gap: 8px; }
    .toggle { position: relative; width: 36px; height: 20px; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; inset: 0; background: #45475a; border-radius: 20px;
              cursor: pointer; transition: .2s; }
    .slider:before { content: ""; position: absolute; width: 14px; height: 14px;
                     left: 3px; bottom: 3px; background: white; border-radius: 50%;
                     transition: .2s; }
    input:checked + .slider { background: #89b4fa; }
    input:checked + .slider:before { transform: translateX(16px); }
    .toggle-label { font-size: 13px; color: #cdd6f4; }
    .new-chat-btn { background: #89b4fa; color: #1e1e2e; border: none; border-radius: 8px;
                    padding: 10px; font-size: 14px; font-weight: 600; cursor: pointer;
                    margin-top: auto; }

    /* 채팅 영역 */
    .chat-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
    .chat-header { padding: 16px 24px; background: white; border-bottom: 1px solid #e0e0e0;
                   font-size: 16px; font-weight: 600; color: #333; }
    .messages { flex: 1; overflow-y: auto; padding: 20px 24px; display: flex;
                flex-direction: column; gap: 16px; }
    .msg { display: flex; flex-direction: column; max-width: 75%; }
    .msg.user { align-self: flex-end; align-items: flex-end; }
    .msg.assistant { align-self: flex-start; align-items: flex-start; }
    .bubble { padding: 12px 16px; border-radius: 16px; font-size: 15px;
              line-height: 1.6; white-space: pre-wrap; word-break: break-word; }
    .msg.user .bubble    { background: #89b4fa; color: #1e1e2e; border-bottom-right-radius: 4px; }
    .msg.assistant .bubble { background: white; color: #333; border: 1px solid #e0e0e0;
                              border-bottom-left-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    .think-wrap { margin-bottom: 6px; }
    .think-toggle { font-size: 12px; color: #89b4fa; cursor: pointer; user-select: none;
                    background: none; border: none; padding: 2px 0; }
    .think-box { background: #1e1e2e; color: #a6e3a1; font-size: 12px; font-family: monospace;
                 padding: 10px 12px; border-radius: 8px; margin-top: 4px; display: none;
                 max-height: 200px; overflow-y: auto; white-space: pre-wrap; }
    .think-box.open { display: block; }
    .cursor { display: inline-block; width: 2px; height: 1em; background: #89b4fa;
              animation: blink .7s infinite; vertical-align: text-bottom; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

    /* 입력창 */
    .input-area { padding: 16px 24px; background: white; border-top: 1px solid #e0e0e0;
                  display: flex; gap: 12px; align-items: flex-end; }
    .input-area textarea { flex: 1; border: 1px solid #ddd; border-radius: 12px; padding: 12px 16px;
                           font-size: 15px; resize: none; max-height: 160px; outline: none;
                           font-family: inherit; line-height: 1.5; }
    .input-area textarea:focus { border-color: #89b4fa; }
    .send-btn { background: #89b4fa; color: #1e1e2e; border: none; border-radius: 12px;
                padding: 12px 20px; font-size: 15px; font-weight: 600; cursor: pointer; }
    .send-btn:disabled { opacity: .4; cursor: default; }
    .temp-val { font-size: 12px; color: #a6adc8; }
  </style>
</head>
<body>

<div class="sidebar">
  <h3>설정</h3>

  <div>
    <label>시스템 프롬프트</label>
    <textarea id="systemPrompt">당신은 친절하고 유능한 한국어 AI 어시스턴트입니다.</textarea>
  </div>

  <div>
    <label>Temperature <span class="temp-val" id="tempVal">0.70</span></label>
    <input type="range" id="temperature" min="0" max="2" step="0.05" value="0.7"
           oninput="document.getElementById('tempVal').textContent=parseFloat(this.value).toFixed(2)">
  </div>

  <div class="toggle-row">
    <label class="toggle">
      <input type="checkbox" id="enableThink">
      <span class="slider"></span>
    </label>
    <span class="toggle-label">생각 모드 (Thinking)</span>
  </div>

  <button class="new-chat-btn" onclick="newChat()">＋ 새 대화</button>
</div>

<div class="chat-wrap">
  <div class="chat-header">Qwen3.5-35B-A3B</div>
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
  fetch("/reset", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({session_id: sessionId})
  });
  document.getElementById("messages").innerHTML = "";
}

function addMessage(role) {
  const div = document.createElement("div");
  div.className = "msg " + role;

  let thinkWrap = null, thinkBox = null, bubble = null;

  if (role === "assistant") {
    thinkWrap = document.createElement("div");
    thinkWrap.className = "think-wrap";
    thinkWrap.style.display = "none";

    const btn = document.createElement("button");
    btn.className = "think-toggle";
    btn.textContent = "▶ 생각 과정 보기";
    thinkBox = document.createElement("div");
    thinkBox.className = "think-box";
    btn.onclick = () => {
      const open = thinkBox.classList.toggle("open");
      btn.textContent = (open ? "▼ " : "▶ ") + "생각 과정 보기";
    };
    thinkWrap.appendChild(btn);
    thinkWrap.appendChild(thinkBox);
    div.appendChild(thinkWrap);
  }

  bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    bubble.appendChild(cursor);
  }
  div.appendChild(bubble);
  document.getElementById("messages").appendChild(div);
  scrollToBottom();
  return { div, bubble, thinkWrap, thinkBox };
}

function scrollToBottom() {
  const m = document.getElementById("messages");
  m.scrollTop = m.scrollHeight;
}

async function sendMessage() {
  if (isStreaming) return;
  const input = document.getElementById("input");
  const text = input.value.trim();
  if (!text) return;

  // 사용자 메시지 표시
  const { bubble: userBubble } = addMessage("user");
  userBubble.textContent = text;
  input.value = "";
  input.style.height = "auto";

  // 어시스턴트 메시지 준비
  const { bubble, thinkWrap, thinkBox } = addMessage("assistant");

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
        enable_thinking: document.getElementById("enableThink").checked,
        temperature: parseFloat(document.getElementById("temperature").value),
      }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const evt = JSON.parse(line.slice(6));

        if (evt.type === "think") {
          thinkWrap.style.display = "";
          thinkBox.textContent += evt.text;
          thinkBox.scrollTop = thinkBox.scrollHeight;
        } else if (evt.type === "reply") {
          // 커서 제거 후 텍스트 추가
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
          replyText += evt.text;
          bubble.textContent = replyText;
          const newCursor = document.createElement("span");
          newCursor.className = "cursor";
          bubble.appendChild(newCursor);
          scrollToBottom();
        } else if (evt.type === "done") {
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
        } else if (evt.type === "error") {
          const cursor = bubble.querySelector(".cursor");
          if (cursor) cursor.remove();
          bubble.textContent = "❌ 오류: " + evt.text;
          bubble.style.color = "#f38ba8";
        }
      }
    }
  } catch (e) {
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
    print(f"백엔드: {args.backend}")
    print(f"모델:   {args.model}")
    print(f"접속:   http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
