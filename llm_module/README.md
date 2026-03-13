# LLM Module — Qwen3.5-35B-A3B 챗봇 서버

**Qwen3.5-35B-A3B (Q4_K_S GGUF)** 모델을 llama.cpp로 로드해 OpenAI 호환 API로 서빙하는 챗봇 서버입니다.
논-띵킹(non-thinking) 모드 기본, 8×RTX 2080 Ti GPU 분산 추론.

---

## 아키텍처

```
브라우저 (Flask Web UI)
  └─ POST /chat  (SSE 스트리밍)
       └─ Flask server.py (port 8083)
            └─ OpenAI API (http://127.0.0.1:8180/v1)
                 └─ llama-server subprocess
                      └─ Qwen3.5-35B-A3B-Q4_K_S.gguf
                           └─ 8× CUDA GPU (tensor split)
```

---

## 디렉토리 구조

```
korean-modular-sds/
└── llm_module/
    └── server.py
```

llama-server 바이너리는 별도 빌드 필요 (아래 참고).

---

## 환경 설정

### 1. Conda 가상환경

```bash
conda create -n korean-modular-sds python=3.11 -y
conda activate korean-modular-sds
```

### 2. Python 패키지 설치

```bash
pip install flask openai huggingface_hub
```

### 3. llama-server 빌드 (CUDA)

`qwen35moe` 아키텍처 지원을 위해 **2026년 3월 이후** llama.cpp 소스가 필요합니다.

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

빌드 완료 후 `build/bin/llama-server` 생성됩니다.

### 4. 모델 다운로드

서버 첫 실행 시 HuggingFace Hub에서 자동 다운로드됩니다 (~20GB).
또는 수동 다운로드:

```bash
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q4_K_S.gguf
```

---

## server.py 경로 설정

`server.py` 상단의 두 상수를 환경에 맞게 수정합니다.

```python
LLAMA_SERVER_BIN = "/path/to/llama.cpp/build/bin/llama-server"

MODEL_PATH = (
    "/home/user/.cache/huggingface/hub/"
    "models--unsloth--Qwen3.5-35B-A3B-GGUF/blobs/<hash>"
)
```

모델 파일 경로 확인:

```bash
huggingface-cli scan-cache | grep Qwen3.5-35B-A3B
```

---

## 실행

```bash
conda activate korean-modular-sds
cd /path/to/korean-modular-sds/llm_module
python server.py
```

서버 기동 후 브라우저에서 접속:

```
http://localhost:8083
```

llama-server 로그 확인:

```bash
tail -f /tmp/llama-server.log
```

### 옵션

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `8083` | Flask 웹 UI 포트 |
| `--ctx` | `8192` | 컨텍스트 길이 |

---

## 주요 파라미터

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `n_gpu_layers` | `-1` | 모든 레이어 GPU 오프로드 |
| `tensor_split` | `1,1,1,1,1,1,1,1` | 8 GPU 균등 분산 |
| `temperature` | `0.7` | UI 슬라이더로 조정 가능 |
| `top_p` | `0.8` | nucleus sampling |
| `top_k` | `20` | top-k sampling |
| `enable_thinking` | `False` | 논-띵킹 모드 (빠른 응답) |
