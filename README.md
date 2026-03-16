# Korean / English Modular SDS

실시간 음성 대화 시스템 (Spoken Dialogue System) — 한국어 / 영어 지원

**STT → LLM → TTS** 파이프라인을 브라우저에서 완전히 동작시킵니다.
마이크로 말하면 → AI가 생각하고 → 음성으로 대답합니다.

| 언어 | 실행 파일 | STT 모델 |
|------|-----------|----------|
| 한국어 | `run.py` | NeMo Conformer-CTC-BPE (로컬 `.nemo`) |
| English | `run_eng.py` | NeMo `stt_en_conformer_ctc_large` (자동 다운로드) |

---

## 아키텍처

```
브라우저 마이크
  └─ 200ms WebM 청크 → STT (port 8081, Conformer-CTC-BPE)
       └─ commit 이벤트 (확정 텍스트)
            └─ LLM (port 8083, Qwen3.5-35B-A3B)
                 └─ 토큰 스트리밍 → 문장 단위로 분리
                      └─ TTS (port 8082, Supertonic)
                           └─ base64 WAV → Web Audio API 순차 재생
```

통합 UI 서버 (`run.py`, port 8090)가 세 모듈을 subprocess로 실행하고
모든 API를 브라우저에 프록시합니다.

---

## 사전 요구사항

| 항목 | 버전 / 조건 |
|------|------------|
| OS | Ubuntu 20.04 이상 |
| GPU | CUDA 지원 GPU (모델 크기에 따라 VRAM 필요, 권장: 8× RTX 2080 Ti 이상) |
| CUDA | 12.4 이상 |
| conda | Miniconda 또는 Anaconda |
| ffmpeg | `sudo apt-get install ffmpeg` |
| git | 기본 설치 |

---

## 1단계 — 레포 클론

```bash
git clone https://github.com/<your-org>/korean-modular-sds.git
cd korean-modular-sds
```

---

## 2단계 — Korean-Streaming-ASR 준비

> **한국어 버전(`run.py`)에만 필요합니다.**
> 영어 버전(`run_eng.py`)은 NeMo가 모델을 자동 다운로드하므로 이 단계를 건너뜁니다.

STT 모듈은 Facebook Denoiser + NeMo Conformer-CTC 모델을 사용합니다.
레포를 `korean-modular-sds/`와 **같은 레벨**에 클론한 뒤 심볼릭 링크를 만듭니다.

```bash
# 프로젝트 루트의 상위 디렉토리로 이동
cd ..

# 레포 클론
git clone https://github.com/SUNGBEOMCHOI/Korean-Streaming-ASR.git

# 다시 프로젝트 루트로 돌아와서 심볼릭 링크 생성
cd korean-modular-sds
ln -s ../Korean-Streaming-ASR Korean-Streaming-ASR
```

### 체크포인트 파일 배치

아래 Google Drive 폴더에서 `denoiser.th`와 `Conformer-CTC-BPE.nemo`를 다운로드합니다.

> [Google Drive 다운로드 폴더](https://drive.google.com/drive/folders/1Adv8kYXV1XGGoLY1XA36EI38kfk0r0WZ?usp=drive_link)

다운로드한 파일을 아래 위치에 배치합니다.

```
Korean-Streaming-ASR/checkpoint/
├── denoiser.th              ← Facebook Denoiser 가중치
└── Conformer-CTC-BPE.nemo   ← NeMo 한국어 ASR 모델
```

---

## 3단계 — llama-server 빌드

LLM 모듈은 `llama-server` 바이너리를 subprocess로 실행합니다.
`qwen35moe` 아키텍처 지원을 위해 **2026년 3월 이후** 버전의 llama.cpp가 필요합니다.

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

빌드 완료 후 `build/bin/llama-server` 바이너리가 생성됩니다.

---

## 4단계 — Conda 환경 생성 및 패키지 설치

모든 모듈이 **`korean-modular-sds`** 환경 하나를 공용으로 사용합니다.

```bash
conda create -n korean-modular-sds python=3.11 -y
conda activate korean-modular-sds
```

### 패키지 설치

```bash
# 웹 서버 / LLM 클라이언트
pip install flask openai huggingface_hub requests

# TTS
pip install supertonic

# STT — PyTorch (CUDA 12.4 기준, 버전에 맞게 조정)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# STT — NeMo ASR
pip install nemo_toolkit[asr]

# STT — VAD
pip install silero-vad

# STT — Denoiser 런타임 의존성
# (denoiser 패키지 자체는 설치하지 않음 — hydra-core/omegaconf 버전 충돌)
pip install julius sounddevice pystoi librosa

# 오디오 변환
sudo apt-get install -y ffmpeg
```

> **주의**: `denoiser`를 pip로 설치하면 `hydra-core` / `omegaconf` 버전이 NeMo와 충돌합니다.
> Denoiser는 `Korean-Streaming-ASR/src` 경로를 sys.path로 직접 추가해 import합니다 (server.py가 자동 처리).

---

## 5단계 — LLM 모델 다운로드

서버 첫 실행 시 HuggingFace Hub에서 자동 다운로드됩니다 (약 20GB).
또는 수동으로 미리 다운로드:

```bash
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-Q4_K_S.gguf
```

---

## 6단계 — llm_module/server.py 경로 설정

`llm_module/server.py` 상단의 두 상수를 환경에 맞게 수정합니다.

```python
# llama-server 바이너리 경로 (3단계에서 빌드한 경로)
LLAMA_SERVER_BIN = "/path/to/llama.cpp/build/bin/llama-server"

# 모델 파일 경로 (HuggingFace 캐시 내 blob 파일)
MODEL_PATH = (
    "/home/<user>/.cache/huggingface/hub/"
    "models--unsloth--Qwen3.5-35B-A3B-GGUF/blobs/<hash>"
)
```

모델 파일의 실제 경로 확인:

```bash
huggingface-cli scan-cache | grep Qwen3.5-35B-A3B
# 또는
find ~/.cache/huggingface -name "*.gguf" 2>/dev/null
```

---

## 실행

### 한국어 버전

```bash
conda activate korean-modular-sds
cd /path/to/korean-modular-sds
python run.py --port 8080
```

### 영어 버전

```bash
conda activate korean-modular-sds
cd /path/to/korean-modular-sds
python run_eng.py --port 8080
```

> **영어 버전 첫 실행 시**: NeMo가 `nvidia/stt_en_conformer_ctc_large` (~500 MB)를
> NGC에서 자동 다운로드합니다. 이후에는 캐시에서 바로 로드됩니다.

브라우저에서 접속:

```
http://<서버IP>:8080
```

### 시작 순서

`run.py` / `run_eng.py`가 세 모듈을 자동으로 병렬 시작하고 준비될 때까지 대기합니다.

```
[1/3] STT 모듈  — 약 30~60초
[2/3] TTS 모듈  — 약 10~20초
[3/3] LLM 모듈  — 약 60~120초 (llama-server + 모델 로드)
```

UI 상단 배너에서 STT / TTS / LLM 각 모듈의 준비 상태를 실시간으로 확인할 수 있습니다.

### 실행 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `8080` | UI 서버 포트 |

---

## 포트 구성

| 포트 | 모듈 | 설명 |
|------|------|------|
| 8080 | `run.py` / `run_eng.py` | 통합 UI 서버 (외부 노출) |
| 8081 | STT | 전사 서버 (`server.py` 또는 `server_en.py`) |
| 8082 | TTS | Supertonic 합성 서버 |
| 8083 | LLM | Flask 채팅 서버 |
| 8180 | llama-server | 내부 OpenAI 호환 API (외부 비노출) |

---

## 사용 방법

1. 브라우저 접속 후 모든 모듈이 로드될 때까지 대기 (상단 배너 확인)
2. **🎤 버튼** 클릭 → 녹음 시작 (빨간색으로 변함)
3. 말하기 → 하단에 인식 중인 텍스트가 실시간으로 표시됨
4. 말을 멈추면 (~400ms 묵음) 텍스트가 확정되어 AI에게 전달
5. AI 응답이 채팅창에 스트리밍되고, 문장 단위로 TTS 재생
6. 새 발화 시 현재 AI 응답/TTS가 즉시 중단되고 새 입력 처리
7. **새 대화** 버튼으로 대화 히스토리 초기화

---

## 모듈별 상세

| 모듈 | README |
|------|--------|
| STT (음성 인식) | [stt_module/README.md](stt_module/README.md) |
| TTS (음성 합성) | [tts_module/README.md](tts_module/README.md) |
| LLM (언어 모델) | [llm_module/README.md](llm_module/README.md) |

---

## 문제 해결

### STT 모듈이 시작되지 않는 경우

```bash
# Korean-Streaming-ASR 심볼릭 링크 확인
ls korean-modular-sds/Korean-Streaming-ASR/checkpoint/

# nemo 설치 확인
conda run -n korean-modular-sds python -c "import nemo; print(nemo.__version__)"
```

### LLM 모듈이 시작되지 않는 경우

```bash
# llama-server 로그 확인
tail -f /tmp/llama-server.log

# 바이너리 경로 확인 (llm_module/server.py의 LLAMA_SERVER_BIN)
ls -la /path/to/llama.cpp/build/bin/llama-server
```

### 포트 충돌

```bash
# 사용 중인 포트 확인
ss -tlnp | grep -E "808[0-3]|8180"

# 다른 포트로 실행
python run.py --port 9090

# run_eng.py 실행 전 STT 서버가 남아 있는 경우
fuser -k 8081/tcp
```
