# STT Module — 실시간 한국어 음성 전사 서버

브라우저 마이크로 입력된 음성을 실시간으로 한국어 텍스트로 변환합니다.
**Denoiser(Facebook) + Conformer-CTC-BPE(NeMo)** 기반이며, VAD로 발화 구간을 감지해 확정 텍스트를 출력합니다.

---

## 아키텍처

```
브라우저 마이크
  └─ MediaRecorder (200ms 청크, webm)
       └─ POST /transcribe
            └─ ffmpeg: webm → 16kHz mono wav
                 └─ RMS VAD
                      ├─ 음성: 누적 버퍼 → Denoiser → Conformer-CTC → partial SSE
                      └─ 묵음 2청크: commit SSE → 버퍼 초기화
```

---

## 디렉토리 구조

```
korean-modular-sds/          ← 이 레포 루트
├── stt_module/
│   └── server.py
└── Korean-Streaming-ASR/    ← 별도 클론 필요 (아래 참고)
    ├── src/
    │   └── denoiser/        ← Facebook Denoiser 소스
    └── checkpoint/
        ├── denoiser.th      ← Denoiser 체크포인트
        └── Conformer-CTC-BPE.nemo  ← ASR 모델 체크포인트
```

> `server.py`는 실행 위치 기준으로 `../Korean-Streaming-ASR`을 자동으로 찾습니다.

---

## 환경 설정

### 1. 시스템 패키지

```bash
sudo apt-get install ffmpeg
```

### 2. Conda 가상환경 생성

```bash
conda create -n korean-stt python=3.10 -y
conda activate korean-stt
```

### 3. PyTorch 설치 (CUDA 버전에 맞게)

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. NeMo Toolkit 설치

```bash
pip install nemo_toolkit[asr]
```

### 5. 나머지 패키지 설치

```bash
pip install flask librosa numpy
```

---

## Korean-Streaming-ASR 레포 및 체크포인트 확보

### 레포 클론

```bash
# 이 레포 루트(korean-modular-sds/)와 같은 위치에 클론
cd /path/to/korean-modular-sds/..
git clone https://github.com/hyung8758/Korean-Streaming-ASR.git
```

### Denoiser 의존성 설치

```bash
cd Korean-Streaming-ASR
pip install -e src/denoiser
```

### 체크포인트 파일

`Korean-Streaming-ASR/checkpoint/` 디렉토리에 아래 두 파일을 준비합니다.

| 파일 | 설명 |
|------|------|
| `denoiser.th` | Facebook Denoiser 사전학습 모델 |
| `Conformer-CTC-BPE.nemo` | NeMo 한국어 Conformer-CTC 모델 |

- **denoiser.th**: [Korean-Streaming-ASR 레포 releases](https://github.com/hyung8758/Korean-Streaming-ASR) 또는 Facebook Denoiser에서 제공하는 `dns48` 체크포인트 사용
- **Conformer-CTC-BPE.nemo**: [NVIDIA NeMo 한국어 모델](https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularDESC&query=korean) 또는 직접 학습한 `.nemo` 파일

---

## 실행

```bash
conda activate korean-stt
cd /path/to/korean-modular-sds/stt_module
python server.py
```

서버 기동 후 브라우저에서 접속:

```
http://localhost:8081
```

마이크 권한을 허용하면 즉시 실시간 전사가 시작됩니다.

---

## 주요 파라미터

`server.py` 상단에서 조정할 수 있습니다.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `SILENCE_RMS_THRESH` | `0.005` | VAD RMS 임계값 (낮을수록 민감) |
| `SILENCE_CHUNKS_END` | `2` | 묵음 청크 수 → 발화 확정 (2 = 400ms) |
| `DENOISE_DRY` | `0.05` | Denoiser 혼합 비율 (0=완전 denoised, 1=원본) |
