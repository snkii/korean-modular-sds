# TTS Module — 실시간 한국어 TTS 서버

브라우저에서 텍스트를 입력하면 **문장 단위로 즉시 합성 & 재생**합니다.
[Supertonic](https://github.com/supertonic/supertonic) 기반이며, 첫 문장이 합성되는 즉시 재생을 시작하고 이후 문장은 큐에서 이어서 재생됩니다.

---

## 아키텍처

```
브라우저 텍스트 입력
  └─ POST /synthesize → job_id
       └─ 백그라운드 스레드: 문장 분리 → Supertonic TTS (문장별)
            └─ GET /stream/<job_id>  (SSE)
                 └─ base64 WAV 청크 전송 (문장 완료될 때마다)
                      └─ 브라우저 Web Audio API: 도착 즉시 디코드 & 재생 큐
```

---

## 디렉토리 구조

```
korean-modular-sds/
└── tts_module/
    └── server.py
```

---

## 환경 설정

### 1. Conda 환경 생성

`supertonic` 환경이 이미 있다면 그대로 복제합니다.

```bash
conda create --name korean-modular-sds --clone supertonic -y
conda activate korean-modular-sds
```

`supertonic` 환경이 없다면 새로 생성합니다.

```bash
conda create -n korean-modular-sds python=3.11 -y
conda activate korean-modular-sds
pip install supertonic
```

### 2. 추가 패키지 설치

```bash
pip install flask
```

---

## 실행

```bash
conda activate korean-modular-sds
cd /path/to/korean-modular-sds/tts_module
python server.py
```

서버 기동 후 브라우저에서 접속:

```
http://localhost:8082
```

---

## 사용 방법

1. 텍스트 입력창에 읽을 내용을 입력
2. 목소리 / 언어 / 속도 선택
3. **▶ 읽기** 버튼 클릭 → 첫 문장 합성 완료 즉시 재생 시작
4. **■ 중지** 버튼으로 즉시 정지

---

## 지원 목소리

| 구분 | 이름 |
|------|------|
| 남성 | M1, M2, M3, M4, M5 |
| 여성 | F1, F2, F3, F4, F5 |

---

## 주요 파라미터

`server.py` 상단에서 조정할 수 있습니다.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `DEFAULT_VOICE` | `"M1"` | 기본 목소리 |
| `DEFAULT_LANG` | `"ko"` | 기본 언어 (`ko` / `en`) |
| `DEFAULT_SPEED` | `1.05` | 기본 속도 (0.7 ~ 1.5) |
