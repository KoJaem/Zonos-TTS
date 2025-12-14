# Zonos (Zyphra/Zonos) – macOS 사용 가이드

이 문서는 **Zonos를 처음 설치할 때(초기 설정)** 와  
**컴퓨터를 껐다 켠 후 다시 실행할 때** 필요한 최소 절차만 정리한 문서이다.  
(macOS 기준, uv + Gradio 방식)

---

## 1. Zonos 초기 설정 (최초 1회만)

### 1-1. 시스템 의존성 설치 (macOS)

Zonos는 발음 변환을 위해 espeak-ng가 필요하다.

brew install espeak-ng

설치 확인:

espeak-ng --version

---

### 1-2. 프로젝트 다운로드

git clone https://github.com/Zyphra/Zonos.git
cd Zonos

---

### 1-3. Python 환경 준비 (uv 사용)

uv는 가상환경과 의존성을 자동으로 관리해준다.

python3 -m pip install -U uv

의존성 설치 및 프로젝트 등록:

uv sync
uv pip install -e .

---

### 1-4. 정상 동작 테스트 (선택)

샘플 음성 생성:

uv run sample.py

성공 시 프로젝트 폴더에 sample.wav 파일이 생성된다.

---

### 1-5. Gradio Web UI 실행

웹 인터페이스 실행:

uv run gradio_interface.py

터미널에 표시되는 주소(보통 아래 주소)에 접속:

http://127.0.0.1:7860

이후 텍스트 입력 또는 음성 레퍼런스를 넣어 음성 생성 가능.

---

## 2. 컴퓨터 재부팅 후 다시 실행하는 방법

초기 설정이 끝난 이후에는 **재설치할 필요 없음**.

### 2-1. 기본 실행 루트 (가장 많이 쓰는 방법)

cd Zonos
uv run gradio_interface.py

→ 웹 브라우저에서 UI 접속 후 바로 사용 가능

---

### 2-2. 한 번만 음성 생성하고 종료할 때

cd Zonos
uv run sample.py

---

## 3. 업데이트했을 때만 필요한 작업

Zonos 저장소를 업데이트했을 경우에만 실행:

cd Zonos
git pull
uv sync

평소 재부팅 후에는 실행할 필요 없음.

---

## 4. 재부팅 후 빠른 체크리스트

Zonos 폴더 존재 여부:
ls ~/Zonos

espeak-ng 설치 확인:
espeak-ng --version

uv 설치 확인:
uv --version

정상 실행:
uv run gradio_interface.py

---

## 5. 요약

- 초기 설정: 설치 / clone / uv sync → 단 1회
- 재부팅 후: cd Zonos → uv run gradio_interface.py
- Web UI 주소: http://127.0.0.1:7860
- macOS에서는 Docker보다 Python + uv 방식이 안정적

이 문서 하나만 있으면 Zonos 재사용 가능.

---

## 6. 로컬 모델 수동 설정 (선택)

인터넷 연결 없이 사용하거나, 모델을 직접 관리하고 싶다면 아래 절차를 따르세요.

### 6-1. 환경 변수 설정 (필수)
`.env` 파일을 사용하여 오프라인 모드를 제어할 수 있습니다.

1. 프로젝트 루트에 `.env` 파일 생성 (이미 존재할 수 있음)
2. `useOffline` 값 설정:

```properties
# 오프라인 강제 모드 (로컬 파일만 사용, 없으면 에러)
useOffline=true

# 온라인/하이브리드 모드 (로컬 파일 확인 후 없으면 다운로드)
useOffline=false
```

### 6-2. 폴더 생성
프로젝트 폴더(Zonos) 안에 `models` 폴더를 생성하고, 그 안에 아래 구조대로 폴더를 만드세요.

```
Zonos
└── models
    └── Zyphra
        ├── Zonos-v0.1-transformer
        └── Zonos-v0.1-speaker-embedding
```

### 6-2. 파일 다운로드 및 배치
각 폴더에 맞는 파일을 다운로드해서 넣어주세요.

**1. 메인 모델 (Zonos-v0.1-transformer)**
*   **위치**: `Zonos/models/Zyphra/Zonos-v0.1-transformer/`
*   **필수 파일**:
    https://huggingface.co/Zyphra/Zonos-v0.1-transformer/tree/main
    에서 `config.json` 과 `model.safetensors` 다운로드

**2. 스피커 임베딩 모델 (목소리 복제용)**
*   **위치**: `Zonos/models/Zyphra/Zonos-v0.1-speaker-embedding/`
*   **필수 파일** (OS 무관, 모든 OS 공통):
    https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding/tree/main 에서 `ResNet293_SimAM_ASP_base.pt` 과 `ResNet293_SimAM_ASP_base_LDA-128.pt` 다운로드

**3. DAC 모델 (오디오 처리용)**
*   **위치**: `Zonos/models/descript/dac_44khz/`
*   **필수 파일**:
    https://huggingface.co/descript/dac_44khz/tree/main 에서 `config.json`, `model.safetensors`, `preprocessor_config.json` 다운로드

경로는 다음과 같습니다.
```
Project Root
└── models
    ├── Zyphra
    │   ├── Zonos-v0.1-transformer
    │   │   ├── config.json
    │   │   └── model.safetensors
    │   └── Zonos-v0.1-speaker-embedding
    │       ├── ResNet293_SimAM_ASP_base.pt
    │       └── ResNet293_SimAM_ASP_base_LDA-128.pt
    └── descript
        └── dac_44khz
            ├── config.json
            ├── model.safetensors
            └── preprocessor_config.json
```


## 허깅페이스 캐싱 지우기
rm -rf ~/.cache/huggingface/hub
