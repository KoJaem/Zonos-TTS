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

## 6. 트러블슈팅 (문제 해결)

### 6-1. uv sync 실행 시 SSL 인증서 오류 발생

**증상:**
`uv sync` 실행 중 `ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]` 오류가 발생하며 멈추는 경우.

**원인:**
macOS용 Python이 시스템의 SSL 인증서를 기본적으로 신뢰하지 않아 발생하는 문제입니다.

**해결 방법:**

**방법 1 (권장):** 파이썬 인증서 설치 스크립트 실행
터미널에 아래 명령어를 입력하여 인증서를 설치합니다 (Python 3.12 기준).

```bash
/Applications/Python\ 3.12/Install\ Certificates.command
```

**방법 2 (임시):** 환경변수 설정
위 방법이 안 되거나 파일을 찾을 수 없는 경우, 아래 명령어를 입력하고 다시 `uv sync`를 실행합니다.

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem
uv sync
```
